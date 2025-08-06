import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy

from functools import partial

import numpy as np

import torch

import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import pandas as pd
from tqdm import trange
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from train import FedAVG
from funavg import LabelTransform
from train import get_labels, get_plans_and_dataset_json, get_trainer, configure_dropout

from config import (
    FIVAConfig,
    LABEL_SUBSETS,
    # N_ITERATIONS_PER_FED_EPOCH_2D,
    SLICES_PER_DATASET,
    SAMPLES_PER_DATASET,
)


class FIVA:
    def __init__(
        self,
        n_samples_per_client: np.ndarray,
        sample_state_dict: dict,
        mode: str = "full",
    ):
        total_samples = np.sum(n_samples_per_client)
        relative_samples = n_samples_per_client / total_samples

        self.rel_sample_sizes = relative_samples

        self.client_weights = [
            {k: torch.ones_like(v) * rel_sample for k, v in sample_state_dict.items()}
            for rel_sample in relative_samples
        ]

        # this should not have any segmentation head
        self.layer_names = list(sample_state_dict.keys())
        self.mode = mode
        self.epsilon = 1e-5

    def snapshot(self):
        for i, cw in enumerate(self.client_weights):
            if not os.path.exists(f"weights/"):
                os.makedirs(f"weights/")
            if os.path.exists(f"weights/client_weights_{i:03.0f}.pt"):
                os.rename(
                    f"weights/client_weights_{i:03.0f}.pt",
                    f"weights/client_weights_previous_{i:03.0f}.pt",
                )
            torch.save(cw, f"weights/client_weights_{i:03.0f}.pt")

    def __call__(self, client_param_dicts: list, variance_param_dicts: list):

        # sum_inv_var = {k: 0 for k in variance_param_dict[0].keys()}
        sum_inv_var = {k: 0 for k in self.layer_names}

        sum_inv_var_for_weights = {k: 0 for k in self.layer_names}

        # Use net aggregate variance mean per model
        if self.mode == "scalar":
            flat_vars_mean = [
                parameters_to_vector(list(var_dict.values())).mean()
                for var_dict in variance_param_dicts
            ]

            for i, scalar_var in enumerate(flat_vars_mean):
                for lyr in self.layer_names:
                    sum_inv_var_for_weights[lyr] += (
                        self.rel_sample_sizes[i] * 1 / scalar_var
                    )

            for i, scalar_var in enumerate(flat_vars_mean):
                for lyr in self.layer_names:
                    self.client_weights[i][lyr] = (
                        self.rel_sample_sizes[i]
                        * (1 / scalar_var)
                        / sum_inv_var_for_weights[lyr]
                    )
            for sample_size, var_dict in zip(
                self.rel_sample_sizes, variance_param_dicts
            ):
                for lyr in self.layer_names:
                    sum_inv_var[lyr] += sample_size * 1 / var_dict[lyr]

        elif self.mode == "full" or self.mode == "layer":
            for sample_size, var_dict in zip(
                self.rel_sample_sizes, variance_param_dicts
            ):
                for lyr in self.layer_names:
                    sum_inv_var[lyr] += sample_size * 1 / (var_dict[lyr] + self.epsilon)
                    if self.mode == "full":
                        # FULL VARIANCE
                        sum_inv_var_for_weights[lyr] = sum_inv_var[lyr]
                    elif self.mode == "layer":
                        # MEAN VARIANCE LAYER WISE
                        sum_inv_var_for_weights[lyr] += (
                            sample_size * 1 / (var_dict[lyr].mean() + self.epsilon)
                        )

            # # Compute weights for each client
            for i, var_dict in enumerate(variance_param_dicts):
                for lyr in self.layer_names:
                    if self.mode == "full":
                        self.client_weights[i][lyr] = (
                            self.rel_sample_sizes[i]
                            * (1 / (var_dict[lyr] + self.epsilon))
                            / sum_inv_var_for_weights[lyr]
                        )
                    elif self.mode == "layer":
                        self.client_weights[i][lyr] = (
                            self.rel_sample_sizes[i]
                            * (1 / (var_dict[lyr].mean() + self.epsilon))
                            / sum_inv_var_for_weights[lyr]
                        )

        server_state_dict = OrderedDict([])

        for layer_name in self.layer_names:

            ws = [msd[layer_name] for msd in client_param_dicts]
            new_w = torch.zeros_like(ws[0])

            # self.client_weights = [
            #     {k: v[layer_name] for k, v in client_weight.items()}
            #     for client_weight in self.client_weights
            # ]
            layer_weights = [cw[layer_name] for cw in self.client_weights]

            for w, cw in zip(ws, layer_weights):
                new_w += (cw * w).type(w.dtype)

            server_state_dict.update({layer_name: new_w})
            # server_variance_update.update({layer_name: 1/sum_inv_var[layer_name]})

        self.snapshot()
        return server_state_dict, sum_inv_var


def get_parameters_stats(param_dict: dict):

    flat_server_variance = parameters_to_vector(
        list(param_dict.values())
    )  # type: ignore
    mean_server_variance = flat_server_variance.mean()
    std_server_variance = flat_server_variance.std()
    server_variance_stat = {
        "mean": mean_server_variance.item(),
        "std": std_server_variance.item(),
        "min": flat_server_variance.min().item(),
        "max": flat_server_variance.max().item(),
        "n_outliers": torch.sum(
            torch.abs(flat_server_variance - mean_server_variance)
            > 3 * std_server_variance
        ).item(),
    }

    return server_variance_stat


def copy_param_dict(param_dict: dict, state_dict: dict):
    for k in state_dict.keys():
        if "decoder.encoder" in k:
            k1 = k.replace("decoder.", "")
        else:
            k1 = k
        if "all_modules.1" in k1:
            state_dict[k] = param_dict[k1.replace("all_modules.1", "norm")]
        elif "all_modules.0" in k1:
            state_dict[k] = param_dict[k1.replace("all_modules.0", "conv")]
        else:
            state_dict[k] = param_dict[k1]
    return state_dict


def train_fiva(cfg: FIVAConfig, exp_path: Path, from_ckpt: bool = False):
    # Load configuration
    labels = get_labels(cfg.dataset.ids, LABEL_SUBSETS)
    # Using TotalSegmentator dataset for master trainer, common architecture and plan

    ######## CHANGE AFTER DEBUG #################
    master_plan, master_dataset_json = get_plans_and_dataset_json(
        cfg.master_dataset, plans_identifier="nnUNetPlans"
    )
    master_trainer = get_trainer(
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
        plans=master_plan,
        dataset_json=master_dataset_json,
    )

    get_trainer_partial = partial(
        get_trainer,
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
    )

    client_trainers: list[nnUNetTrainer] = []

    for d_set in labels.keys():
        plans, dataset_json = get_plans_and_dataset_json(d_set, "nnUNetPlans")
        # Supply the same architecture, patch_size and batch_size to all clients

        plans["configurations"][cfg.nnunet.config]["batch_size"] = cfg.train.batch_size
        plans["configurations"][cfg.nnunet.config][
            "patch_size"
        ] = master_trainer.configuration_manager.patch_size
        plans["configurations"][cfg.nnunet.config][
            "batch_dice"
        ] = master_trainer.configuration_manager.batch_dice

        plans["configurations"][cfg.nnunet.config]["architecture"] = deepcopy(
            master_plan["configurations"][cfg.nnunet.config]["architecture"]
        )

        t = get_trainer_partial(plans=plans, dataset_json=dataset_json)
        client_trainers.append(t)

    if cfg.train.dropout_p is not None:
        configure_dropout(master_trainer, cfg.train.dropout_p, cfg.nnunet.config[:2])
        for t in client_trainers:
            configure_dropout(t, cfg.train.dropout_p, cfg.nnunet.config[:2])

    master_trainer.initialize()
    conv_op = getattr(nn, f"Conv{cfg.nnunet.config[:2]}")

    # configure master

    # Checkpoint handling for epochs

    if from_ckpt:
        met = torch.load(exp_path / "train_metrics.pt")
        ds = list(labels.keys())[0]
        n_epochs_trained = len(met[ds]["train_losses"])
        # n_epochs_trained = 100

        master_ckpt = torch.load(
            # exp_path / f"ckpt_server_latest.pt", map_location="cpu"
            exp_path / f"ckpt_server_{cfg.cknum}.pt",
            map_location="cpu",
        )

        sd = master_trainer.network.state_dict()
        for k, v in sd.items():
            if k not in master_ckpt:
                # print(k)
                master_ckpt[k] = v
        master_trainer.network.load_state_dict(master_ckpt)
        # del master_ckpt
    else:
        n_epochs_trained = 0

    t_epochs = []
    # # Configure all the clients
    for t in client_trainers:

        t.on_train_start()
        ## UNCOMMENT IF NEEDED
        # t.network = deepcopy(master_trainer.network)

        dataset_name = t.plans_manager.dataset_name

        ds_labels = labels[dataset_name]
        n_seg_heads = len(ds_labels["wanted_labels_in_dataset"]) + 1

        lt = LabelTransform(**ds_labels)

        if isinstance(t.dataloader_train, SingleThreadedAugmenter):
            t.dataloader_train.data_loader.transforms.transforms.append(lt)  # type: ignore
            t.dataloader_val.data_loader.transforms.transforms.append(lt)  # type: ignore
        elif isinstance(t.dataloader_train, NonDetMultiThreadedAugmenter):
            t.dataloader_train.generator.transforms.transforms.append(lt)  # type: ignore
            t.dataloader_val.generator.transforms.transforms.append(lt)  # type: ignore

        # HACK: To get the dataloader to start after adding the transform
        _ = next(t.dataloader_train)  # type: ignore
        _ = next(t.dataloader_val)  # type: ignore

        t.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
            [
                conv_op(m.in_channels, n_seg_heads, m.kernel_size, stride=m.stride)
                for m in t.network.decoder.seg_layers  # type: ignore
            ]
        ).to(cfg.train.device)

        # FIXME: Change this to use 3D as well when needed
        t.num_iterations_per_epoch = cfg.num_iters[t.plans_manager.dataset_name]
        t.num_val_iterations_per_epoch = t.num_iterations_per_epoch // 5 + 1
        t.num_epochs = cfg.num_rounds
        t.save_every = 1

        t.optimizer, t.lr_scheduler = t.configure_optimizers()

        # Load checkpoint
        if from_ckpt:

            t.load_checkpoint(
                str(
                    exp_path
                    / t.plans_manager.dataset_name
                    / "nnUNetTrainer__nnUNetPlans__2d"
                    / "fold_0"
                    / f"checkpoint_{cfg.cknum}.pth"
                )
            )
            # Load segmentation head
            t_ckpt = torch.load(
                exp_path / f"ckpt_{t.plans_manager.dataset_name}_{cfg.cknum}.pt",
                map_location=cfg.train.device,
                # exp_path / f"ckpt_{t.plans_manager.dataset_name}_100.pt"
            )
            # t_ckpt = {f'decoder.seg_layers.{n}': w for n, w in t_ckpt.items()}
            # t_sd = t.network.state_dict()
            # for k, v in t_sd.items():
            #     if k in master_ckpt and not k.startswith(
            #         "_orig_mod.decoder.seg_layers"
            #     ):
            #         # print(f"Found key : {k}")
            #         t_sd[k] = master_ckpt[k]

            # t.network.load_state_dict(t_sd)
            t.network.decoder.seg_layers.load_state_dict(t_ckpt)  # type: ignore

    if from_ckpt:

        for t in client_trainers:
            t.current_epoch = n_epochs_trained
        master_trainer.current_epoch = n_epochs_trained
    # else:
    #     epochs_trained = 0

    master_trainer.num_val_iterations_per_epoch = (
        master_trainer.num_iterations_per_epoch // 5 + 1
    )
    master_trainer.num_epochs = cfg.num_rounds
    master_trainer.save_every = 1

    master_trainer.optimizer, master_trainer.lr_scheduler = (
        master_trainer.configure_optimizers()
    )

    # Set final layers to identity in master trainer
    master_trainer.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
        [nn.Identity() for _ in client_trainers[0].network.decoder.seg_layers]  # type: ignore
    )

    n_samples_per_client = np.array(
        [SAMPLES_PER_DATASET[t.plans_manager.dataset_name] for t in client_trainers]
    )

    forgetting_factor = 0.95
    # forgetting_factor = 0.7
    # forgetting_factor = 0.9
    # forgetting_factor = 0.1

    # Keeping state dict as separate because it contains copied keys for the same underlying tensors.
    # Use state_dict for model loading and checkpointing and param_dict for computation
    server_state_dict = {
        k: v
        for k, v in master_trainer.network.state_dict().items()  # type: ignore
        if not k.startswith("_orig_mod.decoder.seg_layers")
    }
    global_keys = list(server_state_dict.keys())

    if cfg.grad_mode == "fedavg_fallback":
        wts = n_samples_per_client / np.sum(n_samples_per_client)
        fedavg = FedAVG(wts.tolist())
    else:
        fed_var = FIVA(n_samples_per_client, server_state_dict, cfg.agg_mode)

    # Initialize client variances and server variance to 1
    client_variances: list[dict[str, torch.Tensor]] = []
    for t in client_trainers:
        client_variances.append(
            {
                k: torch.ones_like(v) for k, v in t.network.state_dict().items()  # type: ignore
            }
        )

    server_variance = {k: torch.ones_like(v) for k, v in server_state_dict.items()}

    # flat_server_variance = parameters_to_vector(
    #     list(server_variance.values())
    # )  # type: ignore
    # mean_server_variance = flat_server_variance.mean()
    # std_server_variance = flat_server_variance.std()
    # server_variance_stat = {
    #     'mean':mean_server_variance,
    #     'std': std_server_variance,
    #     'min': flat_server_variance.min(),
    #     'max': flat_server_variance.max(),
    #     'n_outliers': torch.sum(
    #         torch.abs(flat_server_variance - mean_server_variance)
    #         > 3 * std_server_variance
    #     ).item(),
    # }
    # variance_tape = [server_variance_stat]
    variance_tape = [get_parameters_stats(server_variance)]
    unclipped_variance_tape = [get_parameters_stats(server_variance)]
    if from_ckpt:
        # Load variances
        server_variance = torch.load(
            exp_path / "variances" / f"ckpt_server_var_{cfg.cknum}.pt",
            map_location=cfg.train.device,
        )
        for t in client_trainers:
            client_variances[client_trainers.index(t)] = torch.load(
                exp_path
                / "variances"
                / f"ckpt_{t.plans_manager.dataset_name}_var_{cfg.cknum}.pt",
                map_location=cfg.train.device,
            )

    epsilon = 1e-8

    ############## Training loop ###################
    for epoch in range(n_epochs_trained, cfg.num_rounds):

        if epoch == 2:
            print("Stopping training at epoch 2 for debugging")
            break

        # Update client weights with server weights
        for i, t in enumerate(client_trainers):
            t_state_dict = t.network.state_dict()  # type: ignore
            for k, v in server_state_dict.items():
                t_state_dict[k] = v.clone()
            # t_state_dict = copy_param_dict(server_param_dict, t_state_dict)

            t.network.load_state_dict(t_state_dict)  # type: ignore

            for gkey in global_keys:
                client_variances[i][gkey] = server_variance[gkey].clone()

        client_state_dicts = []

        # Each client is represented  by a trainer
        for i, t in enumerate(client_trainers):
            print(
                f"#####################     {t.plans_manager.dataset_name}       ####################"
            )
            t.on_epoch_start()
            t.on_train_epoch_start()

            # TODO: Check what is deep supervision
            t.network.decoder.deep_supervision = True  # type: ignore
            train_outputs = []

            # sampling_batch_ids = t.num_iterations_per_epoch//5
            if cfg.grad_mode == "parameter_variance":
                # Initialize mean and ssd for parameter variance
                param_mean = [p for k, p in t.network.state_dict().items()]
                # param_mean = [p for p in t.network.parameters()]
                gradient_ssd = [
                    torch.zeros_like(p) for k, p in t.network.state_dict().items()
                ]
                param_keys = [k for k in t.network.state_dict().keys()]  # type: ignore

            else:
                gradient_mean = [torch.zeros_like(p) for k, p in t.network.state_dict().items()]  # type: ignore
        
                gradient_ssd = [
                    torch.zeros_like(p) for k, p in t.network.state_dict().items()
                ]
        
                param_keys = [k for k in t.network.state_dict().keys()]  # type: ignore

            # gradient_ssd = [torch.zeros_like(p) for p in t.network.parameters()]  # type: ignore
            # Keep layer wise count of batches to tackle none gradients
            net_batch_count = [1 for _ in gradient_ssd]  # type: ignore

            # prev_weights = list(t.network.parameters())
            for batch_id in trange(t.num_iterations_per_epoch):
                # TODO: Check why batches are handled this way
                batch = next(t.dataloader_train)  # type: ignore

                loss = t.train_step(batch)

                if np.isnan(loss["loss"]):

                    t.print_to_log_file(
                        f"Loss is nan for batch {batch_id} in epoch {epoch}"
                    )
                    exit(-1)
                    break

                if not (
                    cfg.grad_mode == "fedavg_fallback"
                    or cfg.grad_mode == "parameter_variance"
                ):
                    # Using Welford's algorithm to calculate running mean and running variance of gradients
                    # Each element in the list is a parameter tensor corresponding to a particular layer
             

                    batch_gradient = [p.grad for k, p in t.network.state_dict().items()]  # type: ignore

                    # Store the past 3 gradients for each layer for debuggingn
                    # if os.path.exists(exp_path / f"batch_gradient_{t.plans_manager.dataset_name}_latest.pt"):
                    #     os.rename(exp_path / f"batch_gradient_{t.plans_manager.dataset_name}_latest.pt", exp_path / f"batch_gradient_{t.plans_manager.dataset_name}_previous.pt")
                    # torch.save(batch_gradient, exp_path / f"batch_gradient_{t.plans_manager.dataset_name}_latest.pt")

                    new_gradient_mean = []
                    new_gradient_ssd = []

                    # for bc, (new_grad, old_mean, old_ssd) in enumerate(
                    #     zip(batch_gradient, gradient_mean, gradient_ssd)
                    # ):
                    for bc, (new_grad, old_mean, old_ssd) in enumerate(
                        zip(batch_gradient, gradient_mean, gradient_ssd)
                    ):
                        if new_grad is None:
                            new_mean = old_mean
                            new_ssd = old_ssd
                            print(f"None gradient found for {bc} in batch {batch_id}")
                            # t.print_to_log_file(f"None gradient found for {bc} in batch {batch_id}")
                            # torch.any()
                        elif torch.any(torch.isnan(new_grad).any()):
                            new_mean = old_mean
                            new_ssd = old_ssd
                            t.print_to_log_file(
                                f"NaN gradient found for {bc} in batch {batch_id}"
                            )
                        else:
                            # print(
                            #     f"Gradient mean for {bc} in batch {batch_id}: {new_grad.mean()}"
                            # )

                            new_mean = old_mean + (new_grad - old_mean) / (
                                net_batch_count[bc]
                            )
                            new_ssd = old_ssd + (new_grad - old_mean) * (
                                new_grad - new_mean
                            )
                            net_batch_count[bc] += 1

                        new_gradient_mean.append(new_mean)
                        new_gradient_ssd.append(new_ssd)

                    gradient_mean = new_gradient_mean
                    gradient_ssd = new_gradient_ssd

                elif cfg.grad_mode == "parameter_variance":
                    # Compute the variance of parameters directly instead of gradients
                    batch_params = [
                        p for k, p in t.network.state_dict().items()  # type: ignore
                    ]

                    new_param_mean = []
                    new_param_ssd = []

                    for bc, (new_param, old_mean, old_ssd) in enumerate(
                        zip(batch_params, param_mean, gradient_ssd)
                    ):
                        if torch.any(torch.isnan(new_param).any()):
                            new_mean = old_mean
                            new_ssd = old_ssd
                            t.print_to_log_file(
                                f"NaN parameter found for {bc} in batch {batch_id}"
                            )
                        else:
                            new_mean = (
                                old_mean + (new_param - old_mean) / net_batch_count[bc]
                            )
                            new_ssd = old_ssd + (new_param - old_mean) * (
                                new_param - new_mean
                            )
                            net_batch_count[bc] += 1

                        new_param_mean.append(new_mean)
                        new_param_ssd.append(new_ssd)

                    param_mean = new_param_mean
                    gradient_ssd = new_param_ssd

                train_outputs.append(loss)

            # Update client variance

            # gradient_variance = [
            #     grad_ssd / (bc) for bc, grad_ssd in zip(net_batch_count, gradient_ssd)
            # ]

            gradient_variance = {
                k: grad_ssd / (bc)
                for k, bc, grad_ssd in zip(param_keys, net_batch_count, gradient_ssd)
            }

            # for k, v in gradient_variance.items():
            #     ic((k, v.mean(), v.std(), v.min(), v.max()))

            if cfg.grad_mode == "unit_debug":
                print("Unit debug mode")
                pass
                # exit
            elif cfg.grad_mode == "fedavg_fallback":
                pass
            else:
                lr = t.optimizer.param_groups[0]["lr"]

                new_var_dict = {}

                # Normal update rule
                # for j, (lyr, var) in enumerate(client_variances[i].items()):
                for j, (lyr, gradient_var) in enumerate(gradient_variance.items()):

                    ######### TRY 3 VERSIONS OF VARIANCE UPDATE RULES #########
                    var = client_variances[i][lyr]
                    if cfg.grad_mode == "weak":
                        new_var_dict[lyr] = var + (lr**2) * gradient_var  # No cov
                    elif cfg.grad_mode == "strong":

                        new_var_dict[lyr] = (
                            var + (lr**2) * net_batch_count[j] * gradient_var
                        )

                    elif cfg.grad_mode == "parameter_variance":
                        # Use parameter variance instead of gradient variance
                        new_var_dict[lyr] = gradient_var
                        # print(gradient_variance[j].mean())

                    elif cfg.grad_mode == "cov":
                        # TBD
                        # new_var_dict[lyr] = (
                        #     var + (lr**2) * net_batch_count[j] * gradient_var
                        # )  # No cov , strong aggregation

                        # new_var_dict[lyr] = var + ((lr* net_batch_count[j])**2) * gradient_variance[j] - 2*lr*torch.diagonal(torch.cov(torch.tensor([net_batch_count[j] *gradient_mean[j], prev_weights[j]]).T)) # with cov
                        pass

                    # torch.save(gradient_mean[j], exp_path / f"gradient_mean_{t.plans_manager.dataset_name}_{epoch}_{j}.pt")
                    # torch.save(gradient_variance[j], exp_path / f"gradient_variance_{t.plans_manager.dataset_name}_{epoch}_{j}.pt")
                    # torch.save(prev_weights[j], exp_path / f"prev_weights_{t.plans_manager.dataset_name}_{epoch}_{j}.pt")

                for k, v in new_var_dict.items():
                    # print(
                    #     f"Variance for {k} in {t.plans_manager.dataset_name} at epoch {epoch}: {v.mean()}"
                    # )
                    client_variances[i][k] = v
                    # client_variances[i] = new_var_dict

            t.on_train_epoch_end(train_outputs)

            # Validation
            with torch.no_grad():
                t.on_validation_epoch_start()
                val_outputs = []
                for batch_id in trange(t.num_val_iterations_per_epoch):
                    val_outputs.append(t.validation_step(next(t.dataloader_val)))  # type: ignore
                t.on_validation_epoch_end(val_outputs)

            t.on_epoch_end()

            # Get all state_dicts except for the segmentation head
            t_state_dict = {
                k: v
                for k, v in t.network.state_dict().items()  # type: ignore
                if not k.startswith("_orig_mod.decoder.seg_layers")
            }
            client_state_dicts.append(t_state_dict)

        st = time.time()

        # Update server model
        if cfg.grad_mode == "fedavg_fallback":
            print("Fedavg fallback mode")
            server_state_dict = fedavg(client_state_dicts)
        else:
            server_state_dict, sum_inverse_variances = fed_var(
                client_state_dicts, client_variances
            )

        # Update server variance using sum of inverse variances
        if cfg.grad_mode == "unit_debug":
            print("Unit debug mode")
            pass
            # exit
        elif cfg.grad_mode == "fedavg_fallback":
            print("Fedavg fallback mode")
        else:
            candidate_variance = {}
            for k, v in server_variance.items():
                candidate = 1 / (forgetting_factor * 1 / v + sum_inverse_variances[k])
                candidate_variance[k] = candidate
                server_variance[k] = torch.max(
                    candidate, epsilon * torch.ones_like(candidate)
                )

                if torch.any(torch.isnan(server_variance[k]).any()):
                    print("Nan variance found")
                    print("Sum of inverse variances: ", sum_inverse_variances[k].mean())
            unclipped_variance_tape.append(get_parameters_stats(candidate_variance))
            variance_tape.append(get_parameters_stats(server_variance))

        # Run inference on the server model

        # Save metrics
        train_metrics = {
            t.plans_manager.dataset_name: t.logger.my_fantastic_logging
            for t in client_trainers
        }
        train_metrics["GLOBAL_VAL"] = master_trainer.logger.my_fantastic_logging
        torch.save(train_metrics, exp_path / "train_metrics.pt")

        # Save Checkpoints
        torch.save(server_state_dict, exp_path / f"ckpt_server_latest.pt")

        for t in client_trainers:
            torch.save(
                t.network.decoder.seg_layers.state_dict(),  # type: ignore
                exp_path / f"ckpt_{t.plans_manager.dataset_name}_latest.pt",
            )

        if not cfg.grad_mode == "fedavg_fallback":

            if not os.path.exists(exp_path / "variances"):
                os.makedirs(exp_path / "variances")

            if os.path.exists(exp_path / "variances" / f"ckpt_server_var_latest.pt"):
                os.rename(
                    exp_path / "variances" / f"ckpt_server_var_latest.pt",
                    exp_path / "variances" / f"ckpt_server_var_previous.pt",
                )
            torch.save(
                server_variance, exp_path / "variances" / f"ckpt_server_var_latest.pt"
            )

            # Save the variance statistics as a csv
            variance_stats = pd.DataFrame(variance_tape)
            unclipped_variance_stats = pd.DataFrame(unclipped_variance_tape)
            variance_stats.to_csv(
                exp_path / "variances" / "variance_stats.csv", index=False
            )
            unclipped_variance_stats.to_csv(
                exp_path / "variances" / "unclipped_variance_stats.csv", index=False
            )

            for t in client_trainers:
                if os.path.exists(
                    exp_path
                    / "variances"
                    / f"ckpt_{t.plans_manager.dataset_name}_var_latest.pt"
                ):
                    os.rename(
                        exp_path
                        / "variances"
                        / f"ckpt_{t.plans_manager.dataset_name}_var_latest.pt",
                        exp_path
                        / "variances"
                        / f"ckpt_{t.plans_manager.dataset_name}_var_previous.pt",
                    )
                torch.save(
                    client_variances[client_trainers.index(t)],
                    exp_path
                    / "variances"
                    / f"ckpt_{t.plans_manager.dataset_name}_var_latest.pt",
                )
            # Save state_dict

        if epoch % cfg.checkpoint_every == 0:
            torch.save(server_state_dict, exp_path / f"ckpt_server_{epoch}.pt")
            for t in client_trainers:
                t.save_checkpoint(join(t.output_folder, f"checkpoint_{epoch}.pth"))

                torch.save(
                    t.network.decoder.seg_layers.state_dict(),  # type: ignore
                    exp_path / f"ckpt_{t.plans_manager.dataset_name}_{epoch}.pt",
                )

            if not cfg.grad_mode == "fedavg_fallback":
                torch.save(
                    server_variance,
                    exp_path / "variances" / f"ckpt_server_var_{epoch}.pt",
                )

    for t in client_trainers:
        t.on_train_end()
