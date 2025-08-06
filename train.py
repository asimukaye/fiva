import json
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy

from functools import partial

import numpy as np

import torch

# import torch._dynamo.config
import torch.nn as nn

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

from funavg import LabelTransform

from config import (
    Config,
    LABEL_SUBSETS,
    # N_ITERATIONS_PER_FED_EPOCH_2D,
    SLICES_PER_DATASET,
    SAMPLES_PER_DATASET,
)

# torch._dynamo.config.force_parameter_static_shapes = False

class FedAVG:
    def __init__(self, client_weights):
        self.client_weights = client_weights

    def __call__(self, model_state_dicts):
        layer_names = list(model_state_dicts[0].keys())
        server_state_dict = OrderedDict([])
        for layer_name in layer_names:
            ws = [msd[layer_name] for msd in model_state_dicts]
            new_w = torch.zeros_like(ws[0])
            for w, cw in zip(ws, self.client_weights):
                new_w += (cw * w).type(w.dtype)
            server_state_dict.update({layer_name: new_w})
        return server_state_dict


def get_plans_and_dataset_json(dataset_name: str, plans_identifier: str):
    assert nnUNet_preprocessed is not None, "nnUNet_preprocessed is not set!"
    preprocessed_dataset_folder_base = join(
        nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name)
    )

    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")

    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))

    return plans, dataset_json


def get_trainer(
    configuration: str,
    fold: int,
    device: torch.device,
    plans: dict,
    dataset_json: dict,
    *args,
    **kwargs,
):

    nnunet_trainer = nnUNetTrainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
        *args,
        **kwargs,
    )
    return nnunet_trainer


def get_labels(dataset_names, label_subsets):
    assert nnUNet_preprocessed is not None, "nnUNet_preprocessed is not set!"

    labels = {
        d_name: {"all_labels_in_dataset": {}, "wanted_labels_in_dataset": {}}
        for d_name in dataset_names
    }

    for d_name in dataset_names:
        with open(Path(nnUNet_preprocessed) / d_name / "dataset.json", "r") as f:
            all_labels: dict = json.load(f)["labels"]

        labels[d_name]["all_labels_in_dataset"] = {
            int(v): k for k, v in all_labels.items()
        }

        if label_subsets is None:
            labels[d_name]["wanted_labels_in_dataset"] = all_labels.copy()
        else:
            labels[d_name]["wanted_labels_in_dataset"] = {
                l: i + 1 for i, l in enumerate(label_subsets[d_name])
            }

    return labels





def configure_dropout(trainer: nnUNetTrainer, dropout_p, nnunet_config):
    assert nnunet_config in ["2d", "3d"], "nnUNet config must be 2d or 3d"
    trainer.configuration_manager.network_arch_init_kwargs["dropout_op"] = (
        f"torch.nn.Dropout{nnunet_config}"
    )
    trainer.configuration_manager.network_arch_init_kwargs["dropout_op_kwargs"] = {
        "p": dropout_p
    }

def configure_client(t: nnUNetTrainer, labels: dict, conv_op: nn.Module, cfg: Config, exp_path: Path, from_ckpt: bool):
    
    t.on_train_start()

    dataset_name = t.plans_manager.dataset_name
    ds_labels = labels[dataset_name]

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

    n_seg_heads = len(ds_labels["wanted_labels_in_dataset"]) + 1

    t.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
        [
            conv_op(m.in_channels, n_seg_heads, m.kernel_size, stride=m.stride)
            for m in t.network.decoder.seg_layers  # type: ignore
        ]
    ).to(cfg.train.device)
    # FIXME: Change this to use 3D as well when needed
    t.num_iterations_per_epoch = cfg.num_iters[
        t.plans_manager.dataset_name
    ]
    # t.num_iterations_per_epoch = SLICE[t.plans_manager.dataset_name]
    t.num_val_iterations_per_epoch = t.num_iterations_per_epoch // 5 +1
    t.num_epochs = cfg.num_rounds
    t.save_every = 1
    t.optimizer, t.lr_scheduler = t.configure_optimizers()

    return t


def fix_labels_in_dataset_json(dataset_json, wanted_labels):
    new_labels = { "background": 0 }
    for k, v in dataset_json["labels"].items():
        if k in wanted_labels:
            new_labels[k] = v
    dataset_json["labels"] = new_labels
    return dataset_json



def train_fedavg(
    cfg: Config,
    exp_path: Path,
    from_ckpt: bool,
):

    if cfg.label_subset:
        labels = get_labels(cfg.dataset.ids, LABEL_SUBSETS)
    else:
        labels = get_labels(cfg.dataset.ids, None)

    # Using TotalSegmentator dataset for master trainer, common architecture and plan
    master_plan, master_dset_json = get_plans_and_dataset_json(
    cfg.master_dataset, "nnUNetPlans"
)
    master_trainer = get_trainer(
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
        plans=master_plan,
        dataset_json=master_dset_json,
    )

    get_trainer_partial = partial(
        get_trainer,
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
    )

    # Get client trainiers
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
        #@#@#@#@#@#@#@#@#@#@#@#@#@#
        # dataset_json = fix_labels_in_dataset_json(
        #     dataset_json, labels[d_set]["wanted_labels_in_dataset"]
        # )

        t = get_trainer_partial(plans=plans, dataset_json=dataset_json)
        client_trainers.append(t)

    if cfg.train.dropout_p is not None:
        configure_dropout(master_trainer, cfg.train.dropout_p, cfg.nnunet.config[:2])
        for t in client_trainers:
            configure_dropout(t, cfg.train.dropout_p, cfg.nnunet.config[:2])

    master_trainer.initialize()
    conv_op = getattr(nn, f"Conv{cfg.nnunet.config[:2]}")
    
    #@#@#@#@#@#@#@#@#@#@#@#@#@#
    # for t in client_trainers:
    #     t.initialize()
    # exit()
    # Checkpoint handling: Load master trainer checkpoint
    if from_ckpt:
        # n_epochs_trained = np.max(
        #     [
        #         int(x.name.replace(".pt", "").split("_")[-1])
        #         for x in exp_path.glob("ckpt_*.pt")
        #     ]
        # )
        master_ckpt = torch.load(
            exp_path / f"ckpt_server_{cfg.cknum}.pt", map_location="cpu"
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

    # Configure all the clients
    for t in client_trainers:

        t.on_train_start()
        # Replace with
        # t.network = deepcopy(master_trainer.network)

        dataset_name = t.plans_manager.dataset_name

        ds_labels = labels[dataset_name]

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

        n_seg_heads = len(ds_labels["wanted_labels_in_dataset"]) + 1

        t.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
            [
                conv_op(m.in_channels, n_seg_heads, m.kernel_size, stride=m.stride)
                for m in t.network.decoder.seg_layers  # type: ignore
            ]
        ).to(cfg.train.device)
        # FIXME: Change this to use 3D as well when needed
        t.num_iterations_per_epoch = cfg.num_iters[
            t.plans_manager.dataset_name
        ]
        # t.num_iterations_per_epoch = SLICE[t.plans_manager.dataset_name]
        t.num_val_iterations_per_epoch = t.num_iterations_per_epoch // 5 +1
        t.num_epochs = cfg.num_rounds
        # t.save_every = cfg.checkpoint_every

        t.optimizer, t.lr_scheduler = t.configure_optimizers()

        # Load checkpoint
        if from_ckpt:
            t.load_checkpoint(
                str(
                    exp_path
                    / t.plans_manager.dataset_name
                    / "nnUNetTrainer__nnUNetPlans__2d"
                    / f"fold_{cfg.fold}"
                    / f"checkpoint_{cfg.cknum}.pth"
                )
            )
            # Load segmentation head
            t_ckpt = torch.load(
                exp_path / f"ckpt_{t.plans_manager.dataset_name}_{cfg.cknum}.pt"
            )
            # t_sd = t.network.state_dict()
            # for k, v in t_sd.items():
            #     if k in master_ckpt and not k.startswith(
            #         "_orig_mod.decoder.seg_layers"
            #     ):
            #         t_sd[k] = master_ckpt[k]
            # t_ckpt = {f'decoder.seg_layers.{n}': w for n, w in t_ckpt.items()}
            t.network.decoder.seg_layers.load_state_dict(t_ckpt)  # type: ignore
            del t_ckpt

   
        # del master_ckpt
        
    # Set final layers to identity in master trainer
    # ic(master_trainer.network.decoder.seg_layers.state_dict().keys())
    master_trainer.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
        [nn.Identity() for _ in client_trainers[0].network.decoder.seg_layers]  # type: ignore
    )

    n_samples_per_client = np.array(
        [SAMPLES_PER_DATASET[t.plans_manager.dataset_name] for t in client_trainers]
    )

    # n_labels_per_client = np.array([len(t.dataset_json['labels']) for t in client_trainers])
    # fedavg weights per client
    client_weights_samples = n_samples_per_client / np.sum(n_samples_per_client)
    # client_weights_labels = n_labels_per_client / np.sum(n_labels_per_client)

    client_weights = client_weights_samples
    fed_avg = FedAVG(client_weights=client_weights.tolist())

    server_state_dict = master_trainer.network.state_dict()  # type: ignore

    # Checkpoint handling for epochs
    if from_ckpt:
        m = torch.load(exp_path / "train_metrics.pt")
        ds = list(labels.keys())[0]
        n_epochs_trained = len(m[ds]["train_losses"])
        master_trainer.current_epoch = n_epochs_trained
    else:
        n_epochs_trained = 0

    # if from_ckpt:
    #     epochs_trained = np.max(t_epochs)
        # for t in client_trainers:
            # t.current_epoch = n_epochs_trained


    ############## Training loop ###################
    for epoch in range(n_epochs_trained, cfg.num_rounds):


        # Update client weights with server weights
        for t in client_trainers:
            t_state_dict = t.network.state_dict()  # type: ignore
            for k, v in server_state_dict.items():
                t_state_dict[k] = v.clone()
            # t_state_dict = copy_param_dict(server_param_dict, t_state_dict)

            t.network.load_state_dict(t_state_dict)  # type: ignore

        # Update client weights with master old method
        # for t in client_trainers:
        #     cls_head_state_dict = {
        #         k: v
        #         for k, v in t.network.state_dict().items()  # type: ignore
        #         if k.startswith(
        #             "_orig_mod.decoder.seg_layers"
        #         )  # NOTE: if using torch compile, the module keys start with prefix '_orig_mod.'
        #     }
        #     for k, v in cls_head_state_dict.items():
        #         server_state_dict.update({k: v})
        #     t.network.load_state_dict(server_state_dict)  # type: ignore

        state_dicts = []

        # Each client is represented  by a trainer
        for t in client_trainers:
            print(
                f"#####################     {t.plans_manager.dataset_name}       ####################"
            )

            t.on_epoch_start()
            t.on_train_epoch_start()

            # TODO: Check what is deep supervision
            t.network.decoder.deep_supervision = True  # type: ignore
            train_outputs = []

            # ic(t.dataloader_train.data_loader.indices)

            for batch_id in trange(t.num_iterations_per_epoch):
                # TODO: Check why batches are handled this way
                batch = next(t.dataloader_train)  # type: ignore
                # ic(batch["data"].shape)
                # ic(batch["target"].shape)
                loss = t.train_step(batch)

                train_outputs.append(loss)
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
            state_dict = {
                k: v
                for k, v in t.network.state_dict().items()  # type: ignore
                if not k.startswith("_orig_mod.decoder.seg_layers")
            }
            state_dicts.append(state_dict)

        server_state_dict = fed_avg(state_dicts)

        # for t in client_trainers:
        #     ic(t.logger.my_fantastic_logging)

        # Save metrics
        train_metrics = {
            t.plans_manager.dataset_name: t.logger.my_fantastic_logging
            for t in client_trainers
        }
        torch.save(train_metrics, exp_path / "train_metrics.pt")

        # Save Checkpoints
        torch.save(server_state_dict, exp_path / f"ckpt_server_latest.pt")

        for t in client_trainers:
            torch.save(
                t.network.decoder.seg_layers.state_dict(),  # type: ignore
                exp_path / f"ckpt_{t.plans_manager.dataset_name}_latest.pt")


        # Save state_dict
        if epoch % cfg.checkpoint_every == 0:
            torch.save(server_state_dict, exp_path / f"ckpt_server_{epoch}.pt")
            for t in client_trainers:
                t.save_checkpoint(join(t.output_folder, f"checkpoint_{epoch}.pth"))
                torch.save(
                    t.network.decoder.seg_layers.state_dict(),  # type: ignore
                    exp_path / f"ckpt_{t.plans_manager.dataset_name}_{epoch}.pt",
                )

    for t in client_trainers:
        t.on_train_end()

    # master_trainer.on_train_end()