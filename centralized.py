import json

from pathlib import Path
from copy import deepcopy

from functools import partial

import numpy as np

import torch

# from torch.amp.autocast_mode import autocast
from torch.backends import cudnn
import torch.nn as nn

from tqdm import trange
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from funavg import LabelTransform

from train import get_labels, get_plans_and_dataset_json, get_trainer, configure_dropout
from config import (
    Config,
    LABEL_SUBSETS,
    # N_ITERATIONS_PER_FED_EPOCH_2D,
    SLICES_PER_DATASET,
    SAMPLES_PER_DATASET,
)


def train_centralized(
    cfg: Config,
    exp_path: Path,
    from_ckpt: bool,
) -> None:

    labels = get_labels(cfg.dataset.ids, LABEL_SUBSETS)

    # ckpt_base_path = exp_path if from_ckpt else None

    # Using TotalSegmentator dataset for master trainer, common architecture and plan

    master_plan, master_djson = get_plans_and_dataset_json(
        cfg.master_dataset, "nnUNetPlans"
    )

    # master_plan, master_djson = get_plans_and_dataset_json(
    #     "Dataset005_BTCV", "nnUNetPlans"
    # )
    master_trainer = get_trainer(
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
        plans=master_plan,
        dataset_json=master_djson,
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

    master_trainer.num_epochs = cfg.num_rounds
    avg_iters = np.mean([cfg.num_iters[d] for d in labels.keys()])
    print(f"Average iterations per epoch: {avg_iters}")
    # avg_iters = 250
    master_trainer.num_iterations_per_epoch = int(avg_iters)
    master_trainer.num_val_iterations_per_epoch = master_trainer.num_iterations_per_epoch // 5 +1

    master_trainer.initialize()
    
    conv_op = getattr(nn, f"Conv{cfg.nnunet.config[:2]}")


    # Configure all the clients
    for t in client_trainers:

        t.on_train_start()
        t.network = deepcopy(master_trainer.network)

        dataset_name = t.plans_manager.dataset_name

        ds_labels = labels[dataset_name]

        lt = LabelTransform(**ds_labels)

        if isinstance(t.dataloader_train, SingleThreadedAugmenter):
            t.dataloader_train.data_loader.transforms.transforms.append(lt)  # type: ignore
            t.dataloader_val.data_loader.transforms.transforms.append(lt)  # type: ignore
        elif isinstance(t.dataloader_train, NonDetMultiThreadedAugmenter):
            t.dataloader_train.generator.transforms.transforms.append(lt) # type: ignore
            t.dataloader_val.generator.transforms.transforms.append(lt)  # type: ignore

        # HACK: To get the dataloader to start after adding the transform
        _ = next(t.dataloader_train) # type: ignore
        _ = next(t.dataloader_val) # type: ignore

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


    # Set the decoder layers to identity
    master_trainer.network.decoder.seg_layers = nn.ModuleList( # type: ignore
        [nn.Identity() for _ in range(len(master_trainer.network.decoder.seg_layers))] # type: ignore
    ) # 
    master_trainer.network.decoder.deep_supervision = True # type: ignore
    # backbone = deepcopy(master_trainer.network)
    # cls_heads = []

    loss = master_trainer._build_loss()
    # master_optim, lr_scheduler = master_trainer.configure_optimizers()
    # device = master_trainer.device

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = deepcopy(master_trainer.network)
            self.cls_heads = nn.ModuleDict(
                {
                    t.plans_manager.dataset_name: deepcopy(t.network.decoder.seg_layers) # type: ignore
                    for t in client_trainers
                }
            )

        def forward(self, x):
            out = self.backbone(x)  # [B, C, H, W], # type: ignore
            outs = {n: [m(o) for m, o in zip(cls_head[::-1], out)] for i, (n, cls_head) in enumerate(self.cls_heads.items())}  # type: ignore
            return outs

    model = Model()

    optimizer = torch.optim.SGD(
        model.parameters(),
        master_trainer.initial_lr,
        weight_decay=master_trainer.weight_decay,
        momentum=0.99,
        nesterov=True,
    )
    lr_scheduler = PolyLRScheduler(
        optimizer, master_trainer.initial_lr, master_trainer.num_epochs
    )

    # grad_scaler = deepcopy(master_trainer.grad_scaler)
    if from_ckpt:
        # n_epochs_trained = np.max(
        #     [
        #         int(x.name.replace(".pt", "").split("_")[-1])
        #         for x in exp_path.glob("ckpt_*.pt")
        #     ]
        # )
        # master_ckpt = torch.load(
        #     exp_path / f"ckpt_{n_epochs_trained}.pt", map_location="cpu"
        # )
        # model.load_state_dict(master_ckpt["model"])
        # optimizer.load_state_dict(master_ckpt["optimizer"])
        # lr_scheduler.load_state_dict(master_ckpt["lr_scheduler"])
        # master_trainer.current_epoch = n_epochs_trained

        m = torch.load(exp_path / "train_metrics.pt")
        ds = list(labels.keys())[0]
        n_epochs_trained = len(m[ds]["train_losses"])
        print(f"Resuming Training from epoch {n_epochs_trained}")
        del m # free up memory

        master_ckpt = torch.load(
            exp_path / f"ckpt_latest.pt", map_location="cpu"
        )
        model.load_state_dict(master_ckpt["model"])
        optimizer.load_state_dict(master_ckpt["optimizer"])
        lr_scheduler.load_state_dict(master_ckpt["lr_scheduler"])
        master_trainer.current_epoch = n_epochs_trained

        for t in client_trainers:
            t.load_checkpoint(
                str(
                    exp_path
                    / t.plans_manager.dataset_name
                    / "nnUNetTrainer__nnUNetPlans__2d"
                    / "fold_0"
                    / "checkpoint_latest.pth"
                )
            )


    for epoch in range(master_trainer.current_epoch, master_trainer.num_epochs):
        model.train()
        for t in client_trainers:
            t.on_epoch_start()
            t.on_train_epoch_start()

        train_outputs = {t.plans_manager.dataset_name: [] for t in client_trainers}

        for batch_id in trange(
            master_trainer.num_iterations_per_epoch
        ):  # master_trainer.num_iterations_per_epoch):
            for t in client_trainers:
                print(
                f"#####################     {t.plans_manager.dataset_name}       ####################"
                )
                batch = next(t.dataloader_train)  # type: ignore
                t_data = batch["data"].to(cfg.train.device)
                t_target = batch["target"]
                t_target = [i.to(cfg.train.device, non_blocking=True) for i in t_target]
                optimizer.zero_grad()
                output = model(t_data)
                output = output[t.plans_manager.dataset_name]
                l = loss(output, t_target)
                l.backward()
                optimizer.step()
                train_outputs[t.plans_manager.dataset_name].append(
                    {"loss": l.detach().cpu().numpy()}
                )

        # lr_scheduler.step(epoch)
        for t in client_trainers:
            t.on_train_epoch_end(train_outputs[t.plans_manager.dataset_name])

        val_outputs = {t.plans_manager.dataset_name: [] for t in client_trainers}
        model.eval()
        for batch_id in trange(master_trainer.num_val_iterations_per_epoch):
            for t in client_trainers:
                # print(
                # f"#####################     {t.plans_manager.dataset_name}       ####################"
                # )
                batch = next(t.dataloader_val)  # type: ignore
                t_data = batch["data"].to(cfg.train.device)
                t_target = batch["target"]
                t_target = [i.to(cfg.train.device, non_blocking=True) for i in t_target]

                with torch.no_grad():
                    output = model(t_data)

                output = output[t.plans_manager.dataset_name]
                l = loss(output, t_target)

                output = output[0].detach()
                target = t_target[0].detach()

                # the following is needed for online evaluation. Fake dice (green line)
                axes = [0] + list(range(2, len(output.shape)))

                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(
                    output.shape, device=output.device, dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg
                # predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
                mask = None

                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    predicted_segmentation_onehot, target, axes=axes, mask=mask
                )

                tp_hard = tp.detach().cpu().numpy()
                fp_hard = fp.detach().cpu().numpy()
                fn_hard = fn.detach().cpu().numpy()

                # [1:] in order to remove background
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
                # print(l.item(), tp_hard, t.plans_manager.dataset_name, 'test')
                val_outputs[t.plans_manager.dataset_name].append(
                    {
                        "loss": l.detach().cpu().numpy(),
                        "tp_hard": tp_hard,
                        "fp_hard": fp_hard,
                        "fn_hard": fn_hard,
                    }
                )

        for t in client_trainers:
            print(t.plans_manager.dataset_name)
            t.on_validation_epoch_end(val_outputs[t.plans_manager.dataset_name])
            t.on_epoch_end()

        # _val_outputs = [xx for x in val_outputs.values() for xx in x]
        # master_trainer.on_validation_epoch_end([{'loss': np.array([0.]), 'tp_hard': np.array([0.]), 'fp_hard': np.array([1.]), 'fn_hard': np.array([0.])}])
        # master_trainer.on_epoch_end()
        train_metrics = {
            t.plans_manager.dataset_name: t.logger.my_fantastic_logging
            for t in client_trainers
        }
        torch.save(train_metrics, exp_path / "train_metrics.pt")

        ckpt = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict(),
            )
        torch.save(ckpt, exp_path / f"ckpt_latest.pt")
        
        if epoch % cfg.checkpoint_every == 0:
            ckpt = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict(),
            )
            torch.save(ckpt, exp_path / f"ckpt_{epoch}.pt")
            # torch.save(master_trainer.network.state_dict(), exp_path / f'ckpt_e{epoch}.pt')
            # for t in nnunet_trainers:
            #     torch.save(t.network.state_dict(), exp_path / f'ckpt_e{epoch}_{t.plans_manager.dataset_name}.pt')
    for t in client_trainers:
        t.on_train_end()
    # master_trainer.on_train_end()
