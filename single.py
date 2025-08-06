import json
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy

from functools import partial

import numpy as np

import torch

import torch._dynamo.config
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

from config import (
    Config,
    LABEL_SUBSETS,
    # N_ITERATIONS_PER_FED_EPOCH_2D,
    SLICES_PER_DATASET,
    SAMPLES_PER_DATASET,
)

from train import get_labels, get_plans_and_dataset_json, get_trainer, configure_dropout, fix_labels_in_dataset_json

from funavg import LabelTransform

def train_single(
    cfg: Config,
    exp_path: Path,
    from_ckpt: bool,
) -> None:
    labels = get_labels(cfg.dataset.ids, LABEL_SUBSETS)
    
    master_plan, master_json = get_plans_and_dataset_json(
    cfg.master_dataset, "nnUNetPlans"
)

    plan, djson = get_plans_and_dataset_json(cfg.dataset.ids[0], "nnUNetPlans")
    plan["configurations"][cfg.nnunet.config]["batch_size"] = cfg.train.batch_size
    plan["configurations"][cfg.nnunet.config]["patch_size"] = master_plan[
        "configurations"
    ][cfg.nnunet.config]["patch_size"]
    plan["configurations"][cfg.nnunet.config]["batch_dice"] = master_plan[
        "configurations"
    ][cfg.nnunet.config]["batch_dice"]

    plan["configurations"][cfg.nnunet.config]["architecture"] = deepcopy(
        master_plan["configurations"][cfg.nnunet.config]["architecture"]
    )
    djson = fix_labels_in_dataset_json(
            djson, labels[cfg.dataset.ids[0]]["wanted_labels_in_dataset"]
        )
    trainer = get_trainer(
        configuration=cfg.nnunet.config,
        fold=cfg.fold,
        device=torch.device(cfg.train.device),
        plans=plan,
        dataset_json=djson,
    )

    if cfg.train.dropout_p is not None:
        configure_dropout(trainer, cfg.train.dropout_p, cfg.nnunet.config[:2])

    # conv_op = getattr(nn, f"Conv{cfg.nnunet.config[:2]}")

    trainer.initialize()
    trainer.on_train_start()
    dataset_name = trainer.plans_manager.dataset_name

    # Reduce labels to subset
    ds_labels = labels[dataset_name]

    lt = LabelTransform(**ds_labels)

    if isinstance(trainer.dataloader_train, SingleThreadedAugmenter):
        trainer.dataloader_train.data_loader.transforms.transforms.append(lt)  # type: ignore
        trainer.dataloader_val.data_loader.transforms.transforms.append(lt)  # type: ignore
    elif isinstance(trainer.dataloader_train, NonDetMultiThreadedAugmenter):
        trainer.dataloader_train.generator.transforms.transforms.append(lt)  # type: ignore
        trainer.dataloader_val.generator.transforms.transforms.append(lt)  # type: ignore

    # HACK: To get the dataloader to start after adding the transform
    _ = next(trainer.dataloader_train)  # type: ignore
    _ = next(trainer.dataloader_val)  # type: ignore

    # n_seg_heads = len(ds_labels["wanted_labels_in_dataset"]) + 1
    # trainer.network.decoder.seg_layers = nn.ModuleList(  # type: ignore
    #     [
    #         conv_op(m.in_channels, n_seg_heads, m.kernel_size, stride=m.stride)
    #         for m in trainer.network.decoder.seg_layers  # type: ignore
    #     ]
    # ).to(cfg.train.device)
    # FIXME: Change this to use 3D as well when needed
    trainer.num_iterations_per_epoch = cfg.num_iters[dataset_name]

    # t.num_iterations_per_epoch = SLICE[t.plans_manager.dataset_name]
    trainer.num_val_iterations_per_epoch = trainer.num_iterations_per_epoch // 5 +1
    trainer.num_epochs = cfg.num_rounds
    trainer.save_every = 1

    trainer.optimizer, trainer.lr_scheduler = trainer.configure_optimizers()

    if from_ckpt:
        ckpt_base = (
            exp_path
            / dataset_name
            / f"{cfg.train.name}__nnUNetPlans__{cfg.nnunet.config}"
            / f"fold_{cfg.fold}"
        )
        if (ckpt_base / "checkpoint_latest.pth").exists():
            trainer.load_checkpoint(str(ckpt_base / "checkpoint_latest.pth"))
        elif (ckpt_base / "checkpoint_best.pth").exists():
            trainer.load_checkpoint(str(ckpt_base / "checkpoint_best.pth"))
        else:
            FileNotFoundError(f"Checkpoint not found for {dataset_name}")


    for epoch in range(trainer.current_epoch, trainer.num_epochs):
        trainer.on_epoch_start()

        trainer.on_train_epoch_start()
        trainer.network.decoder.deep_supervision = True  # type: ignore
        train_outputs = []

        for batch_id in trange(trainer.num_iterations_per_epoch):
            train_outputs.append(trainer.train_step(next(trainer.dataloader_train)))  # type: ignore
        trainer.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            trainer.on_validation_epoch_start()
            val_outputs = []
            for batch_id in trange(trainer.num_val_iterations_per_epoch):
                val_outputs.append(trainer.validation_step(next(trainer.dataloader_val)))  # type: ignore
            trainer.on_validation_epoch_end(val_outputs)

        trainer.on_epoch_end()

        # Save metrics
        torch.save(trainer.logger.my_fantastic_logging, exp_path / "train_metrics.pt")

    trainer.on_train_end()

