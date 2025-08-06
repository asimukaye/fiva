from pathlib import Path
from copy import deepcopy
import time
import json
import yaml
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import numpy as np


from testutils import prediction_with_patching
from argparse import ArgumentParser
from config import DATASET_SHORTHANDS
from logutils import get_free_gpu
import torch
import torch.nn as nn

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from train import get_trainer, get_plans_and_dataset_json, nnUNetTrainer

from config import (
    Config,
    LABEL_SUBSETS,
    get_default_config,
    get_single_config,
    ROOT_PATH,
)

from icecream import install, ic

import random
from torch.backends import cudnn
import os

install()
ic.configureOutput(includeContext=True)

DEVICE = f"cuda:{get_free_gpu()}"
# DEVICE = f"cuda:3"
SAVE_OUTPUTS = False
USE_SAMPLING = True
# OG_VERSION = True
OG_VERSION = False
BOOLEAN_FIX = True
COMPUTE_ECE = True

SEED = 0
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn


def eceloss(softmaxes, labels, n_bins=15):
    """
    Modified from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    ece = torch.zeros(1, device=d)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        # if prop_in_bin.item() > 0.0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        if prop_in_bin.item() > 0.0:
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        accuracy_in_bin_list.append(accuracy_in_bin)
        avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)
    # print(acc_in_bin)
    # print(avg_conf_in_bin)
    return ece, acc_in_bin, avg_conf_in_bin


def normalize(x):
    # p05 = np.percentile(x, 0.05)
    p95 = np.percentile(x, 99)
    x = np.clip(x, 0, p95)
    x = (x - x.min()) / (x.max() - x.min())
    return x


def uncertainty(p_hat, var="sum"):
    # ic(p_hat.shape)
    p_mean = torch.mean(p_hat, dim=0)
    # ic(p_mean.shape)
    ale = torch.mean(p_hat * (1 - p_hat), dim=0)
    # ic(ale.shape)

    epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    # ic(epi.shape)
    if var == "sum":
        ale = torch.sum(ale, dim=0)
        epi = torch.sum(epi, dim=0)
    elif var == "top":
        ale = ale[torch.argmax(p_mean)]
        epi = epi[torch.argmax(p_mean)]
    uncert = ale + epi
    # ic(uncert.shape)
    return p_mean, uncert, ale, epi


def hard_dice_score(pred, target):
    # ic(pred.shape, target.shape)
    if not BOOLEAN_FIX:
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, target, axes=(0, 2, 3))
    else:
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred, target.to(torch.bool), axes=(0, 2, 3))
    # ic(tp, fp, fn, _)
    tp = tp.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    fn = fn.detach().cpu().numpy()
    dice = 2 * tp / (2 * tp + fp + fn + 1)
    # ic(dice)
    return dice


def onehot(t, n_cls):
    b, _, h, w = t.shape
    # ic(t.shape)
    t_onehot = torch.zeros((b, n_cls, h, w)).to(t.device)
    t_onehot.scatter_(1, t.long(), 1)
    if not BOOLEAN_FIX:
        return t_onehot
    else:
        return t_onehot.to(torch.int)


def onehot3d(t, n_cls):
    b, _, d, h, w = t.shape
    t_onehot = torch.zeros((b, n_cls, d, h, w)).to(t.device)
    t_onehot.scatter_(1, t.long(), 1)
    return t_onehot


def hard_dice_score3d(pred, target):
    tp, fp, fn, _ = get_tp_fp_fn_tn(pred, target, axes=(0, 2, 3, 4))
    tp = tp.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    fn = fn.detach().cpu().numpy()
    dice = 2 * tp / (2 * tp + fp + fn + 1)
    return dice


def compute_dice_score(pred, target, n_cls):
    pred_onehot = onehot(pred, n_cls=n_cls)
    target_onehot = onehot(target, n_cls=n_cls)
    # We might miss the fact that the classifier does not predict a label though it was trained on it
    # present_labels = present_target_labels * present_pred_labels
    dice = hard_dice_score(pred_onehot, target_onehot)
    return dice


def _sum(tensor_list):
    tensor_sum = torch.zeros_like(tensor_list[0])
    for t in tensor_list:
        tensor_sum += t
    return tensor_sum


def merged_logits_from_sep_cls(logits):
    n_preds_per_label = torch.stack([l.sum((0, 2, 3)) > 0 for l in logits]).sum(0)
    merged_logits = (1 / n_preds_per_label).view(-1, 1, 1) * _sum(logits)
    return merged_logits


def merged_logits_from_sep_cls_with_uncert(logits, uncert, alpha=1.0):
    merged_logits = merged_logits_from_sep_cls(logits)
    merged_logits[:, 0] *= 1 - _sum(uncert)
    return merged_logits


# https://arxiv.org/pdf/2206.10897.pdf
def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def compute_dice_score_2(tps, vols, n_cls):
    tps = tps.detach().cpu().numpy()
    vols = vols.detach().cpu().numpy()
    dice = 2 * tps / (vols + 1)
    dice = {l: d for l, d in zip(range(1, n_cls), dice[1:])}
    return dice


class Model(nn.Module):
    def __init__(
        self,
        # batch_size: int,
        master_trainer: nnUNetTrainer,
        dataset_names: list[str],
        label_subsets,
    ):
        super().__init__()
        # self.batch_size = batch_size

        self.backbone = master_trainer.network.cpu()

        self.cls_heads = nn.ModuleDict({})
        for dataset_name in dataset_names:
            n_labels = len(label_subsets[dataset_name])
            seg_head = nn.ModuleList(
                [
                    nn.Conv2d(m.in_channels, n_labels + 1, m.kernel_size, m.stride)
                    for m in self.backbone.decoder.seg_layers
                ]
            )
            self.cls_heads[dataset_name] = seg_head

        self.backbone.decoder.seg_layers = nn.ModuleList(
            [nn.Identity() for _ in self.backbone.decoder.seg_layers]
        )

    def forward(self, x):
        out = self.backbone(x)
        # ic(out.shape)
        outs = {}
        for n, cls_head in self.cls_heads.items():

            outs[n] = cls_head[-1](out)  # type: ignore
        # outs = {n: cls_head[:-1](out) for i, (n, cls_head) in enumerate(self.cls_heads.items())}
        return outs


def summarize(df: pd.DataFrame, exp_path: Path, exp_name: str):
    structures = [s for s in df.columns[2:] if ("_tp" not in s) and ("_vol" not in s)]
    tp_cols = [s for s in df.columns if "tp" in s]
    vol_cols = [s for s in df.columns if "vol" in s]
    mean_per_dataset = (
        df[["test_dataset", "case"] + structures]
        .groupby(["test_dataset"])
        .apply(lambda x: x[structures].apply(lambda y: round(y.mean(), 3)))
        .reset_index()
    )
    # Sum the values per dataset
    sums_per_dataset = (
        df[["test_dataset", "case"] + tp_cols + vol_cols]
        .groupby(["test_dataset"])
        .apply(lambda x: x[tp_cols + vol_cols].apply(lambda y: y.sum()))
        .reset_index()
    )
    for s in structures:
        sums_per_dataset[s] = (
            2 * sums_per_dataset[f"{s}_tp"] / sums_per_dataset[f"{s}_vol"]
        )
        sums_per_dataset[s + "_avg"] = mean_per_dataset[s]

    sums_per_dataset["mean_dice"] = sums_per_dataset[structures].mean(axis=1)
    # Remove volume and tp columns and round to 4 decimals
    sums_per_dataset[
        ["test_dataset", "mean_dice"] + structures + [s + "_avg" for s in structures]
    ].round(4).to_csv(exp_path / f"{exp_name}_summarized.csv", index=False)


def summarize_with_fun(df: pd.DataFrame, exp_path: Path, exp_name: str):
    structures = [
        s
        for s in df.columns[2:]
        if ("_tp" not in s) and ("_vol" not in s) and ("_fun" not in s)
    ]
    structures_fun = [f"{s}_fun" for s in structures]

    tp_cols = [s for s in df.columns if "tp" in s]
    vol_cols = [s for s in df.columns if "vol" in s]
    mean_per_dataset = (
        df[["test_dataset", "case"] + structures + structures_fun]
        .groupby(["test_dataset"])
        .apply(
            lambda x: x[structures + structures_fun].apply(lambda y: round(y.mean(), 4))
        )
        .reset_index()
    )
    # Sum the values per dataset
    sums_per_dataset = (
        df[["test_dataset", "case"] + tp_cols + vol_cols]
        .groupby(["test_dataset"])
        .apply(lambda x: x[tp_cols + vol_cols].apply(lambda y: y.sum()))
        .reset_index()
    )
    for s in structures:
        sums_per_dataset[s] = (
            2 * sums_per_dataset[f"{s}_tp"] / sums_per_dataset[f"{s}_vol"]
        )
        sums_per_dataset[s + "_avg"] = mean_per_dataset[s]
        sums_per_dataset[s + "_fun"] = (
            2 * sums_per_dataset[f"{s}_tp_fun"] / sums_per_dataset[f"{s}_vol_fun"]
        )
        sums_per_dataset[s + "_avg_fun"] = mean_per_dataset[s + "_fun"]

    sums_per_dataset["mean_dice"] = sums_per_dataset[structures].mean(axis=1)
    sums_per_dataset["mean_dice_avg"] = sums_per_dataset[
        [s + "_avg" for s in structures]
    ].mean(axis=1)
    sums_per_dataset["mean_dice_fun"] = sums_per_dataset[structures_fun].mean(axis=1)
    sums_per_dataset["mean_dice_fun_avg"] = sums_per_dataset[
        [s + "_avg" for s in structures]
    ].mean(axis=1)
    # Remove volume and tp columns and round to 4 decimals
    sums_per_dataset.round(4).to_csv(
        exp_path / f"{exp_name}_summarized.csv", index=False
    )


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"[SEED] Simulator global seed is set to: {seed}!")



def simple_inference(
    exp_name: str,
    label_subsets,
    model: Model,
    train_dataset_names: list[str],
    test_dataset_names: list[str],
    data_src: Path,
    train_ckpt_path: Path,
):

    train_labels = [label_subsets[ds_name] for ds_name in train_dataset_names]
    train_labels = np.unique([xx for x in train_labels for xx in x])
    train_labels = {l: i + 1 for i, l in enumerate(train_labels)}
    n_cls = len(train_labels) + 1
    print(n_cls)

    print(train_labels)
    test_dataset_labels = {}

    test_dataset_cases: dict[str, list] = {}

    for ds_name in test_dataset_names:
        ds_dir = ds_name
        with open(data_src / "nnUNet_preprocessed" / ds_dir / "dataset.json", "r") as f:
            test_dataset_labels[ds_name] = json.load(f)["labels"]

        with open(
            data_src / "nnUNet_preprocessed" / ds_dir / "splits_final.json", "r"
        ) as f:
            fold_splits = json.load(f)
            test_dataset_cases[ds_name] = fold_splits[0]["val"]
            if ds_name == "Dataset001_amos_full":
                test_dataset_cases[ds_name].extend(fold_splits[0]["train"])

    exp_date = time.strftime("%y-%m-%d")
    exp_time = time.strftime("%H-%M-%S")
    exp_path = data_src / "test_output" / exp_date / exp_name / exp_time
    exp_path.mkdir(exist_ok=True, parents=True)

    exp_config = {
        "exp_name": exp_name,
        "exp_date": exp_date,
        "exp_time": exp_time,
        "train_expt": train_ckpt_path,
        "use_sampling": USE_SAMPLING,
        "train_dataset_names": train_dataset_names,
        "test_dataset_names": test_dataset_names,
    }
    with open(exp_path / "test_config.yaml", "w") as f:
        yaml.dump(exp_config, f)

    dices = []
    net_dices = []

    for test_ds_name in test_dataset_names:
        all_test_ds_labels = test_dataset_labels[test_ds_name]
        test_ds_labels = label_subsets[test_ds_name]
        # Mask the labels that are not present in the dataset
        label_mask = np.array(
            [True if l in test_ds_labels else False for l in train_labels]
        )

        intersection_labels = [t for t in train_labels if t in test_ds_labels]
        if not len(intersection_labels):
            print(f"No intersection between train and test labels for {test_ds_name}")
            continue

        print(test_ds_name)
        ds_dir = test_ds_name

        cases = test_dataset_cases[test_ds_name]
        img_paths = [
            data_src / "nnUNet_preprocessed" / ds_dir / "nnUNetPlans_2d" / f"{case}.npy"
            for case in cases
        ]

        for i, (case, img_path) in enumerate(zip(cases, img_paths)):
            # Use only 20 images for now
            if i == 20:
                break

            img_np = np.load(
                data_src
                / "nnUNet_preprocessed"
                / ds_dir
                / "nnUNetPlans_2d"
                / f"{case}.npy"
            )
            img = torch.tensor(img_np).float()

            seg_np = np.load(
                data_src
                / "nnUNet_preprocessed"
                / ds_dir
                / "nnUNetPlans_2d"
                / f"{case}_seg.npy"
            )
            seg = torch.tensor(seg_np).float()

            # Map the test labels to the training labels
            seg_new = torch.zeros_like(seg)
            for l, i in train_labels.items():
                if l in all_test_ds_labels:
                    j = int(all_test_ds_labels[l])
                    seg_new[seg == j] = i

            tps = []
            fps = []
            fns = []
            avg_dices = []

            # Find foreground slices
            idxs = seg_new.sum((0, 2, 3)) > 0

            img_new_slices, seg_new_slices = (
                img.permute(1, 0, 2, 3)[idxs],
                seg_new.permute(1, 0, 2, 3)[idxs],
            )

            for img_slice, seg_slice in zip(
                # tqdm(img_new_slices[::5], leave=False), seg_new_slices[::5]
                tqdm(img_new_slices, leave=False),
                seg_new_slices,
            ):
                with torch.no_grad():

                    out = prediction_with_patching(
                        model, img_slice.to(DEVICE)[None], patch_size, DEVICE
                    )
                    out_softmax = {
                        k: torch.softmax(v.permute(1, 0, 2, 3).cpu(), dim=1)
                        for k, v in out.items()
                    }
                    # ic(out_softmax)

                mean_logits = []
                for train_ds_name, outs in out_softmax.items():
                    mean, uncert, ale, epi = uncertainty(outs, var="sum")
                    # ic(mean)
                    # ic(mean.shape)
                    mean_new = torch.zeros(
                        len(train_labels) + 1, mean.size(1), mean.size(2)
                    )
                    # ic(mean_new.shape)
                    mean_new[0] = mean[0]
                    ds_label_subset = np.array(label_subsets[train_ds_name])
                    # ic(ds_label_subset)

                    for k, i in train_labels.items():
                        if k in ds_label_subset:
                            # ic(np.argwhere(ds_label_subset == k))
                            # ic(np.argwhere(ds_label_subset == k)[0, 0] + 1)
                            mean_new[i] = mean[
                                np.argwhere(ds_label_subset == k)[0, 0] + 1
                            ]
                    mean_logits.append(mean_new[None])

                merged_logits = merged_logits_from_sep_cls(mean_logits)
                merged_pred = merged_logits.argmax(dim=1, keepdim=True)

                target_onehot = onehot(seg_slice[None], n_cls=n_cls)
                merged_pred_onehot = onehot(merged_pred, n_cls=n_cls)

                dice = hard_dice_score(merged_pred_onehot, target_onehot)[1:]

                tp, fp, fn, _ = get_tp_fp_fn_tn(
                    merged_pred_onehot, target_onehot.to(torch.bool), axes=(0, 2, 3)
                )

                mask = np.zeros((n_cls - 1,))
                mask[(torch.unique(seg_slice)[1:] - 1).long()] = 1
                mask = mask.astype(bool)

                tp[1:][~mask] = 0
                fp[1:][~mask] = 0
                fn[1:][~mask] = 0

                tps.append(tp)
                fps.append(fp)
                fns.append(fn)

                dice[~mask] = np.nan

                avg_dices.append(np.nanmean(dice[mask]))
                dice = {l: d for l, d in zip(train_labels, dice)}

                dices.append(
                    {
                        "case": f"{case}_{i}",
                        "test_dataset": test_ds_name,
                        **dice,
                    }
                )
            if len(merged_logits) == 0:
                continue
            tps = torch.stack(tps).sum(0)
            fps = torch.stack(fps).sum(0)
            fns = torch.stack(fns).sum(0)

            vol = 2 * tps + fps + fns
            net_dice = 2 * tps / (vol + 1)
            net_dice = net_dice.detach().cpu().numpy()[1:]
            tps = tps.detach().cpu().float().numpy()[1:]
            vol = vol.detach().cpu().float().numpy()[1:]

            net_dice[~label_mask] = np.nan
            tps[~label_mask] = np.nan
            vol[~label_mask] = np.nan

            net_dice = {l: d for l, d in zip(train_labels, net_dice)}
            tp_dict = {f"{l}_tp": d for l, d in zip(train_labels, tps)}
            vol_dict = {f"{l}_vol": d for l, d in zip(train_labels, vol)}
            print("Case:", case)
            print("Net dice:", np.nanmean(list(net_dice.values())))

            net_dices.append(
                {
                    "case": f"{case}",
                    "test_dataset": test_ds_name,
                    **net_dice,
                    **tp_dict,
                    **vol_dict,
                }
            )

            print("Case Slice avg Dice:", np.nanmean(avg_dices))

        df_dice = pd.DataFrame(dices)
        df_dice.to_csv(exp_path / f"{exp_name}_dice.csv")

        df_net_dice = pd.DataFrame(net_dices)
        summarize(df_net_dice, exp_path, exp_name)
        df_net_dice.to_csv(exp_path / f"{exp_name}_net_dice.csv")


def test(
    exp_name: str,
    label_subsets,
    model: Model,
    train_dataset_names: list[str],
    test_dataset_names: list[str],
    data_src: Path,
    train_ckpt_path: Path,
    with_FUN: bool = True,
    model_variance=None,
):

    train_labels = [label_subsets[ds_name] for ds_name in train_dataset_names]
    train_labels = np.unique([xx for x in train_labels for xx in x])
    train_labels = {l: i + 1 for i, l in enumerate(train_labels)}
    n_cls = len(train_labels) + 1
    print(n_cls)

    print(train_labels)
    test_dataset_labels = {}
    if not OG_VERSION:
        test_dataset_cases: dict[str, list] = {}
    # test_dataset_lbl_files = {}
    for ds_name in test_dataset_names:
        if ds_name == "Dataset001_amos_full":
            ds_dir = "Dataset001_amos"
        else:
            ds_dir = ds_name

        if not OG_VERSION:
            with open(
                data_src / "nnUNet_preprocessed" / ds_dir / "dataset.json", "r"
            ) as f:
                test_dataset_labels[ds_name] = json.load(f)["labels"]

            with open(
                data_src / "nnUNet_preprocessed" / ds_dir / "splits_final.json", "r"
            ) as f:
                fold_splits = json.load(f)
                test_dataset_cases[ds_name] = fold_splits[0]["val"]
                if ds_name == "Dataset001_amos_full":
                    test_dataset_cases[ds_name].extend(fold_splits[0]["train"])
        else:
            with open(
                data_src / "nnUNet_preprocessed_batches2" / ds_name / "dataset.json",
                "r",
            ) as f:
                test_dataset_labels[ds_name] = json.load(f)["labels"]

    exp_date = time.strftime("%y-%m-%d")
    exp_time = time.strftime("%H-%M-%S")
    exp_path = data_src / "test_output" / exp_date / exp_name / exp_time
    exp_path.mkdir(exist_ok=True, parents=True)

    mc_iter = 10
    # mc_iter = 3
    start_slice = 1

    exp_config = {
        "exp_name": exp_name,
        "exp_date": exp_date,
        "exp_time": exp_time,
        "train_expt": train_ckpt_path,
        "mc_iters": mc_iter,
        "use_sampling": USE_SAMPLING,
        "train_dataset_names": train_dataset_names,
        "test_dataset_names": test_dataset_names,
        "seed": SEED
        # "test_dataset_cases": test_dataset_cases,
    }
    with open(exp_path / "test_config.yaml", "w") as f:
        yaml.dump(exp_config, f)

    if USE_SAMPLING:
        assert model_variance is not None

        model_weights = model.backbone.state_dict()

    dices = []
    net_dices = []

    eces_log = []
    eces_fun_all = []
    eces_van_all = []
    accs_fun_all = []
    accs_van_all = []
    conf_fun_all = []
    conf_van_all = []
    eces_van, eces_fun = {i: [] for i in range(1, n_cls + 1)}, {
        i: [] for i in range(1, n_cls + 1)
    }
    for test_ds_name in test_dataset_names:
        all_test_ds_labels = test_dataset_labels[test_ds_name]
        test_ds_labels = label_subsets[test_ds_name]
        # Mask the labels that are not present in the dataset
        label_mask = np.array(
            [True if l in test_ds_labels else False for l in train_labels]
        )

        intersection_labels = [t for t in train_labels if t in test_ds_labels]
        if not len(intersection_labels):
            print(f"No intersection between train and test labels for {test_ds_name}")
            continue

        print(test_ds_name)
        if test_ds_name == "Dataset001_amos_full":
            ds_dir = "Dataset001_amos"
        else:
            ds_dir = test_ds_name

        if OG_VERSION:
            _fold = "validation_fold0"
            img_paths = list(
                (
                    data_src
                    / "nnUNet_preprocessed_batches2"
                    / test_ds_name
                    / _fold
                    / "images"
                ).glob("*.pt")
            )
            img_paths = sorted(img_paths)
            cases = [p.stem for p in img_paths]
        else:
            cases = test_dataset_cases[test_ds_name]
            img_paths = [
                data_src
                / "nnUNet_preprocessed"
                / ds_dir
                / "nnUNetPlans_2d"
                / f"{case}.npy"
                for case in cases
            ]

        for i, (case, img_path) in enumerate(zip(cases, img_paths)):

            # Use only 20 images for now
            if i == 20:
                break

            if OG_VERSION:
                img = torch.load(img_path)
                seg_path = (
                    data_src
                    / "nnUNet_preprocessed_batches2"
                    / test_ds_name
                    / _fold
                    / "labels"
                    / img_path.name
                )
                seg = torch.load(seg_path)
                # case = case.stem
            else:
                img_np = np.load(
                    data_src
                    / "nnUNet_preprocessed"
                    / ds_dir
                    / "nnUNetPlans_2d"
                    / f"{case}.npy"
                )
                img = torch.tensor(img_np).float()

                seg_np = np.load(
                    data_src
                    / "nnUNet_preprocessed"
                    / ds_dir
                    / "nnUNetPlans_2d"
                    / f"{case}_seg.npy"
                )
                seg = torch.tensor(seg_np).float()

            # Map the test labels to the training labels
            seg_new = torch.zeros_like(seg)
            for l, i in train_labels.items():
                if l in all_test_ds_labels:
                    j = int(all_test_ds_labels[l])
                    seg_new[seg == j] = i

            merged_logits_all = []
            merged_logits_fun_all = []
            uncerts_all = []
            inputs_all = []
            targets_all = []
            ds_logits = {}
            preds = {}

            tps = []
            fps = []
            fns = []
            tps_fun = []
            fps_fun = []
            fns_fun = []
            avg_dices = []

            # Find foreground slices
            idxs = seg_new.sum((0, 2, 3)) > 0

            img_new_slices, seg_new_slices = (
                img.permute(1, 0, 2, 3)[idxs],
                seg_new.permute(1, 0, 2, 3)[idxs],
            )

            for img_slice, seg_slice in zip(
                tqdm(img_new_slices[start_slice::5], leave=False), seg_new_slices[start_slice::5]
            ):

                inputs_all.append(img_slice)
                targets_all.append(seg_slice)

                with torch.no_grad():
                    mc_outputs = []
                    # MC DROPOUT

                    # mc_outputs = [{k: torch.softmax(v.cpu(), dim=1) for k, v in model(img_slice.to(DEVICE)[None]).items()} for _ in range(mc_iter)]
                    # Sampling
                    for _ in range(mc_iter):
                        if USE_SAMPLING:
                            sample_weights = {}
                            for k, v in model_variance.items():
                                sample_weights[k] = torch.normal(model_weights[k], v)
                            model.backbone.load_state_dict(sample_weights)

                        if OG_VERSION:
                            out = model(img_slice.to(DEVICE)[None])
                            out_softmax = {
                                k: torch.softmax(v.cpu(), dim=1) for k, v in out.items()
                            }
                        else:
                            out = prediction_with_patching(
                                model, img_slice.to(DEVICE)[None], patch_size, DEVICE
                            )
                            out_softmax = {
                                k: torch.softmax(v.permute(1, 0, 2, 3).cpu(), dim=1)
                                for k, v in out.items()
                            }

                        # print([osm.shape for osm in out_softmax.values()])
                        mc_outputs.append(out_softmax)

                # collate
                mc_outputs_collated = {}
                for train_ds_name in train_dataset_names:
                    mc_outputs_collated[train_ds_name] = torch.cat(
                        [o[train_ds_name] for o in mc_outputs], dim=0
                    )

                # compute uncertainty
                mean_logits, uncerts = [], []
                for train_ds_name, mc_outputs in mc_outputs_collated.items():
                    mean, uncert, ale, epi = uncertainty(mc_outputs, var="sum")

                    mean_new = torch.zeros(
                        len(train_labels) + 1, mean.size(1), mean.size(2)
                    )
                    # ic(mean_new.shape)
                    mean_new[0] = mean[0]
                    # ic(mean.shape)
                    ds_label_subset = np.array(label_subsets[train_ds_name])

                    for k, i in train_labels.items():
                        if k in ds_label_subset:
                            mean_new[i] = mean[
                                np.argwhere(ds_label_subset == k)[0, 0] + 1
                            ]

                    mean_logits.append(mean_new[None])
                    uncerts.append(uncert)

                    if train_ds_name not in preds:
                        preds[train_ds_name] = []
                    preds[train_ds_name].append(mean_new.argmax(0, keepdim=True))

                    if train_ds_name not in ds_logits:
                        ds_logits[train_ds_name] = []
                    ds_logits[train_ds_name].append(mean_new[None])

                if len(train_dataset_names) > 1:
                    merged_logits = merged_logits_from_sep_cls(mean_logits)
                    merged_logits_fun = merged_logits_from_sep_cls_with_uncert(
                        mean_logits, uncerts
                    )

                    merged_logits_all.append(merged_logits)
                    merged_logits_fun_all.append(merged_logits_fun)
                    uncerts_all.append(_sum(uncerts))

                    merged_pred = merged_logits.argmax(dim=1, keepdim=True)
                    merged_pred_fun = merged_logits_fun.argmax(dim=1, keepdim=True)

                    target_onehot = onehot(seg_slice[None], n_cls=n_cls)
                    merged_pred_onehot = onehot(merged_pred, n_cls=n_cls)
                    merged_pred_fun_onehot = onehot(merged_pred_fun, n_cls=n_cls)

                    dice = hard_dice_score(merged_pred_onehot, target_onehot)[1:]
                    dice_fun = hard_dice_score(merged_pred_fun_onehot, target_onehot)[
                        1:
                    ]

                    mask = np.zeros((n_cls - 1,))
                    mask[(torch.unique(seg_slice)[1:] - 1).long()] = 1
                    mask = mask.astype(bool)

                    tp, fp, fn, _ = get_tp_fp_fn_tn(
                        merged_pred_onehot, target_onehot.to(torch.bool), axes=(0, 2, 3)
                    )
                    tp[1:][~mask] = 0
                    fp[1:][~mask] = 0
                    fn[1:][~mask] = 0
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)

                    tp_fun, fp_fun, fn_fun, _ = get_tp_fp_fn_tn(
                        merged_pred_fun_onehot,
                        target_onehot.to(torch.bool),
                        axes=(0, 2, 3),
                    )
                    tp_fun[1:][~mask] = 0
                    fp_fun[1:][~mask] = 0
                    fn_fun[1:][~mask] = 0
                    tps_fun.append(tp_fun)
                    fps_fun.append(fp_fun)
                    fns_fun.append(fn_fun)

                    # then ignored py pandas
                    dice[~mask] = np.nan
                    dice_fun[~mask] = np.nan

                    # print("Slice:", case)
                    # print("OG dice:", np.nanmean(dice[mask]))
                    # print("FUN dice:", np.nanmean(dice_fun[mask]))
                    avg_dices.append(np.nanmean(dice[mask]))

                    # print("FUN dice mask:", (dice_fun[mask]))
                    dice = {l: d for l, d in zip(train_labels, dice)}
                    dice_fun = {f"{l}_fun": d for l, d in zip(train_labels, dice_fun)}

                    dices.append(
                        {
                            "case": f"{case}_{i}",
                            "test_dataset": test_ds_name,
                            **dice,
                            **dice_fun,
                        }
                    )

                else:
                    merged_logits_all.append(mean_logits[0])
                    uncerts_all.append(uncerts[0])
                    pred = mean_logits[0].argmax(dim=1, keepdim=True)
                    uncert = uncerts[0]

                    target_onehot = onehot(seg_slice[None], n_cls=n_cls)
                    pred_onehot = onehot(pred, n_cls=n_cls)

                    tp, fp, fn, _ = get_tp_fp_fn_tn(
                        pred_onehot, target_onehot.to(torch.bool), axes=(0, 2, 3)
                    )

                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)

                    dice = hard_dice_score(pred_onehot, target_onehot)[1:]
                    avg_dices.append(np.nanmean(dice))

                    dice = {l: d for l, d in zip(train_labels, dice)}

                    dices.append(
                        {"case": f"{case}_{i}", "test_dataset": test_ds_name, **dice}
                    )

            if len(merged_logits_all) == 0:
                continue

            if len(train_dataset_names) > 1:

                tps = torch.stack(tps).sum(0)
                fps = torch.stack(fps).sum(0)
                fns = torch.stack(fns).sum(0)

                vol = 2 * tps + fps + fns
                net_dice = 2 * tps / (vol + 1)
                net_dice = net_dice.detach().cpu().numpy()[1:]
                tps = tps.detach().cpu().float().numpy()[1:]
                vol = vol.detach().cpu().float().numpy()[1:]

                net_dice[~label_mask] = np.nan
                tps[~label_mask] = np.nan
                vol[~label_mask] = np.nan

                net_dice = {l: d for l, d in zip(train_labels, net_dice)}
                tp_dict = {f"{l}_tp": d for l, d in zip(train_labels, tps)}
                vol_dict = {f"{l}_vol": d for l, d in zip(train_labels, vol)}
                print("Case:", case)
                print("Net dice:", np.nanmean(list(net_dice.values())))

                tps_fun = torch.stack(tps_fun).sum(0)
                fps_fun = torch.stack(fps_fun).sum(0)
                fns_fun = torch.stack(fns_fun).sum(0)

                vol_fun = 2 * tps_fun + fps_fun + fns_fun
                net_dice_fun = 2 * tps_fun / (vol_fun + 1)
                net_dice_fun = net_dice_fun.detach().cpu().numpy()[1:]
                tps_fun = tps_fun.detach().cpu().float().numpy()[1:]
                vol_fun = vol_fun.detach().cpu().float().numpy()[1:]

                net_dice_fun[~label_mask] = np.nan
                tps_fun[~label_mask] = np.nan
                vol_fun[~label_mask] = np.nan

                net_dice_fun = {
                    f"{l}_fun": d for l, d in zip(train_labels, net_dice_fun)
                }
                tp_dict_fun = {f"{l}_tp_fun": d for l, d in zip(train_labels, tps_fun)}
                vol_dict_fun = {
                    f"{l}_vol_fun": d for l, d in zip(train_labels, vol_fun)
                }

                # print("Case:", case)
                print("Net dice FUN:", np.nanmean(list(net_dice_fun.values())))

                net_dices.append(
                    {
                        "case": f"{case}",
                        "test_dataset": test_ds_name,
                        **net_dice,
                        **tp_dict,
                        **vol_dict,
                        **net_dice_fun,
                        **tp_dict_fun,
                        **vol_dict_fun,
                    }
                )

                print("Slice avg Dice:", np.nanmean(avg_dices))

                logits = torch.cat(merged_logits_all, dim=0).permute(1, 0, 2, 3)[None]
                logits_fun = torch.cat(merged_logits_fun_all, dim=0).permute(
                    1, 0, 2, 3
                )[None]
                targets = torch.cat(targets_all, dim=0)

                merged_logits_all = logits.numpy()

                merged_logits_fun_all = logits_fun.numpy()
                targets_all = targets.numpy()

                uncerts_all = torch.stack(uncerts_all).numpy()
                inputs_all = torch.cat(inputs_all, dim=0).numpy()

                if COMPUTE_ECE:
                    van_ece_all, acc_van, conf_van = eceloss(normalize(logits), targets)
                    fun_ece_all, acc_fun, conf_fun = eceloss(
                        normalize(logits_fun), targets
                    )
                    eces_van_all.append(van_ece_all)
                    eces_fun_all.append(fun_ece_all)
                    accs_van_all.append(acc_van)
                    accs_fun_all.append(acc_fun)
                    conf_van_all.append(conf_van)
                    conf_fun_all.append(conf_fun)

                    ece_log = {
                        "case": f"{case}",
                        "test_dataset": test_ds_name,
                        "ece_van": van_ece_all.item(),
                        "ece_fun": fun_ece_all.item(),
                        # "acc_van": acc_van.tolist(),
                        # "acc_fun": acc_van.tolist(),
                        # "conf_van": conf_van.item(),
                        # "conf_fun": conf_van.item(),
                    }
                    for i in torch.unique(targets)[1:]:
                        i_logits = torch.stack(
                            [logits[0, j][targets == i] for j in range(logits.size(1))]
                        )
                        i_ece, i_acc, i_conf = eceloss(
                            i_logits[None], targets[targets == i]
                        )
                        eces_van[i.item()].append(i_ece)

                        i_logits_fun = torch.stack(
                            [
                                logits_fun[0, j][targets == i]
                                for j in range(logits.size(1))
                            ]
                        )
                        i_ece_fun, i_acc_fun, i_conf_fun = eceloss(
                            i_logits_fun[None], targets[targets == i]
                        )
                        eces_fun[i.item()].append(i_ece_fun)
                        ece_log[f"ece_van_{i.item()}"] = i_ece.item()
                        ece_log[f"ece_fun_{i.item()}"] = i_ece_fun.item()

                    print(f"ECE: {van_ece_all.item()}, ECE FUN: {fun_ece_all.item()}")

                    eces_log.append(ece_log)
                    # exit()

                if SAVE_OUTPUTS:

                    torch.save(
                        merged_logits_all,
                        exp_path / f"{test_ds_name}_{case}_merged_logits.pt",
                    )
                    torch.save(
                        merged_logits_fun_all,
                        exp_path / f"{test_ds_name}_{case}_merged_logits_fun.pt",
                    )

                    targets = sitk.GetImageFromArray(targets_all.astype(np.uint8))
                    sitk.WriteImage(
                        targets, exp_path / f"{test_ds_name}_{case}_targets.nii.gz"
                    )
                    if test_ds_name == "Dataset001_amos":
                        ml = sitk.GetImageFromArray(
                            merged_logits_all.argmax(axis=1).astype(np.uint8)[0]
                        )
                        ml_fun = sitk.GetImageFromArray(
                            merged_logits_fun_all.argmax(axis=1).astype(np.uint8)[0]
                        )
                        u = sitk.GetImageFromArray(uncerts_all.astype(np.float32))
                        inputs = sitk.GetImageFromArray(inputs_all.astype(np.float32))

                        sitk.WriteImage(
                            ml, exp_path / f"{test_ds_name}_{case}_ml.nii.gz"
                        )
                        sitk.WriteImage(
                            ml_fun, exp_path / f"{test_ds_name}_{case}_ml_fun.nii.gz"
                        )
                        sitk.WriteImage(u, exp_path / f"{test_ds_name}_{case}_u.nii.gz")
                        sitk.WriteImage(
                            inputs, exp_path / f"{test_ds_name}_{case}_inputs.nii.gz"
                        )

                # for ds_name, ds_pred in preds.items():
                #     ds_pred = torch.cat(ds_pred, dim=0)
                #     ds_pred = sitk.GetImageFromArray(ds_pred.numpy().astype(np.uint8))
                #     sitk.WriteImage(
                #         ds_pred,
                #         exp_path / f"{test_ds_name}_{case}_{ds_name}_pred.nii.gz",
                #     )
                # for ds_name, logits in ds_logits.items():
                #     torch.save(
                #         logits,
                #         exp_path / f"{test_ds_name}_{case}_{ds_name}_logits.pt",
                #     )
            else:

                tps = torch.stack(tps).sum(0)
                fps = torch.stack(fps).sum(0)
                fns = torch.stack(fns).sum(0)

                vol = 2 * tps + fps + fns
                net_dice = 2 * tps / (vol + 1)

                # Ignore background
                net_dice = net_dice.detach().cpu().numpy()[1:]
                tps = tps.detach().cpu().float().numpy()[1:]
                vol = vol.detach().cpu().float().numpy()[1:]

                net_dice[~label_mask] = np.nan
                tps[~label_mask] = np.nan
                vol[~label_mask] = np.nan

                net_dice = {l: d for l, d in zip(train_labels, net_dice)}
                tp_dict = {f"{l}_tp": d for l, d in zip(train_labels, tps)}
                vol_dict = {f"{l}_vol": d for l, d in zip(train_labels, vol)}
                print("Case:", case)
                print("Net dice:", np.nanmean(list(net_dice.values())))

                net_dices.append(
                    {
                        "case": f"{case}",
                        "test_dataset": test_ds_name,
                        **net_dice,
                        **tp_dict,
                        **vol_dict,
                    }
                )

                print("Avg Dice:", np.nanmean(avg_dices))

                if SAVE_OUTPUTS:
                    merged_logits_all = (
                        torch.cat(merged_logits_all, dim=0)
                        .permute(1, 0, 2, 3)[None]
                        .numpy()
                    )

                    uncerts_all = torch.stack(uncerts_all).numpy()
                    inputs_all = torch.cat(inputs_all, dim=0).numpy()
                    targets_all = torch.cat(targets_all, dim=0).numpy()
                    targets = sitk.GetImageFromArray(targets_all.astype(np.uint8))

                    if ds_name == "Dataset001_amos":
                        ml = sitk.GetImageFromArray(
                            merged_logits_all.argmax(axis=1).astype(np.uint8)[0]
                        )
                        u = sitk.GetImageFromArray(uncerts_all.astype(np.float32))
                        inputs = sitk.GetImageFromArray(inputs_all.astype(np.float32))
                        sitk.WriteImage(
                            ml, exp_path / f"{test_ds_name}_{case}_ml.nii.gz"
                        )
                        sitk.WriteImage(u, exp_path / f"{test_ds_name}_{case}_u.nii.gz")
                        sitk.WriteImage(
                            inputs, exp_path / f"{test_ds_name}_{case}_inputs.nii.gz"
                        )
                        sitk.WriteImage(
                            targets, exp_path / f"{test_ds_name}_{case}_targets.nii.gz"
                        )
                    # # exit()

            df_dice = pd.DataFrame(dices)
            df_dice.to_csv(exp_path / f"{exp_name}_dice.csv")

            df_net_dice = pd.DataFrame(net_dices)
            if len(train_dataset_names) > 1:
                summarize_with_fun(df_net_dice, exp_path, exp_name)
            else:
                summarize(df_net_dice, exp_path, exp_name)
            df_net_dice.to_csv(exp_path / f"{exp_name}_net_dice.csv")

            if COMPUTE_ECE:
                eces_path = exp_path / "eces"
                eces_path.mkdir(exist_ok=True, parents=True)
                eces_van_all_np = torch.stack(eces_van_all).numpy()
                eces_fun_all_np = torch.stack(eces_fun_all).numpy()
                accs_van_all_np = torch.stack(accs_van_all).numpy()
                accs_fun_all_np = torch.stack(accs_fun_all).numpy()
                conf_van_all_np = torch.stack(conf_van_all).numpy()
                conf_fun_all_np = torch.stack(conf_fun_all).numpy()
                np.save(eces_path / f"eces_van_all_accs.npy", accs_van_all_np)
                np.save(eces_path / f"eces_fun_all_accs.npy", accs_fun_all_np)
                np.save(eces_path / f"eces_van_all_conf.npy", conf_van_all_np)
                np.save(eces_path / f"eces_fun_all_conf.npy", conf_fun_all_np)
                np.save(eces_path / f"eces_van_all.npy", eces_van_all_np)
                np.save(eces_path / f"eces_fun_all.npy", eces_fun_all_np)
                for i in range(1, n_cls + 1):
                    van_ece_np = np.array(eces_van[i])
                    fun_ece_np = np.array(eces_fun[i])
                    np.save(eces_path / f"eces_van_{i}.npy", van_ece_np)
                    np.save(eces_path / f"eces_fun_{i}.npy", fun_ece_np)

                df_ece_log = pd.DataFrame(eces_log)
                df_ece_log.to_csv(exp_path / f"{exp_name}_ece_log.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", "-m", default="federated")
    parser.add_argument("--ckpt", "-c", default="")
    parser.add_argument("--dataset", "-i", default="ts")
    parser.add_argument("-d", "--debug", action="store_true")

    args = parser.parse_args()
    # gpu_id = get_free_gpu()
    all_labels = [LABEL_SUBSETS[ds_name] for ds_name in LABEL_SUBSETS.keys()]
    all_labels = np.unique([xx for x in all_labels for xx in x])
    # all_labels = {l: i+1 for i, l in enumerate(all_labels)}
    n_cls = len(all_labels) + 1

    MODE = args.mode

    set_seed(SEED)
    # MODE = "federated"
    # MODE = "centralized"

    master_plan, master_dataset_json = get_plans_and_dataset_json(
        "Dataset003_TotalSegmentator", plans_identifier="nnUNetPlans"
    )

    if MODE == "centralized":
        USE_SAMPLING = False

        train_ids = [
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            # "Dataset010_Liver",
            # "Dataset011_Pancreas",
            # "Dataset012_Spleen",
            "Dataset003_TotalSegmentator",
        ]
        test_ids = deepcopy(train_ids)
        test_ids.append("Dataset001_amos")

        cfg = get_default_config("centralized")
        cfg.train.batch_size = 1
        cfg.train.dropout_p = 0.2
        # cfg.train.dropout_p = None  # type: ignore

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )
        patch_size = master_plan["configurations"]["2d"]["patch_size"]

        configuration = "2d"
        if cfg.train.dropout_p is not None:
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op"
            ] = f"torch.nn.Dropout{configuration[:2]}"
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op_kwargs"
            ] = {"p": cfg.train.dropout_p}

        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        # ckpt_root = Path(ROOT_PATH) / "output/25-02-18/central/22-46-54"
        # ckpt = torch.load(ckpt_root / "ckpt_260.pt", map_location=DEVICE)[
        #     "model"
        # ]

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-04/central/23-12-09"
        # # ckpt_num = '450'
        # # exp_name = "central_subset"

        # ckpt_num = "1000"
        # exp_name = "central_1000_wd_new_method"
        ckpt_root = Path(ROOT_PATH) / "output/25-04-03/central/13-52-43"
        # # ckpt_num = "2000"
        ckpt_num = "1000"
        exp_name = "central_1000_no_msd"
        # ckpt_root = Path(ROOT_PATH) / "output/25-03-13/central/16-06-07"
        # ckpt_num = "500"
        # exp_name = "central_500_wd"

        train_ckpt_path = ckpt_root / f"ckpt_{ckpt_num}.pt"
        ckpt = torch.load(train_ckpt_path, map_location=DEVICE)["model"]

        model_ckpt2 = {}
        for k, v in ckpt.items():
            # if "all_modules.2" in k:
            if False:
                model_ckpt2[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                model_ckpt2[k] = v

        model.load_state_dict(model_ckpt2)

        test(
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
        )

    elif MODE == "central_simple":

        train_ids = [
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            # "Dataset010_Liver",
            # "Dataset011_Pancreas",
            # "Dataset012_Spleen",
            "Dataset003_TotalSegmentator",
        ]
        test_ids = deepcopy(train_ids)
        test_ids.append("Dataset001_amos")

        cfg = get_default_config("centralized")
        cfg.train.batch_size = 1
        cfg.train.dropout_p = None  # type: ignore

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )
        patch_size = master_plan["configurations"]["2d"]["patch_size"]
        configuration = "2d"
        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-13/central/16-06-07"
        # # ckpt_num = "2000"
        # ckpt_num = "1300"
        # exp_name = "central_1300_simple_na"

        ckpt_root = Path(ROOT_PATH) / "output/25-04-03/central/13-52-43"
        # # ckpt_num = "2000"
        ckpt_num = "1000"
        exp_name = "central_1000_simple_no_msd"

        train_ckpt_path = ckpt_root / f"ckpt_{ckpt_num}.pt"
        ckpt = torch.load(train_ckpt_path, map_location=DEVICE)["model"]

        model_ckpt2 = {}
        for k, v in ckpt.items():
            if "all_modules.2" in k:
                # if False:
                model_ckpt2[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                model_ckpt2[k] = v

        model.load_state_dict(model_ckpt2)
        simple_inference(
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
        )
    elif MODE == "federated_simple":
        USE_SAMPLING = False
        train_ids = [
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            # "Dataset010_Liver",
            # "Dataset011_Pancreas",
            # "Dataset012_Spleen",
            "Dataset003_TotalSegmentator",
        ]
        test_ids = deepcopy(train_ids)
        test_ids.append("Dataset001_amos")

        cfg = get_default_config("federated")
        cfg.train.batch_size = 1
        cfg.train.dropout_p = None  # type: ignore

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )
        patch_size = master_plan["configurations"]["2d"]["patch_size"]
        configuration = "2d"
        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/federated/19-25-37"
        # ckpt_num = "2000"
        # exp_name = "fed_2000_simple_inf"

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/federated/19-25-37"
        # ckpt_num = "1600"
        # exp_name = "fed_1600_simple_inf_rerun"
        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/fedvar/18-26-02"
        # ckpt_num = "2000"
        # exp_name = "fed_deb_2000_simple"

        ckpt_root = Path(ROOT_PATH) / "output/25-04-03/federated/12-42-08"
        ckpt_num = "1000"
        exp_name = "fed_1000_simple_nomsd"

        # ckpt_root = Path(ROOT_PATH) / "output/25-04-06/fedvar/13-57-03"
        # ckpt_num = "1000"
        # exp_name = "fvf_1000_simple_nomsd"

        # ckpt_root = Path(ROOT_PATH) / "output/25-04-06/fedvar/13-56-16"
        # ckpt_num = "700"
        # exp_name = "fvl_700_simple_nomsd"

        train_ckpt_path = ckpt_root / f"ckpt_server_{ckpt_num}.pt"

        backbone_ckpt_og = torch.load(train_ckpt_path, map_location=DEVICE)
        backbone_ckpt = {}

        for k, v in backbone_ckpt_og.items():
            if "all_modules.2" in k:
                # if False:
                backbone_ckpt[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                backbone_ckpt[k] = v

        model.backbone.load_state_dict(backbone_ckpt)

        for train_ds_name in train_ids:
            cls_head_ckpt = torch.load(
                ckpt_root / f"ckpt_{train_ds_name}_{ckpt_num}.pt",
                map_location=DEVICE,
            )
            model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

        simple_inference(
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
        )

    elif MODE == "federated":
        USE_SAMPLING = False
        train_ids = [
            "Dataset003_TotalSegmentator",
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            # "Dataset010_Liver",
            # "Dataset011_Pancreas",
            # "Dataset012_Spleen",
        ]
        test_ids = deepcopy(train_ids)
        test_ids.append("Dataset001_amos")

        cfg = get_default_config("federated")
        cfg.train.batch_size = 1
        cfg.train.dropout_p = 0.2
        # cfg.train.dropout_p = None # type: ignore

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )
        patch_size = master_plan["configurations"]["2d"]["patch_size"]

        configuration = "2d"
        if cfg.train.dropout_p is not None:
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op"
            ] = f"torch.nn.Dropout{configuration[:2]}"
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op_kwargs"
            ] = {"p": cfg.train.dropout_p}

        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        # ckpt_root = Path(ROOT_PATH) / "output/25-02-19/federated/13-42-08"
        # ckpt_num = '250'
        # exp_name = "federated_subset"

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/federated/19-25-37"
        # ckpt_num = 550
        # exp_name = "fedavg_fixed_lr_550"
        # ckpt_num = 1150
        # exp_name = "fedavg_fixed_1150_nd"
        # ckpt_num = 2000
        # exp_name = "fedavg_2k_wd_patching"

        ckpt_root = Path(ROOT_PATH) / "feduniseg_output/25-04-03/federated/12-42-08"
        # ckpt_num = 1000
        ckpt_num = 1500
        exp_name = f"fedavg_{ckpt_num}_wd_nomsd_start_1"

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-04/federated/15-50-09/"
        # ckpt_num = '1000'
        # exp_name = "federated_1000"

        train_ckpt_path = ckpt_root / f"ckpt_server_{ckpt_num}.pt"

        backbone_ckpt_og = torch.load(train_ckpt_path, map_location=DEVICE)
        backbone_ckpt = {}

        for k, v in backbone_ckpt_og.items():
            # if "all_modules.2" in k:
            if False:
                backbone_ckpt[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                backbone_ckpt[k] = v

        model.backbone.load_state_dict(backbone_ckpt)

        for train_ds_name in train_ids:
            cls_head_ckpt = torch.load(
                ckpt_root / f"ckpt_{train_ds_name}_{ckpt_num}.pt",
                map_location=DEVICE,
            )
            model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

        test(
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
        )

    elif MODE == "fedvar":

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/fedvar/18-26-02"
        # ckpt_num = 550
        # exp_name = "fedvar_unit_debug_550"
        # USE_SAMPLING = False

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-08/fedvar/18-26-02"
        # ckpt_num = 2000
        # exp_name = "fedvar_debug_2000"
        # USE_SAMPLING = False

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-26/fedvar/17-35-08"
        # ckpt_num = 2000
        # exp_name = "fedvar_full_2000_dr"
        # USE_SAMPLING = False

        ckpt_root = Path(ROOT_PATH) / "feduniseg_output/25-04-06/fedvar/13-57-03"
        ckpt_num = "1500"
        exp_name = f"fvf_{ckpt_num}_sm_nomsd_start_4"
        USE_SAMPLING = True

        # /home/asim.ukaye/fed_learning/datasets/output/25-05-30/fedvar/11-30-59/ckpt_Dataset004_LiTS_1000.pt
        # ckpt_root = Path(ROOT_PATH) / "output/25-05-30/fedvar/11-30-59/"
        # ckpt_num = "1000"
        # exp_name = f"fvf_{ckpt_num}_sm_nomsd_rerun_seed_0"
        # USE_SAMPLING = True

        # ckpt_root = Path(ROOT_PATH) / "output/25-05-30/fedvar/09-54-53"
        # ckpt_num = "1500"
        # exp_name = f"fvf_{ckpt_num}_sm_nomsd_param_var"
        # USE_SAMPLING = True

        # ckpt_root = Path(ROOT_PATH) / "output/25-06-02/fedvar/17-34-33"
        # ckpt_num = "700"
        # exp_name = f"fvf_{ckpt_num}_sm_nomsd_gradrerun_seed_1"
        # USE_SAMPLING = True

        # ckpt_root = Path(ROOT_PATH) / "output/25-06-03/fedvar/12-57-55"
        # # ckpt_num = "500"
        # ckpt_num = "1500"
        # exp_name = f"fvf_{ckpt_num}_sm_nomsd_rerun_seed_2"
        # USE_SAMPLING = True



        # ckpt_root = Path(ROOT_PATH) / "output/25-04-06/fedvar/13-56-16"
        # ckpt_num = "1500"
        # exp_name = f"fvl_{ckpt_num}_wd_nomsd"
        # USE_SAMPLING = False

        # ckpt_root = Path(ROOT_PATH) / "output/25-03-26/fedvar/17-38-27"
        # ckpt_num = 2000
        # exp_name = "fedvar_layer_2000_dr"
        # USE_SAMPLING = False

        # High variance
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-23/fedvar/21-55-40"
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-28/fedvar/10-06-24"
        # ckpt_num = 120
        # exp_name = "fedvar_subset_highvar_full_nd"
        # USE_SAMPLING = False

        # Layer wise
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-26/fedvar/21-59-03"
        # ckpt_num = 80
        # exp_name = "fedvar_subset_highvar_nd_layerwise"

        # Scalar
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-26/fedvar/22-51-59"
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-28/fedvar/08-33-54"
        # ckpt_num = 120
        # exp_name = "fedvar_subset_highvar_wd_scalar"
        # USE_SAMPLING = False

        # ckpt_root = Path(ROOT_PATH) / "output/25-02-26/fedvar/22-51-59"
        # ckpt_num = 80
        # exp_name = "fedvar_subset_highvar_nd_scalar"
        # USE_SAMPLING = False

        # # Layer wise with dropout
        # ckpt_root = Path(ROOT_PATH) / "output/25-02-26/fedvar/21-59-03"
        # ckpt_num = 80
        # USE_SAMPLING = False
        # exp_name = "fedvar_subset_highvar_wd_layerwise"

        train_ids = [
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            # "Dataset010_Liver",
            # "Dataset011_Pancreas",
            # "Dataset012_Spleen",
            "Dataset003_TotalSegmentator",
        ]
        test_ids = deepcopy(train_ids)
        test_ids.append("Dataset001_amos")

        cfg = get_default_config("federated")
        cfg.train.batch_size = 1
        if USE_SAMPLING:
            cfg.train.dropout_p = None  # type: ignore
        else:
            cfg.train.dropout_p = 0.2
        # cfg.train.dropout_p = 0.2  # type: ignore

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )
        patch_size = master_plan["configurations"]["2d"]["patch_size"]

        configuration = "2d"
        if cfg.train.dropout_p is not None:
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op"
            ] = f"torch.nn.Dropout{configuration[:2]}"
            master_trainer.configuration_manager.network_arch_init_kwargs[
                "dropout_op_kwargs"
            ] = {"p": cfg.train.dropout_p}

        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        train_ckpt_path = ckpt_root / f"ckpt_server_{ckpt_num}.pt"

        backbone_ckpt_og = torch.load(train_ckpt_path, map_location=DEVICE)

        backbone_ckpt = {}
        for k, v in backbone_ckpt_og.items():
            if "all_modules.2" in k:
            # if False:
                backbone_ckpt[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                backbone_ckpt[k] = v

        model.backbone.load_state_dict(backbone_ckpt)

        for train_ds_name in train_ids:
            cls_head_ckpt = torch.load(
                ckpt_root / f"ckpt_{train_ds_name}_{ckpt_num}.pt",
                map_location=DEVICE,
            )
            model.cls_heads[train_ds_name].load_state_dict(cls_head_ckpt)

        if USE_SAMPLING:
            backbone_variance_dict = torch.load(
                ckpt_root / f"variances/ckpt_server_var_{ckpt_num}.pt",
                map_location=DEVICE,
            )
            variance_ckpt = deepcopy(backbone_ckpt)
            for k in variance_ckpt.keys():
                # ic(k, id(variance_ckpt[k]))

                # if k in backbone_variance_dict:
                #     surrogate_ckpt[k] = backbone_variance_dict[k]
                # else:
                if "decoder.encoder" in k:
                    k1 = k.replace("decoder.", "")
                else:
                    k1 = k

                if "all_modules.1" in k1:
                    variance_ckpt[k] = backbone_variance_dict[
                        k1.replace("all_modules.1", "norm")
                    ]
                elif "all_modules.0" in k1:
                    variance_ckpt[k] = backbone_variance_dict[
                        k1.replace("all_modules.0", "conv")
                    ]
                else:
                    variance_ckpt[k] = backbone_variance_dict[k1]

            # ic(all(k1 == k2 for k1, k2 in zip(backbone_ckpt.keys(), variance_ckpt.keys())))

            # for k, v in backbone_ckpt:
            #     if k in backbone_variance_dict:
            #         backbone_variance[k] = backbone_variance_dict[k]

            # ic(backbone_variance.keys())
            # ic(backbone_ckpt.keys())
            # for k, v in backbone_variance.items():
            #     ic(backbone_ckpt[k].shape)
            # exit()
        else:
            variance_ckpt = None

        test(
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
            model_variance=variance_ckpt,  # type: ignore
        )
    
    elif MODE == "individual":
        USE_SAMPLING = False
        OG_VERSION = False

        assert args.dataset in DATASET_SHORTHANDS.keys()
        train_ids = [DATASET_SHORTHANDS[args.dataset]]

        test_ids = [
            "Dataset003_TotalSegmentator",
            "Dataset004_LiTS",
            "Dataset005_BTCV",
            "Dataset008_abdomenct1k",
            "Dataset009_learn2reg",
            "Dataset010_Liver",
            "Dataset011_Pancreas",
            "Dataset012_Spleen",
            "Dataset001_amos",
        ]

   

        cfg = get_single_config(args.dataset)
        # master_plan, master_dataset_json = get_plans_and_dataset_json(
        #     cfg.master_dataset, plans_identifier="nnUNetPlans"
        # )

        master_trainer = get_trainer(
            configuration=cfg.nnunet.config,
            fold=cfg.fold,
            device=torch.device(DEVICE),
            plans=master_plan,
            dataset_json=master_dataset_json,
        )

        patch_size = master_plan["configurations"]["2d"]["patch_size"]
        master_trainer.initialize()

        model = Model(
            dataset_names=train_ids,
            label_subsets=LABEL_SUBSETS,
            master_trainer=master_trainer,
        ).to(DEVICE)

        CKPT_DICT = {
            "amos": "output/25-02-18/single/17-24-41",
            "ts": "output/25-02-21/single/21-09-48",
            "l2r": "output/25-02-19/single/16-51-41",
            "btcv": "output/cscc/16-36-24",
            "lits": "output/25-02-19/single/18-00-52",
            "abd1k": "output/25-02-18/single/16-45-17",
            "liver": "output/cscc/17-28-17",
            "pancreas": "output/cscc/16-48-11",
            "spleen": "output/25-02-19/single/16-42-38",
        }

        if args.ckpt:
            ckpt_root = Path(ROOT_PATH) / args.ckpt
            exp_name = f"individual_central_{args.dataset}_test"

        else:
            ckpt_root = (
                Path(ROOT_PATH)
                / CKPT_DICT[args.dataset]
                / DATASET_SHORTHANDS[args.dataset]
                / "nnUNetTrainer__nnUNetPlans__2d"
                / "fold_0"
            )
            exp_name = f"individual_{args.dataset}"

        train_ckpt_path = ckpt_root / "checkpoint_best.pth"
        model_ckpt = torch.load(train_ckpt_path, map_location=DEVICE)["network_weights"]

        model_ckpt2 = {}
        for k, v in model_ckpt.items():
            if "all_modules.2" in k:
                model_ckpt2[k.replace("all_modules.2", "all_modules.1")] = v
            else:
                model_ckpt2[k] = v

        k1s = list(model_ckpt.keys())
        k2s = list(model_ckpt2.keys())

        backbone_ckpt = {
            f"_orig_mod.{k}": v
            for k, v in model_ckpt2.items()
            if not k.startswith("decoder.seg_layers")
        }
        cls_head_ckpt = {
            k.removeprefix("decoder.seg_layers."): v
            for k, v in model_ckpt2.items()
            if k.startswith("decoder.seg_layers")
        }

        model.backbone.load_state_dict(backbone_ckpt)
        model.cls_heads[train_ids[0]].load_state_dict(cls_head_ckpt)

        test(
            exp_name=exp_name,
            label_subsets=LABEL_SUBSETS,
            model=model,
            train_dataset_names=train_ids,
            test_dataset_names=test_ids,
            data_src=Path(ROOT_PATH),
            train_ckpt_path=train_ckpt_path,
        )

    else:
        raise ValueError("Invalid mode")


