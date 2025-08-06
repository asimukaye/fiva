from copy import deepcopy

import torch
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform

# LabelTransform is used to transform the labels in the dataset to the wanted labels in the dataset.


class LabelTransform(SegOnlyTransform):
    def __init__(self, all_labels_in_dataset, wanted_labels_in_dataset):
        self.all_labels_in_dataset = {
            int(k): v for k, v in all_labels_in_dataset.items()
        }
        self.wanted_labels_in_dataset = wanted_labels_in_dataset
        super().__init__()

    def _apply_to_segmentation(
        self, segmentation: torch.Tensor, **params
    ) -> torch.Tensor:
        if type(segmentation) == list:
            new_seg_l = [torch.zeros_like(s) for s in segmentation]
            seg_l = segmentation
        else:
            new_seg_l = [torch.zeros_like(segmentation)]
            seg_l = [segmentation]

        for l, nl in zip(seg_l, new_seg_l):
            for i in range(1, len(self.all_labels_in_dataset)):
                al = self.all_labels_in_dataset[i]
                if al in self.wanted_labels_in_dataset:
                    nl[l == i] = self.wanted_labels_in_dataset[al]
                    # ic(self.all_labels_in_dataset[i], self.wanted_labels_in_dataset[al])

        if type(segmentation) == list:
            return [l.float() for l in new_seg_l]
        else:
            return [l.float() for l in new_seg_l][0]


class LabelTransformold:
    def __init__(self, all_labels_in_dataset, wanted_labels_in_dataset):

        self.all_labels_in_dataset = {
            int(k): v for k, v in all_labels_in_dataset.items()
        }
        self.wanted_labels_in_dataset = wanted_labels_in_dataset

    def __call__(self, data):
        label = data["target"]
        if type(label) == list:
            new_label = [torch.zeros_like(l) for l in label]
        else:
            new_label = [torch.zeros_like(label)]
            label = [label]

        for l, nl in zip(label, new_label):
            for i in range(1, len(self.all_labels_in_dataset)):
                al = self.all_labels_in_dataset[i]
                if al in self.wanted_labels_in_dataset:
                    nl[l == i] = self.wanted_labels_in_dataset[al]
        data["target"] = [l.float() for l in new_label]
        data["cur_task"] = list(self.wanted_labels_in_dataset.keys())
        return data


class LabelTransformNNUnetold(LabelTransform):
    def __call__(self, data, target, keys, properties):
        _data = super().__call__({"target": target})
        _data["data"] = data.float()
        return _data


# Only to include fedprox


class FednnUNetTrainer(nnUNetTrainer):

    def __init__(self, fedprox_mu=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fedprox_mu = fedprox_mu

    def on_train_epoch_start(self):
        super().on_train_epoch_start()

        # Ignore the decoder segmentation layers
        self.grad_org = deepcopy(self.network.state_dict())
        self.grad_org = {
            k: v
            for k, v in self.grad_org.items()
            if not k.startswith("decoder.seg_layers")
        }

    # THis is re-written just for fedprox :')
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            # FEDPROX
            if self.fedprox_mu is not None:
                reg_loss = 0.0
                for name, param in self.network.named_parameters():
                    if "weight" in name and name in self.grad_org:
                        reg_loss += torch.norm(param - self.grad_org[name], 2)
                reg_loss = reg_loss * 0.5 * self.fedprox_mu
                l += reg_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if self.fedprox_mu is not None:
                reg_loss = 0.0
                for name, param in self.network.named_parameters():
                    if "weight" in name and name in self.grad_org:
                        reg_loss += torch.norm(param - self.grad_org[name], 2)
                reg_loss = reg_loss * 0.5 * self.fedprox_mu
                l += reg_loss

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
