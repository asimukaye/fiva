import torch
from torch.nn import Module
from torch import Tensor
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import (
    compute_steps_for_sliding_window,
    compute_gaussian,
)
from tqdm import tqdm
import numpy as np


def prediction_with_patching(
    model: Module,
    input_image: Tensor,
    patch_size: list[int],
    device: str,
):
    step_size = 0.5
    prediction_heads = model.cls_heads.keys()
    # num_segmentation_heads = model.cls_h_heads
    use_gaussian = True

    data, slicer_revert_padding = pad_nd_image(
        input_image, tuple(patch_size), "constant", {"value": 0}, True
    )

    slicers = []
    image_size = data.shape[1:]

    steps = compute_steps_for_sliding_window(
        image_size[1:], tuple(patch_size), step_size
    )

    # print(
    #     f"n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size i {image_size}, tile_size {patch_size}, tile_step_size {step_size}\nsteps:\n{steps}"
    # )
    for d in range(image_size[0]):
        for sx in steps[0]:
            for sy in steps[1]:
                slicers.append(
                    tuple(
                        [
                            slice(None),
                            d,
                            *[
                                slice(si, si + ti)
                                for si, ti in zip((sx, sy), patch_size)
                            ],
                        ]
                    )
                )

    predicted_logits_dict = {}
    n_predictions_dict = {}

    # predicted_logits = n_predictions
    prediction = gaussian = workon = None

    results_device = device

    if use_gaussian:
        gaussian = compute_gaussian(
            tuple(patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )
    else:
        gaussian = 1

    try:
        # move data to device

        data = data.to(results_device)  # type: ignore

        # preallocate arrays
        for head in prediction_heads:
            num_segmentation_heads = model.cls_heads[head][-1].out_channels
            # ic(num_segmentation_heads)

            predicted_logits_dict[head] = torch.zeros(
                (num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            n_predictions_dict[head] = torch.zeros(
                data.shape[1:], dtype=torch.half, device=results_device
            )
        # predicted_logits = torch.zeros(
        #     (num_segmentation_heads, *data.shape[1:]),
        #     dtype=torch.half,
        #     device=results_device,
        # )

        # n_predictions = torch.zeros(
        #     data.shape[1:], dtype=torch.half, device=results_device
        # )

        for sl in slicers:
            workon = data[sl][None]
            workon = workon.to(device)

            prediction_out = model(workon)

            for head in prediction_heads:
                prediction = prediction_out[head][0]
                # ic(prediction.shape)
                if use_gaussian:
                    prediction *= gaussian
                predicted_logits_dict[head][sl] += prediction
                n_predictions_dict[head][sl[1:]] += gaussian
            # if use_gaussian:
            #     prediction *= gaussian
            # predicted_logits[sl] += prediction
            # n_predictions[sl[1:]] += gaussian

        for head in prediction_heads:
            predicted_logits_dict[head] /= n_predictions_dict[head]

            # predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits_dict[head])):
                raise RuntimeError(
                    "Encountered inf in predicted array. Aborting... If this problem persists, "
                    "reduce value_scaling_factor in compute_gaussian or increase the dtype of "
                    "predicted_logits to fp32"
                )
    except Exception as e:
        del predicted_logits_dict, n_predictions_dict, prediction, gaussian, workon
        torch.cuda.empty_cache()
        raise e

    for head in prediction_heads:
        predicted_logits = predicted_logits_dict[head]

        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        predicted_logits_dict[head] = predicted_logits
    # with torch.no_grad():
    #     input_image = input_image.to(device)
    #     pred = model(input_image)

    return predicted_logits_dict
