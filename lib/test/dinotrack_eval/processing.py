"""Crop / normalization helpers for the DINOTrack evaluator — verbatim from
dinotrack_v1/data/processing.py (sample_target, _to_tensor_normalize).
The framework's lib/train/data/processing_utils.sample_target is NOT reused
because the v1 tracker's crop box return convention is load-bearing for
number continuity with the v1 results."""

from __future__ import annotations

import math

import cv2 as cv
import numpy as np
import torch


def sample_target(im: np.ndarray, target_bb, search_area_factor: float,
                  output_sz: int, return_crop_box: bool = False):
    """Square crop centered at target_bb [x,y,w,h], side ~ factor*sqrt(wh).
    Returns crop (output_sz, output_sz, 3) uint8 RGB and resize_factor;
    optionally also the crop box [x1, y1, side] in original-image units
    (needed by the inference tracker to map the prediction back, §4.1)."""
    x, y, w, h = target_bb.tolist() if not isinstance(target_bb, list) else target_bb
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        raise Exception("Too small bounding box.")
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    x1_pad, x2_pad = max(0, -x1), max(x1 + crop_sz - im.shape[1] + 1, 0)
    y1_pad, y2_pad = max(0, -y1), max(y1 + crop_sz - im.shape[0] + 1, 0)
    im_crop = im[y1 + y1_pad:y1 + crop_sz - y2_pad, x1 + x1_pad:x1 + crop_sz - x2_pad, :]
    im_crop = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                cv.BORDER_CONSTANT)
    resize_factor = output_sz / crop_sz
    im_crop = cv.resize(im_crop, (output_sz, output_sz), interpolation=cv.INTER_LINEAR)
    if return_crop_box:
        # padded crop covers [x1, x1+crop_sz] x [y1, y1+crop_sz] in original units
        return im_crop, resize_factor, [x1, y1, crop_sz]
    return im_crop, resize_factor


def _to_tensor_normalize(img: np.ndarray, mean, std) -> torch.Tensor:
    x = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)
    return (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
