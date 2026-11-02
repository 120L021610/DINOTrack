"""CenterNet-style gaussian center labels for the score map (Eq. 9 focal-loss term).

[UNSPECIFIED] DINOTrack §3.5 says only "focal loss on the classification branch";
label generation follows OSTrack lib/utils/heapmap_utils.py (per user instruction):
gaussian radius from CornerNet min-overlap formula (min_overlap=0.7), one channel
per batch element. Ported from OSTrack.
"""

from __future__ import annotations

import numpy as np
import torch


def generate_heatmap(bboxes: torch.Tensor, feat_sz: int = 16) -> torch.Tensor:
    """bboxes: (B,4) normalized [x1,y1,w,h] in crop units [0,1].
    Returns (B, feat_sz, feat_sz) gaussian score maps, peak=1 at target center."""
    gt_scoremap = torch.zeros(bboxes.shape[0], feat_sz, feat_sz)
    bbox = bboxes * feat_sz                        # normalized -> feature cells
    wh = bbox[:, 2:]                               # (B,2) w,h in cells
    centers_int = (bbox[:, :2] + wh / 2).round()   # (B,2) center in cells
    for i in range(bboxes.shape[0]):
        radius = _gaussian_radius(wh[i:i + 1], min_overlap=0.7)
        radius = int(torch.clamp_min(radius, 0).item())
        _draw_gaussian(gt_scoremap[i], centers_int[i], radius)
    return gt_scoremap.to(bboxes.device)


def _gaussian_radius(box_size: torch.Tensor, min_overlap: float) -> torch.Tensor:
    """CornerNet "bug-version" as shipped in OSTrack (kept for exact behavior)."""
    width, height = box_size[..., 0], box_size[..., 1]
    a1, b1 = 1, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + torch.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + torch.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + torch.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return torch.min(r1, torch.min(r2, r3))


def _draw_gaussian(fmap: torch.Tensor, center: torch.Tensor, radius: int, k: float = 1.0) -> None:
    diameter = 2 * radius + 1
    x = np.arange(0, diameter, dtype=np.float32)
    sigma = diameter / 6.0
    xx = (x - x[diameter // 2]) ** 2 / (2 * sigma**2)
    gauss1d = np.exp(-xx)
    g = np.outer(gauss1d, gauss1d)
    gauss = g.astype(np.float32)
    gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
    cx, cy = int(center[0]), int(center[1])
    height, width = fmap.shape[:2]
    left, right = min(cx, radius), min(width - cx, radius + 1)
    top, bottom = min(cy, radius), min(height - cy, radius + 1)
    masked = fmap[cy - top: cy + bottom, cx - left: cx + right]
    g_masked = torch.from_numpy(gauss[radius - top: radius + bottom,
                                      radius - left: radius + right])
    # Match CenterNet/OSTrack exactly.  In particular, radius == 0 must draw
    # the single positive center pixel.  The previous left/top > 0 guard
    # silently produced an all-negative target for every radius-0 object.
    if min(masked.shape) > 0 and min(g_masked.shape) > 0:
        torch.maximum(masked, g_masked * k, out=masked)
