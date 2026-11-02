"""Bounding-box helpers.

Ported from OSTrack lib/utils/box_ops.py (reference implementation per user
instruction; §3.5 uses GIoU loss [33] and L1 on decoded boxes).
"""

from __future__ import annotations

import torch
from torchvision.ops import generalized_box_iou as _tv_giou  # reference impl [33]


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    c_x, c_y, w, h = x.unbind(-1)
    return torch.stack([c_x - 0.5 * w, c_y - 0.5 * h, c_x + 0.5 * w, c_y + 0.5 * h], dim=-1)


def box_xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x1, y1, w, h = x.unbind(-1)
    return torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU [33], inputs (N,4) xyxy -> element-wise (N,) in [-1,1].
    torchvision returns the full pairwise matrix; we take its diagonal (the
    OSTrack-style per-pair loss convention)."""
    return torch.diagonal(_tv_giou(boxes1, boxes2), dim1=-2, dim2=-1)  # (N,)


def giou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Eq. 9 term L_GIoU = mean(1 - GIoU). Returns (scalar loss, iou tensor)."""
    giou = generalized_box_iou(boxes1, boxes2)                # (N,)
    area1, area2 = _box_area(boxes1), _box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])              # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])              # (N,2)
    wh = (rb - lt).clamp(min=0)                               # (N,2)
    inter = wh[:, 0] * wh[:, 1]                               # (N,)
    union = (area1 + area2 - inter).clamp(min=1e-12)
    iou = inter / union
    return (1 - giou).mean(), iou


def clip_box(box: list, H: int, W: int, margin: float = 0.0) -> list:
    """Keep [x,y,w,h] inside the frame with `margin` px (OSTrack clip_box;
    [UNSPECIFIED] margin value in DINOTrack — OSTrack uses 10)."""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]
