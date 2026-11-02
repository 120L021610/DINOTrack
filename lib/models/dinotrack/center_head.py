"""Center-based prediction head — §3.5.

§3.5: "a center-based prediction head ... three parallel branches that predict
the classification score map P in R^{1 x (Hx/P) x (Wx/P)}, the size map
S in R^{2 x (Hx/P) x (Wx/P)}, and the local offset map O in R^{2 x (Hx/P) x
(Wx/P)} to compensate for discretization errors. The target bounding box is
recovered by locating peaks on the score map and combining them with the
corresponding size and offset predictions."

All three maps therefore have the same 16x16 spatial resolution for a
256x256 search image and P=16.  This also matches OSTrack CenterPredictor.

Conv stack structure (4x conv-bn-relu + 1x1, channels 256->32), size sigmoid
encoding and offset convention follow OSTrack CenterPredictor (reference
implementation per user instruction; the paper says only "Following common
practice", §3.5). [UNSPECIFIED] weight init: xavier_uniform for convs (OSTrack).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(in_planes: int, out_planes: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def _stack(in_planes: int, channel: int, first_stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        _conv(in_planes, channel, stride=first_stride),
        _conv(channel, channel // 2),
        _conv(channel // 2, channel // 4),
        _conv(channel // 4, channel // 8),
        nn.Conv2d(channel // 8, 1, kernel_size=1),
    )


def _stack2(in_planes: int, channel: int, first_stride: int = 1) -> nn.Sequential:
    """Same stack, but final 1x1 conv outputs 2 channels (size / offset)."""
    return nn.Sequential(
        _conv(in_planes, channel, stride=first_stride),
        _conv(channel, channel // 2),
        _conv(channel // 2, channel // 4),
        _conv(channel // 4, channel // 8),
        nn.Conv2d(channel // 8, 2, kernel_size=1),
    )


class CenterHead(nn.Module):
    def __init__(self, in_channels: int, channel: int, feat_size: int):
        super().__init__()
        self.feat_sz = feat_size  # Hx/P = 16 (§3.5, 256/16)
        self.score_head = _stack(in_channels, channel)          # (B,1,16,16)
        self.offset_head = _stack2(in_channels, channel)        # (B,2,16,16)
        self.size_head = _stack2(in_channels, channel)            # (B,2,16,16)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # OSTrack CenterPredictor init

    def forward(self, x: torch.Tensor):
        """x: (B, C, 16, 16) refined search features.
        Returns score_map (B,1,16,16) sigmoid'd, bbox (B,4) [cx,cy,w,h] in crop
        normalized [0,1], size_map (B,2,16,16), offset_map (B,2,16,16)."""
        score = self._sigmoid(self.score_head(x))               # Eq. 9 focal target
        size = self._sigmoid(self.size_head(x))                 # w,h normalized [0,1]
        offset = self.offset_head(x)                            # sub-cell correction (raw)
        bbox = self.cal_bbox(score, size, offset)
        return score, bbox, size, offset

    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        # OSTrack clamps to keep focal-loss log finite
        return torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)

    def cal_bbox(self, score_map: torch.Tensor, size_map: torch.Tensor,
                 offset_map: torch.Tensor) -> torch.Tensor:
        """§3.5 "recovered by locating peaks on the score map and combining
        them with the corresponding size and offset predictions"."""
        b = score_map.shape[0]
        flat_idx = torch.argmax(score_map.flatten(1), dim=1)     # (B,) peak cell
        idx_y = flat_idx // self.feat_sz                          # (B,)
        idx_x = flat_idx % self.feat_sz
        idx16 = flat_idx.view(b, 1, 1).expand(b, 2, 1)
        offset = offset_map.flatten(2).gather(dim=2, index=idx16).squeeze(-1)  # (B,2)
        size = size_map.flatten(2).gather(dim=2, index=idx16).squeeze(-1)     # (B,2)
        # OSTrack center convention: (cell_index + offset) / feat_sz
        cx = (idx_x.to(torch.float32) + offset[:, 0]) / self.feat_sz
        cy = (idx_y.to(torch.float32) + offset[:, 1]) / self.feat_sz
        return torch.stack([cx, cy, size[:, 0], size[:, 1]], dim=-1)  # (B,4) cxcywh [0,1]
