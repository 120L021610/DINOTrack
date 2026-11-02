"""Focal loss on the center heatmap (Eq. 9 first term).

[UNSPECIFIED] internal alpha/beta: DINOTrack cites Lin et al. [26]; values
(alpha=2, beta=4) follow the OSTrack implementation (lib/utils/focal_loss.py),
per user instruction. Distinct from the Eq. 9 combination weights
(focal 1.0 / L1 5.0 / GIoU 2.0, §4.1) — see DinoTrackActor in train/actor.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # prediction: (B,1,H,W) sigmoid'd score map; target: same-shape gaussian map
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.beta)
        prediction = torch.clamp(prediction, 1e-12)
        positive_loss = (torch.log(prediction)
                         * torch.pow(1 - prediction, self.alpha) * positive_index)
        negative_loss = (torch.log(1 - prediction)
                         * torch.pow(prediction, self.alpha)
                         * negative_weights * negative_index)
        num_positive = positive_index.sum()
        if num_positive == 0:
            return -negative_loss.sum()
        return -(positive_loss.sum() + negative_loss.sum()) / num_positive
