"""Hann window for inference score-map smoothing.

Faithful port of OSTrack lib/test/utils/hann.py (hann1d/hann2d, centered=True)
— OSTrack's tracker applies `output_window * pred_score_map` unconditionally
before `cal_bbox` (lib/test/tracker/ostrack.py:87-89).

[UNSPECIFIED] in DINOTrack (§4.1 says only "no test-time adaptation"): per
user instruction (2026-09-19) the inference default matches the OSTrack
convention — Hann ON, multiplicative. config.test.hann can disable it.
"""

from __future__ import annotations

import math

import torch


def hann1d(sz: int, centered: bool = True) -> torch.Tensor:
    """1D cosine window (OSTrack hann.py)."""
    if centered:
        return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1))
                                    * torch.arange(1, sz + 1).float()))
    w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2))
                             * torch.arange(0, sz // 2 + 1).float()))
    return torch.cat([w, w[1:sz - sz // 2].flip((0,))])


def hann2d(sz: int, centered: bool = True) -> torch.Tensor:
    """2D cosine window, shape (1, 1, sz, sz)."""
    return (hann1d(sz, centered).reshape(1, 1, -1, 1)
            * hann1d(sz, centered).reshape(1, 1, 1, -1))
