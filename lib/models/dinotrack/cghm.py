"""Class-Guided Heatmap Modulation (CGHM) — §3.3, Eq. 4–5, Fig. 2.

§3.3: "we first compute the spatial similarity map by performing a pixel-wise
dot product operation on the semantic prototype c^l and the search feature
x^l ... apply a temperature-scaled softmax:
    M^l = Softmax( x^l (c^l)^T / tau )           (Eq. 4)
where M^l in R^{(Hx/P)x(Wx/P)} ... tau ... empirically set it to 0.07.
... modulate the original projected features x^l using a residual enhancement
scheme:
    x~^l = x^l (.) (1 + M^l)                     (Eq. 5)"

Only SEARCH features are modulated (template path is Conv->Concat, Fig. 1b).

``similarity="dot"`` is the literal Eq. 4 implementation.  The optional
``"cosine"`` mode L2-normalizes only the two operands used to form the
similarity map.  It leaves the feature being modulated untouched, so Eq. 5
remains exactly ``x * (1 + M)``.  Cosine mode is an explicitly documented
legacy experiment, not a normalization stated in the paper; it is not used by
the formal recipe because its measured LaSOT result was worse than dot-CGHM.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGHM(nn.Module):
    def __init__(self, tau: float = 0.07, similarity: str = "dot"):
        super().__init__()
        self.tau = tau  # §3.3
        if similarity not in {"dot", "cosine"}:
            raise ValueError(f"unknown CGHM similarity: {similarity!r}")
        self.similarity = similarity

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, D, H, W) projected search feature of one layer;
        c: (B, D) projected template class token of the same layer.
        Returns (modulated x~ (B, D, H, W), heatmap M (B, 1, H, W))."""
        # Eq. 4 similarity over the channel dimension. Dot mode is the PDF's
        # literal formula. Cosine mode changes only the heatmap operands; the
        # original x is retained below for Eq. 5 residual modulation.
        if self.similarity == "cosine":
            x_sim = F.normalize(x, p=2, dim=1, eps=1e-6)
            c_sim = F.normalize(c, p=2, dim=1, eps=1e-6)
        else:
            x_sim, c_sim = x, c
        s = (x_sim * c_sim.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        # softmax over the spatial (flattened) dimension with temperature
        b, h, w = s.shape
        M = torch.softmax(s.flatten(1) / self.tau, dim=1).view(b, 1, h, w)
        # Eq. 5 residual enhancement, broadcast over D
        return x * (1.0 + M), M
