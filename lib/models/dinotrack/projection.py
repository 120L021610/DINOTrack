"""Channel-alignment projection phi(.) — §3.2 Eq. 1.

§3.2 "we employ a shared projection layer phi(.) (implemented as a 1 x 1
convolution) to map the channel dimension from C to D. Consequently...
x^l = phi(f_x^l), c^l = phi(t_cls^l)   (Eq. 1)"

[UNSPECIFIED] whether "shared" spans the K layers / branches. Literal
reading (one phi everywhere) is the default; NeckCfg.per_layer_proj=True
gives one phi per layer as an ablation.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, c_in: int, d_out: int, num_layers: int, per_layer: bool = False):
        super().__init__()
        one = nn.Conv2d(c_in, d_out, kernel_size=1)
        if per_layer:
            # [UNSPECIFIED] per-layer independent projections (ablation option)
            self.proj = nn.ModuleList([_clone(one) for _ in range(num_layers)])
        else:
            self.proj = nn.ModuleList([one])
        self.per_layer = per_layer

    def _p(self, layer_idx: int) -> nn.Conv2d:
        return self.proj[layer_idx] if self.per_layer else self.proj[0]

    def forward_map(self, f: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        # f: (B, C, H, W) -> x: (B, D, H, W)     Eq. 1 x^l = phi(f^l)
        return self._p(layer_idx)(f)

    def forward_cls(self, t: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        # t: (B, C) -> c: (B, D)                 Eq. 1 c^l = phi(t_cls^l)
        y = self._p(layer_idx)(t.unsqueeze(-1).unsqueeze(-1))  # (B, D, 1, 1)
        return y.squeeze(-1).squeeze(-1)


def _clone(module: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(module)
