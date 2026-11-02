"""Standard and differential attention used by the DINOTrack interaction block.

Differential attention follows Eq. 6--7 of DINOTrack and the released
Differential Transformer implementation: Q/K contain two 64-dimensional
sub-heads, while the shared value has 128 dimensions per differential head.
The differential result is RMS-normalized per head before the output
projection. Standard-attention ablations use a genuine 768-dimensional MHA.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from lib.models.dinotrack.dino_config import DiffBlockCfg
from lib.models.dinotrack.layers import RMSNorm


class StandardAttention(nn.Module):
    """Full-width multi-head self-attention for the Table 5 baseline."""

    def __init__(self, cfg: DiffBlockCfg, dim_in: int = 768):
        super().__init__()
        if dim_in % cfg.n_heads:
            raise ValueError(f"dim_in={dim_in} must be divisible by n_heads={cfg.n_heads}")
        self.attn = nn.MultiheadAttention(dim_in, cfg.n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False)[0]


class DifferentialAttention(nn.Module):
    """Differential attention with six 2x64 Q/K heads and six 128-d V heads."""

    def __init__(self, cfg: DiffBlockCfg, dim_in: int = 768):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.lambda_init = cfg.lambda_init
        qk_dim = 2 * self.n_heads * self.head_dim
        value_dim = qk_dim
        if qk_dim != dim_in:
            raise ValueError(
                "differential attention requires dim_in == 2*n_heads*head_dim "
                f"({dim_in} != {qk_dim})"
            )

        self.q_proj = nn.Linear(dim_in, qk_dim, bias=False)
        self.k_proj = nn.Linear(dim_in, qk_dim, bias=False)
        self.v_proj = nn.Linear(dim_in, value_dim, bias=False)
        self.o_proj = nn.Linear(value_dim, dim_in, bias=False)

        # The reference implementation shares each lambda vector across heads.
        self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim).normal_(std=cfg.lambda_std))
        self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim).normal_(std=cfg.lambda_std))
        self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim).normal_(std=cfg.lambda_std))
        self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim).normal_(std=cfg.lambda_std))
        self.subln = RMSNorm(2 * self.head_dim, eps=cfg.norm_eps)

    def _lambda(self) -> torch.Tensor:
        return (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
                - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
                + self.lambda_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x, debug=False)

    def forward_debug(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._forward(x, debug=True)

    def _forward(self, x: torch.Tensor, debug: bool):
        b, length, _ = x.shape
        # Q/K: (B,h,2,L,64); V: (B,h,L,128).
        q = self.q_proj(x).view(b, length, self.n_heads, 2, self.head_dim)
        k = self.k_proj(x).view(b, length, self.n_heads, 2, self.head_dim)
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)
        v = self.v_proj(x).view(b, length, self.n_heads, 2 * self.head_dim)
        v = v.transpose(1, 2)

        weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(weights, dim=-1)
        a1, a2 = weights.unbind(dim=2)
        lam = self._lambda()
        y = torch.matmul(a1 - lam * a2, v)

        # Stabilization used by released Differential Transformer attention.
        y = self.subln(y) * (1.0 - self.lambda_init)
        out = self.o_proj(y.transpose(1, 2).reshape(b, length, -1))

        if debug:
            parts = {"q1": q[:, :, 0], "k1": k[:, :, 0],
                     "q2": q[:, :, 1], "k2": k[:, :, 1], "v": v,
                     "A1": a1, "A2": a2, "lambda": lam}
            return out, parts
        return out
