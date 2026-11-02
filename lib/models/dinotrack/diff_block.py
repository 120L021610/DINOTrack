"""Diff Block (Differential Denoising Interaction, DDI) — §3.4, Fig. 3.

Fig. 3 order: Input -> LayerNorm -> attention -> per-head RMSNorm -> OutProj ->
DropPath -> residual -> LayerNorm -> MLP -> DropPath -> residual. The per-head
RMSNorm belongs to differential attention itself, as in its reference
implementation; the standard-attention ablation does not apply it.

§3.4 "we replace the standard Transformer module with our proposed Diff
Block"; single block ("we use a single Diff Block").

[UNSPECIFIED] MLP activation (GELU assumed — differential-transformer [42]
convention) and MLP expansion ratio (4x), DropPath rate (0.0), norms' eps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.models.dinotrack.dino_config import DiffBlockCfg
from lib.models.dinotrack.diff_attention import DifferentialAttention, StandardAttention
from lib.models.dinotrack.layers import DropPath


class DiffBlock(nn.Module):
    def __init__(self, cfg: DiffBlockCfg, dim: int = 768):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, eps=cfg.norm_eps)   # Fig. 3 "LayerNorm"
        if cfg.attention == "diff":
            self.attn = DifferentialAttention(cfg, dim_in=dim)  # Eq. 6-7
        elif cfg.attention == "std":
            self.attn = StandardAttention(cfg, dim_in=dim)      # Table 5 baseline
        else:
            raise ValueError(f"unknown attention type: {cfg.attention!r}")
        self.drop_path1 = DropPath(cfg.drop_path)              # Fig. 3 DropPath
        self.norm_mlp = nn.LayerNorm(dim, eps=cfg.norm_eps)
        hidden = int(dim * cfg.mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path2 = DropPath(cfg.drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 321, 768) -> (B, 321, 768); pre-norm + residual (Fig. 3)
        x = x + self.drop_path1(self.attn(self.norm_attn(x)))
        x = x + self.drop_path2(self.mlp(self.norm_mlp(x)))
        return x
