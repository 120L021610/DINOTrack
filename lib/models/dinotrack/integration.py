"""Multi-level Integration module incorporated with CGHM — §3.2–3.3, Eq. 1–3, Fig. 1b.

§3.2 Eq. 3:
    F_z = Concat_{l in S} ( c^l, Flat(phi(f_z^l)) )
    F_x = Concat_{l in S} ( Flat(x~^l) )
where "Concat(.) is executed along the channel dimension, and Flat(.) reshapes
the 2D spatial feature maps into a 1D token sequence".

Shapes (DINOv3-Base, K=6, D=128, §4.1):
  F_z: (B, 65, 6*128) = (B, 65, 768)   (one CLS token per layer prepended;
       64 if use_cls_token=False, Table 5 #1)
  F_x: (B, 256, 768)                   search, CGHM-modulated per Eq. 4–5

Note: F_x depends on the template class tokens c^l (through CGHM), so at
inference the cached template features must include cls_z, not only F_z.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from lib.models.dinotrack.dino_config import NeckCfg
from lib.models.dinotrack.cghm import CGHM
from lib.models.dinotrack.projection import Projection


class MultiLevelIntegration(nn.Module):
    def __init__(self, cfg: NeckCfg, backbone_c: int, patch_size: int):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.num_layers
        self.D = cfg.proj_dim
        self.use_cls_token = cfg.use_cls_token   # Table 5 ablation column 1
        self.use_cghm = cfg.use_cghm             # Table 5 ablation column 2
        self.proj = Projection(c_in=backbone_c, d_out=cfg.proj_dim,
                               num_layers=cfg.num_layers, per_layer=cfg.per_layer_proj)
        self.cghm = CGHM(tau=cfg.tau, similarity=cfg.cghm_similarity)

    # --------------------------------------------------------------- template side
    def forward_template(self, feat_z: List[torch.Tensor], cls_z: List[torch.Tensor],
                         ) -> torch.Tensor:
        """feat_z: K x (B, C, 8, 8); cls_z: K x (B, C) -> F_z (B, Lz, K*D)."""
        z_tokens = []
        for l in range(self.K):
            c_l = self.proj.forward_cls(cls_z[l], l)              # (B, D)   Eq. 1
            xz_l = self.proj.forward_map(feat_z[l], l)            # (B, D, 8, 8) Eq. 1
            tz = xz_l.flatten(2).transpose(1, 2)                  # (B, 64, D) Flat
            if self.use_cls_token:
                # Eq. 3: c^l prepended to the token sequence of layer l
                tz = torch.cat([c_l.unsqueeze(1), tz], dim=1)     # (B, 65, D)
            z_tokens.append(tz)
        return torch.cat(z_tokens, dim=-1)                        # (B, Lz, K*D=768)

    # ----------------------------------------------------------------- search side
    def forward_search(self, feat_x: List[torch.Tensor], cls_z: List[torch.Tensor],
                       return_heatmaps: bool = False):
        """feat_x: K x (B, C, 16, 16); cls_z: K x (B, C) template class tokens ->
        F_x (B, 256, K*D). Optionally also the per-layer heatmaps M^l (B,1,16,16)."""
        x_tokens, heatmaps = [], []
        for l in range(self.K):
            c_l = self.proj.forward_cls(cls_z[l], l)              # (B, D) Eq. 1
            xx_l = self.proj.forward_map(feat_x[l], l)            # (B, D, 16, 16) Eq. 1
            if self.use_cghm:
                xx_l, M_l = self.cghm(xx_l, c_l)                  # Eq. 2/4/5
                heatmaps.append(M_l)
            x_tokens.append(xx_l.flatten(2).transpose(1, 2))      # (B, 256, D)
        F_x = torch.cat(x_tokens, dim=-1)                         # (B, 256, K*D=768)
        if return_heatmaps:
            return F_x, heatmaps
        return F_x

    # ------------------------------------------------------------------ full pair
    def forward(self, feat_z: List[torch.Tensor], cls_z: List[torch.Tensor],
                feat_x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.forward_template(feat_z, cls_z),
                self.forward_search(feat_x, cls_z))
