"""DINOTrack — full model, §3.1 / Fig. 1(a), implemented on the (renamed)
OSTrack framework.

    Z, X -> frozen weight-shared DINOv3 -> Multi-level Integration w/ CGHM
         -> (Diff Block)*1 -> center head -> box B_t

§4.1: backbone fully frozen; only neck + head are optimized. Template is
fixed (first-frame) at inference: "DINOTrack runs online without any
test-time adaptation" (§4.1) — hence F_z/cls_z are computed once and cached.

This file replaces the stock OSTrack wrapper (former lib/models/ostrack/
ostrack.py). The model internals are a verbatim port of
dinotrack_v1/models/dinotrack.py; the framework-contract adaptations are:
  * forward() takes the kwargs the OSTrack actor passes (template, search,
    ce_template_mask, ce_keep_rate, return_last_attn) — the CE / attention
    args are ignored (DINOTrack has no candidate elimination);
  * pred_boxes is returned as (B, 1, 4) to match the OSTrack actor's
    num_queries convention (the evaluator's tracker uses track(), which
    keeps the v1 (B, 4) contract);
  * `box_head` is exposed as a property (== self.head) and CenterHead
    carries `.feat_sz` / `.cal_bbox`, keeping the framework test tracker
    and `lib/test/parameter` checkpoint path conventions working;
  * the builder reads the framework easydict cfg via dino_config.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from lib.models.dinotrack.backbone import DINOv3FrozenBackbone
from lib.models.dinotrack.center_head import CenterHead
from lib.models.dinotrack.diff_block import DiffBlock
from lib.models.dinotrack.dino_config import model_cfgs_from_ostrack
from lib.models.dinotrack.integration import MultiLevelIntegration

TemplateCache = Tuple[torch.Tensor, List[torch.Tensor]]


class DINOTrack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        b, n, d, h = model_cfgs_from_ostrack(cfg)
        # §3.1 "two weight-shared and parameter-frozen DINOv3 backbone networks"
        # = one network applied to each branch
        self.backbone = DINOv3FrozenBackbone(b, num_layers_to_take=n.num_layers)
        self.c_total = n.num_layers * n.proj_dim          # §4.1 C = 6 x 128 = 768
        self.integration = MultiLevelIntegration(n, b.embed_dim, b.patch_size)
        self.diff_blocks = nn.ModuleList([DiffBlock(d, dim=self.c_total)
                                          for _ in range(d.n_blocks)])  # §3.4 single block
        self.feat_sz_s = cfg.DATA.SEARCH.SIZE // b.patch_size   # 16 (§3.5 Hx/P)
        self.feat_len_s = self.feat_sz_s ** 2             # 256 search tokens
        self.head = CenterHead(self.c_total, h.channels, self.feat_sz_s)

    # ------------------------------------------------- framework contract
    @property
    def box_head(self) -> CenterHead:
        """OSTrack-convention alias: the framework test tracker calls
        net.box_head.cal_bbox(...) and reads net.box_head.feat_sz."""
        return self.head

    # ------------------------------------------------------------------ helpers
    def _interaction(self, F_z: torch.Tensor, F_x: torch.Tensor) -> torch.Tensor:
        """§3.4: F_in = [F_z ; F_x] through the Diff Block(s) -> search-part."""
        x = torch.cat([F_z, F_x], dim=1)                  # (B, 65+256, 768)
        for blk in self.diff_blocks:
            x = blk(x)
        return x[:, -self.feat_len_s:]                    # (B, 256, 768) refined search

    def _decode(self, enc_opt: torch.Tensor) -> Dict[str, torch.Tensor]:
        # (B, 256, 768) -> (B, 768, 16, 16) feature map for the conv head
        opt_feat = enc_opt.transpose(1, 2).reshape(-1, self.c_total,
                                                   self.feat_sz_s, self.feat_sz_s)
        score, bbox, size, offset = self.head(opt_feat)   # §3.5
        return {"score_map": score, "pred_boxes": bbox, "size_map": size,
                "offset_map": offset}

    # ------------------------------------------------------------------ forward
    def forward(self, template: torch.Tensor, search: torch.Tensor,
                ce_template_mask=None, ce_keep_rate=None,
                return_last_attn: bool = False) -> Dict[str, torch.Tensor]:
        """template: (B,3,128,128); search: (B,3,256,256) (§4.1), already
        normalized (ImageNet mean/std applied by the data pipeline).
        Returns score_map, pred_boxes (B,1,4) cxcywh crop-normalized,
        size_map, offset_map. The ce_* / return_last_attn kwargs are accepted
        for OSTrack-actor signature compatibility and ignored."""
        fz, cz = self.backbone(template)                  # K x (B,768,8,8), K x (B,768)
        fx, _ = self.backbone(search)                     # K x (B,768,16,16)
        F_z, F_x = self.integration(fz, cz, fx)           # (B,65,768), (B,256,768)
        out = self._decode(self._interaction(F_z, F_x))
        out["pred_boxes"] = out["pred_boxes"].unsqueeze(1)  # (B,4) -> (B,1,4)
        return out

    def forward_with_heatmaps(self, template: torch.Tensor, search: torch.Tensor):
        """Same as forward() (without the pred_boxes unsqueeze) but also
        returns the CGHM maps M^l (Eq. 4) for the Fig. 5 style visualization."""
        fz, cz = self.backbone(template)
        fx, _ = self.backbone(search)
        F_z = self.integration.forward_template(fz, cz)
        F_x, heatmaps = self.integration.forward_search(fx, cz, return_heatmaps=True)
        out = self._decode(self._interaction(F_z, F_x))
        return out, heatmaps

    # ---------------------------------------------------------------- inference
    def encode_template(self, template: torch.Tensor) -> TemplateCache:
        """First frame: returns (F_z, cls_z) to cache (§4.1 fixed template)."""
        with torch.no_grad():
            fz, cz = self.backbone(template)              # (B,3,128,128) -> features
            return self.integration.forward_template(fz, cz), cz

    @torch.no_grad()
    def track(self, template_cache: TemplateCache, search: torch.Tensor,
              ) -> Dict[str, torch.Tensor]:
        """One subsequent frame given the cached template (v1 (B,4) contract)."""
        F_z, cz = template_cache
        fx, _ = self.backbone(search)                     # (B,3,256,256)
        F_x = self.integration.forward_search(fx, cz)     # (B, 256, 768) w/ CGHM
        return self._decode(self._interaction(F_z, F_x))


def build_dinotrack(cfg, training: bool = True) -> DINOTrack:
    net = DINOTrack(cfg)
    if training:
        frozen = sum(p.numel() for p in net.backbone.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[dinotrack] backbone params (frozen): {frozen / 1e6:.2f}M | "
              f"trainable neck+head params: {trainable / 1e6:.2f}M")
    return net
