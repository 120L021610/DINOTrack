"""Frozen DINOv3 backbone wrapper (§3.1–3.2, §4.1).

§3.1 "These two inputs [template Z, search X] are processed by two weight-
shared and parameter-frozen DINOv3 backbone networks" — one network, applied
to each branch.
§4.1 "we freeze the parameters of the backbone network throughout the
training phase" and "only optimize the parameters of the neck network and
the predicted head".

The local checkpoint (models/dinov3-vitb16-pretrain-lvd1689m) is in
HuggingFace transformers format, so intermediate layers are taken via
`output_hidden_states=True` and each selected layer output is passed
through the model's final LayerNorm. This reproduces the semantics of the
Meta repo's `get_intermediate_layers(..., norm=True, return_class_token=True)`
(verified numerically: norm(hidden_states[-1]) == last_hidden_state).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.dinotrack.dino_config import BackboneCfg


class DINOv3FrozenBackbone(nn.Module):
    """§3.2 "extract feature representations from the last K layers": per layer
    returns the spatial map f^l (B, C, H/P, W/P) and the class token (B, C)."""

    def __init__(self, cfg: BackboneCfg, num_layers_to_take: int = 6):
        super().__init__()
        from transformers import AutoModel  # local import: heavy dep kept optional

        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.weights, output_hidden_states=True,
                                               local_files_only=True)
        # §4.1 fully frozen; never receives gradients even in train() mode
        self.model.eval()
        self.model.requires_grad_(False)
        self.num_prefix_tokens = 1 + cfg.register_tokens  # [CLS, 4 x register]
        K = num_layers_to_take
        # hidden_states[0] = embeddings output, hidden_states[l] = block l (1-based)
        self.layer_indices = list(range(cfg.num_layers - K + 1, cfg.num_layers + 1))
        assert len(self.layer_indices) == K

    def train(self, mode: bool = True):  # noqa: D102 - backbone stays in eval
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """img: (B, 3, H, W), H,W multiples of 16 (128² / 256², §4.1).
        Returns (patch_maps, cls_tokens), each a list of K tensors:
          patch_maps[l]: (B, C, H/16, W/16) spatial feature f^l   (§3.2 f_z^l/f_x^l)
          cls_tokens[l]: (B, C)             class token t_cls^l   (§3.2)
        """
        assert img.shape[-1] % self.cfg.patch_size == 0
        assert img.shape[-2] % self.cfg.patch_size == 0
        out = self.model(pixel_values=img)
        h, w = img.shape[-2] // self.cfg.patch_size, img.shape[-1] // self.cfg.patch_size
        patch_maps, cls_tokens = [], []
        for idx in self.layer_indices:
            hs = out.hidden_states[idx]                 # (B, 1+4+hw, C) pre-norm
            hs = self.model.norm(hs)                    # final LayerNorm (§3.3 plan decision)
            cls_tokens.append(hs[:, 0])                 # (B, C) — t_cls^l
            patch = hs[:, self.num_prefix_tokens:]      # (B, hw, C); register tokens dropped
            patch_maps.append(patch.transpose(1, 2).reshape(-1, self.cfg.embed_dim, h, w))
        return patch_maps, cls_tokens
