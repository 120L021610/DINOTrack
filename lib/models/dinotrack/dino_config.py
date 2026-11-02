"""DINOTrack model dataclasses — verbatim from dinotrack_v1/dinotrack/config.py
(only the model-related dataclasses; documentation comments preserved), plus a
single adapter that builds them from the framework's easydict cfg so the
OSTrack-side builder (lib/models/dinotrack/dinotrack.py) stays config-driven.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DINO_CONFIG_FILE = Path(__file__).resolve()
# lib/models/dinotrack/dino_config.py -> framework root (repo root when
# standalone; the original nested layout put models/ one level up)
_FRAMEWORK_ROOT = _DINO_CONFIG_FILE.parents[3]


@dataclass
class BackboneCfg:
    # §4.1 "pre-trained DINOv3-Base model" (ViT-B/16, LVD-1689M weights)
    weights: str = "models/dinov3-vitb16-pretrain-lvd1689m"
    num_layers: int = 12          # from checkpoint config.json: num_hidden_layers
    embed_dim: int = 768          # config.json hidden_size; "C" initial embedding dim (§3.2)
    patch_size: int = 16          # §3.2 "patches of size P x P, where P = 16"
    register_tokens: int = 4      # config.json num_register_tokens. [UNSPECIFIED] paper only
                                  # mentions "class token"; registers are dropped from the
                                  # per-layer features (they are neither t_cls nor f_x).


@dataclass
class NeckCfg:
    # crop sizes moved to data.search.size / data.template.size (§4.1 128x128 / 256x256)
    num_layers: int = 6           # §4.1 "aggregates the features of the last 6 layers" (K)
    proj_dim: int = 128           # §4.1 "projects them onto a channel dimension D = 128"
    # §3.2 "a shared projection layer phi(.) (implemented as a 1x1 convolution)".
    # [UNSPECIFIED] whether shared across the K layers; literal reading = one phi for all
    # layers and both branches. per_layer_proj=True is provided as an ablation.
    per_layer_proj: bool = False
    use_cls_token: bool = True    # §3.2 Eq. 3 (c^l prepended to F_z). False = Table 5 #1
    use_cghm: bool = True         # §3.3. False = Table 5 #1/#2/#4
    tau: float = 0.07             # §3.3 "we empirically set it to 0.07"
    # "dot" is the literal Eq. 4 implementation and the formal recipe.
    # "cosine" remains loadable only for the rejected historical experiment;
    # the paper does not state L2 normalization and it reduced LaSOT accuracy.
    cghm_similarity: str = "dot"


@dataclass
class DiffBlockCfg:
    attention: str = "diff"       # "diff" = §3.4 Eq. 6; "std" = Table 5 #1 baseline
    n_blocks: int = 1             # §3.4 "we use a single Diff Block"
    n_heads: int = 6              # §4.1 "6 attention heads"
    head_dim: int = 64            # §4.1 "the dimension of the attention heads is set to 64"
    lambda_init: float = 0.2      # §3.4 Eq. 8 with d=0 -> 0.2 (Table 6 sweep {0,0.2,0.5,0.8})
    # §3.4 Eq. 7 "initial values follow a normal distribution".
    # Differential Transformer [42] released initialization.
    lambda_std: float = 0.1
    mlp_ratio: int = 4            # [UNSPECIFIED] Fig. 3 shows "MLP" only; standard 4x
    drop_path: float = 0.0        # [UNSPECIFIED] Fig. 3 shows DropPath, no rate
    norm_eps: float = 1e-5        # [UNSPECIFIED] LayerNorm/RMSNorm eps -> PyTorch default
    # Legacy config key retained so old YAML files still parse. The released
    # implementation uses post-attention RMSNorm and (1-lambda_init) scaling.
    renorm_diff_attn: bool = False


@dataclass
class HeadCfg:
    channels: int = 256   # [UNSPECIFIED] "lightweight fully convolutional prediction
                          # head" (§3.1); OSTrack HEAD.NUM_CHANNELS = 256 as reference


def _resolve_weights(path: str) -> str:
    """MODEL.BACKBONE.WEIGHTS is repo-root-relative ('models/dinov3-...'); training
    runs with cwd = the framework root (dinotrack/), so resolve explicitly."""
    if os.path.isabs(path) or os.path.isdir(path):
        return path
    for root in (_FRAMEWORK_ROOT, _FRAMEWORK_ROOT.parent):
        candidate = root / path
        if candidate.is_dir():
            return str(candidate)
    return path


def model_cfgs_from_ostrack(cfg) -> tuple:
    """Build the four v1 dataclasses from the framework easydict cfg."""
    b = BackboneCfg(
        weights=_resolve_weights(cfg.MODEL.BACKBONE.WEIGHTS),
        num_layers=cfg.MODEL.BACKBONE.NUM_LAYERS,
        embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,
        patch_size=cfg.MODEL.BACKBONE.PATCH_SIZE,
        register_tokens=cfg.MODEL.BACKBONE.REGISTER_TOKENS,
    )
    n = NeckCfg(
        num_layers=cfg.MODEL.NECK.NUM_LAYERS,
        proj_dim=cfg.MODEL.NECK.PROJ_DIM,
        per_layer_proj=cfg.MODEL.NECK.PER_LAYER_PROJ,
        use_cls_token=cfg.MODEL.NECK.USE_CLS_TOKEN,
        use_cghm=cfg.MODEL.NECK.USE_CGHM,
        tau=cfg.MODEL.NECK.TAU,
        cghm_similarity=cfg.MODEL.NECK.CGHM_SIMILARITY,
    )
    d = DiffBlockCfg(
        attention=cfg.MODEL.DIFF.ATTENTION,
        n_blocks=cfg.MODEL.DIFF.N_BLOCKS,
        n_heads=cfg.MODEL.DIFF.N_HEADS,
        head_dim=cfg.MODEL.DIFF.HEAD_DIM,
        lambda_init=cfg.MODEL.DIFF.LAMBDA_INIT,
        lambda_std=cfg.MODEL.DIFF.LAMBDA_STD,
        mlp_ratio=cfg.MODEL.DIFF.MLP_RATIO,
        drop_path=cfg.MODEL.DIFF.DROP_PATH,
        norm_eps=cfg.MODEL.DIFF.NORM_EPS,
    )
    h = HeadCfg(channels=cfg.MODEL.HEAD.NUM_CHANNELS)
    return b, n, d, h
