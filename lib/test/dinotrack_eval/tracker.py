"""DINOTrack single-object tracker — online inference (§4.1).

Verbatim port of dinotrack_v1/test/tracker.py with the framework-cfg
adaptation (cfg.TEST.* / cfg.DATA.* instead of the v1 lowercase schema) and
the zoom path dropped (not transplanted; lives on in dinotrack_v1).

§4.1 "During the inference phase, DINOTrack runs online without any test-time
adaptation" -> the template crop is extracted ONCE from the first-frame GT and
its features (F_z + per-layer class tokens needed by CGHM) are cached; every
subsequent frame only forwards the search crop.

[UNSPECIFIED] Hann window: paper describes none; default ON to match the
OSTrack tracker implementation (user decision 2026-09-19) — multiplicative
window on the score map, as in OSTrack lib/test/tracker/ostrack.py:87-89.
"""

from __future__ import annotations

import numpy as np
import torch

from lib.test.dinotrack_eval.processing import _to_tensor_normalize, sample_target
from lib.utils.dino.box_ops import clip_box
from lib.utils.dino.hann import hann2d


class DINOTrackTracker:
    def __init__(self, model: torch.nn.Module, cfg, device: str = "cuda"):
        self.model = model.eval()
        self.cfg = cfg
        self.device = device
        # inference crop geometry comes from TEST.* (settable independently of
        # the training crops)
        self.template_sz = cfg.TEST.TEMPLATE_SIZE
        self.search_sz = cfg.TEST.SEARCH_SIZE
        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.mean, self.std = cfg.DATA.MEAN, cfg.DATA.STD
        self.clip_margin = getattr(cfg.TEST, "CLIP_MARGIN", 10.0)
        self._hann = None
        self.hann_enabled = bool(getattr(cfg.TEST, "HANN", True))

    def _hann_window(self, n: int, device) -> torch.Tensor:
        if self._hann is None or self._hann.shape[-1] != n:
            self._hann = hann2d(n, centered=True).to(device)   # OSTrack convention
        return self._hann

    @torch.no_grad()
    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        x = _to_tensor_normalize(crop, self.mean, self.std)   # (3, S, S)
        return x.unsqueeze(0).to(self.device)

    def initialize(self, image: np.ndarray, init_bbox) -> None:
        """image: RGB uint8 (H,W,3); init_bbox: [x, y, w, h]."""
        crop, _rf = sample_target(image, list(map(float, init_bbox)),
                                  self.template_factor, self.template_sz)
        t = self._preprocess(crop)
        self.z_cache = self.model.encode_template(t)           # (F_z, cls_z) cached
        # UAV benchmarks may mark an invisible first frame with a degenerate
        # box (all-zero rows, or ARD100's -100 sentinel): floor the state to
        # a 1-px non-negative box so the loop cannot be poisoned at init.
        bb = [float(v) for v in init_bbox]
        self.state = [max(bb[0], 0.0), max(bb[1], 0.0), max(bb[2], 1.0), max(bb[3], 1.0)]

    @torch.no_grad()
    def track(self, image: np.ndarray) -> dict:
        cx = self.state[0] + self.state[2] / 2.0
        cy = self.state[1] + self.state[3] / 2.0
        search_box = [cx - self.state[2] / 2.0, cy - self.state[3] / 2.0,
                      self.state[2], self.state[3]]            # prev box xywh
        search_box[2] = max(search_box[2], 1.0)                # never let a
        search_box[3] = max(search_box[3], 1.0)                # sub-pixel box crash the crop

        crop, rf, (x1, y1, side) = sample_target(image, search_box,
                                                 self.search_factor, self.search_sz,
                                                 return_crop_box=True)
        s = self._preprocess(crop)

        out = self.model.track(self.z_cache, s)
        score = out["score_map"]
        if self.hann_enabled:   # default ON, matches OSTrack's tracker
            score = score * self._hann_window(score.shape[-1], score.device)
        # re-decode with the (possibly hann-modulated) score map
        bbox = self.model.head.cal_bbox(score, out["size_map"], out["offset_map"])[0]

        # normalized cxcywh in crop -> original-frame pixels
        c_x = x1 + bbox[0].item() * side
        c_y = y1 + bbox[1].item() * side
        w = bbox[2].item() * side
        h = bbox[3].item() * side
        pred = [c_x - w / 2.0, c_y - h / 2.0, w, h]
        H, W = image.shape[:2]
        self.state = clip_box(pred, H, W, margin=self.clip_margin)
        return {"state": self.state, "score_map": score}
