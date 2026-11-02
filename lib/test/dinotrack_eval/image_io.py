"""Image loading helpers (cv2, BGR->RGB). OSTrack uses jpeg4py optionally;
we rely on cv2 only (present via opencv-python) — [UNSPECIFIED] loader choice."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Read image as RGB uint8. Raises on failure (sampler resamples)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
