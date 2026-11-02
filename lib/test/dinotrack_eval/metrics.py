"""Benchmark metrics — LaSOT protocol AUC / P_norm / Precision (§4.2), GOT-10k
AO / SR0.5 / SR0.7, TrackingNet P / P_norm.

Standard implementations (overlap thresholds 0:0.05:1; normalized precision
at threshold 0.20). These are metric computation only (§4.2); no
test-time adaptation is applied — paper reports these numbers directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_sequence_metrics(
    pred_xywh: np.ndarray,
    gt_xywh: np.ndarray,
    target_visible: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Return LaSOT (AUC, normalized precision, precision@20).

    This mirrors pytracking/OSTrack's LaSOT protocol. Occluded/out-of-view
    frames remain in the denominator and count as localization failures.
    """
    pred = np.asarray(pred_xywh, dtype=np.float64)
    gt = np.asarray(gt_xywh, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError(f"expected matching (N,4) boxes, got {pred.shape} and {gt.shape}")
    if len(gt) == 0:
        return float("nan"), float("nan"), float("nan")

    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    if target_visible is not None:
        visible = np.asarray(target_visible, dtype=bool).reshape(-1)
        if len(visible) != len(gt):
            raise ValueError("target_visible and boxes must have equal length")
        valid &= visible

    pred_center = pred[:, :2] + 0.5 * (pred[:, 2:] - 1.0)
    gt_center = gt[:, :2] + 0.5 * (gt[:, 2:] - 1.0)
    center_delta = pred_center - gt_center
    center_err = np.linalg.norm(center_delta, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        center_err_norm = np.linalg.norm(center_delta / gt[:, 2:], axis=1)

    tl = np.maximum(pred[:, :2], gt[:, :2])
    br = np.minimum(pred[:, :2] + pred[:, 2:] - 1.0,
                    gt[:, :2] + gt[:, 2:] - 1.0)
    intersection = np.prod(np.maximum(br - tl + 1.0, 0.0), axis=1)
    union = np.prod(pred[:, 2:], axis=1) + np.prod(gt[:, 2:], axis=1) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = intersection / union

    # Invalid/invisible frames are failures, not omitted from LaSOT curves.
    iou[~valid] = -1.0
    center_err[~valid] = np.inf
    center_err_norm[~valid] = np.inf
    success = np.array([(iou > t).mean() for t in np.arange(0.0, 1.01, 0.05)])
    auc = float(success.mean())
    # pytracking/OSTrack reports index 20 of the normalized precision curve,
    # i.e. the fraction with per-axis normalized center error <= 0.20.
    p_norm = float((center_err_norm <= 0.20).mean())
    prec20 = float((center_err <= 20.0).mean())
    return auc, p_norm, prec20


def got10k_metrics(pred_xywh: np.ndarray, gt_xywh: np.ndarray) -> Tuple[float, float, float]:
    """AO (mean IoU), SR@0.5, SR@0.7 — official GOT-10k definitions on all
    valid frames (w>0/h>0 kept, one-shot already guaranteed by dataset split)."""
    valid = (gt_xywh[:, 2] > 0) & (gt_xywh[:, 3] > 0)
    gt, pred = gt_xywh[valid], pred_xywh[valid]
    if len(gt) == 0:
        return float("nan"), float("nan"), float("nan")
    w_i = np.maximum(0.0, np.minimum(pred[:, 0] + pred[:, 2], gt[:, 0] + gt[:, 2])
                     - np.maximum(pred[:, 0], gt[:, 0]))
    h_i = np.maximum(0.0, np.minimum(pred[:, 1] + pred[:, 3], gt[:, 1] + gt[:, 3])
                     - np.maximum(pred[:, 1], gt[:, 1]))
    inter = w_i * h_i
    union = (gt[:, 2] + 1) * (gt[:, 3] + 1) + (pred[:, 2] + 1) * (pred[:, 3] + 1) - inter
    iou = inter / union
    return float(iou.mean()), float((iou > 0.5).mean()), float((iou > 0.7).mean())


def _plain_iou(p: np.ndarray, g: np.ndarray) -> float:
    """Rectangle IoU WITHOUT the inclusive-pixel (+1) convention — matches
    FocusTrack analys_antiuav.ExperimentAntiUAV410.iou() exactly."""
    w_i = max(min(p[0] + p[2], g[0] + g[2]) - max(p[0], g[0]), 0)
    h_i = max(min(p[1] + p[3], g[1] + g[3]) - max(p[1], g[1]), 0)
    inter = w_i * h_i
    union = p[2] * p[3] + g[2] * g[3] - inter
    return float(inter / union) if union > 0 else 0.0


def state_accuracy_scores(pred_xywh: np.ndarray, gt_xywh: np.ndarray,
                          exist: np.ndarray) -> float:
    """Anti-UAV state-accuracy (SA) — FocusTrack analys_antiuav.eval() rules:
    - invisible frame (exist==0): score = 1 iff the prediction is an "absent"
      marker (len<4), else 0. Real trackers always emit a box -> 0 (pessimistic
      convention shared by published Anti-UAV baselines);
    - visible frame with degenerate GT (len<4 or sum==0): frame skipped;
    - visible frame: plain IoU.
    SA = mean over the retained per-frame scores.
    """
    pred = np.asarray(pred_xywh, dtype=np.float64)
    gt = np.asarray(gt_xywh, dtype=np.float64)
    ex = np.asarray(exist).reshape(-1)
    scores = []
    for _pred, _gt, _ex in zip(pred, gt, ex[:min(len(pred), len(gt), len(ex))]):
        if int(_ex) == 0:
            scores.append(0.0 if len(_pred) == 4 else 1.0)
            continue
        if len(_gt) < 4 or np.sum(_gt) == 0:
            continue
        scores.append(_plain_iou(_pred, _gt) if len(_pred) == 4 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def attribute_breakdown(per_seq_rows, attrs_by_seq: dict,
                        n_metric_cols: int = 4):
    """Per-attribute mean of per-sequence metrics (= mean of the 21-point
    curves averaged per sequence, linear identity; matches FocusTrack's
    att-curve averaging). per_seq_rows: (name, m1..mn); attrs_by_seq:
    name -> {attr: bool}. Returns {attr: (n_seqs, mean m1..mn)}."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in per_seq_rows:
        a = attrs_by_seq.get(row[0]) or {}
        for att, on in a.items():
            if on:
                buckets[att].append([float(v) if v == v else float("nan")
                                     for v in row[1:n_metric_cols + 1]])
    out = {}
    for att, rows in buckets.items():
        arr = np.array(rows, dtype=np.float64)
        out[att] = (len(rows), *np.nanmean(arr, axis=0).tolist())
    return out
