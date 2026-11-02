"""Evaluate DINOTrack on a benchmark (§4.2 Tables 2–3) — OSTrack-framework port
of dinotrack_v1/test/evaluation/run.py. The loop, per-seq outputs, metrics
aggregation and submission export are verbatim; only config loading and the
checkpoint format changed (framework easydict cfg; ck["net"] state dicts).

Usage (from the framework root dinotrack/):
  PYTHONPATH=. python -m lib.test.dinotrack_eval.run \
      --config experiments/dinotrack/vitb16_full_8gpu.yaml \
      --ckpt output/checkpoints/train/dinotrack/vitb16_full_8gpu/DINOTrack_ep0245.pth.tar \
      --dataset lasot

Datasets: lasot, tnl2k, lasot_ext, trackingnet, got10k_val, got10k_test,
and the UAV benchmarks (uav_anti_uav, antiuav318_rgb, antiuav410, ard100 —
the latter four also get an Anti-UAV state-accuracy (SA) column and, where the
dataset ships attribute files, a per-attribute breakdown). got10k_test /
trackingnet have no public GT -> submission files are exported.
--seqs restricts a run to a shard of sequences; merge shards with merge_shards.py.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from lib.config.dinotrack import config as dino_config
from lib.models.dinotrack import build_dinotrack
from lib.test.dinotrack_eval import metrics
from lib.test.dinotrack_eval import datasets as ds_loaders
from lib.test.dinotrack_eval.image_io import load_image
from lib.test.dinotrack_eval.tracker import DINOTrackTracker

_FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # repo/framework root


def _default_data_root() -> str:
    """Standalone layout keeps data/ at the framework root; the original nested
    layout (repo/dinotrack/) put data/ one level up at the repo root."""
    parent = os.path.join(os.path.dirname(_FRAMEWORK_ROOT), "data")
    if not os.path.isdir(os.path.join(_FRAMEWORK_ROOT, "data")) and os.path.isdir(parent):
        return parent
    return os.path.join(_FRAMEWORK_ROOT, "data")

# Datasets scored with the extra Anti-UAV state-accuracy (SA) column; those
# with per-sequence attribute files additionally get an attribute breakdown.
UAV_SA_DATASETS = ("uav_anti_uav", "antiuav318_rgb", "antiuav410", "ard100")


def _default_ckpt(cfg, config_name: str) -> str:
    """Checkpoint layout produced by run_training.py --save_dir output."""
    return os.path.join(_FRAMEWORK_ROOT, "output", "checkpoints", "train",
                        "dinotrack", config_name,
                        f"DINOTrack_ep{cfg.TEST.EPOCH:04d}.pth.tar")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="experiment yaml under experiments/dinotrack/")
    ap.add_argument("--ckpt", default="",
                    help="framework checkpoint (.pth.tar, key 'net'); omit to "
                         "resolve from TEST.EPOCH in the standard layout")
    ap.add_argument("--dataset", required=True, choices=sorted(ds_loaders.LOADERS))
    ap.add_argument("--out", default="", help="results root (default: <ws>/output/test/<dataset>)")
    ap.add_argument("--seqs", default="",
                    help="optional file with one sequence name per line — "
                         "evaluate only those sequences (sharded evaluation)")
    ap.add_argument("--data-root", default=_default_data_root(),
                    help="dataset root holding the lasot/ got10k/ tnl2k/ ... symlinks")
    args = ap.parse_args()

    cfg = dino_config.cfg
    dino_config.update_config_from_file(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    ckpt = args.ckpt or _default_ckpt(cfg, config_name)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_dinotrack(cfg, training=False).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    state = ck.get("net", ck.get("model"))                # framework or v1 ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.startswith("backbone.")]
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    out_root = args.out or os.path.join(_FRAMEWORK_ROOT, "output", "test", args.dataset)
    boxes_dir = os.path.join(out_root, "per_seq")
    os.makedirs(boxes_dir, exist_ok=True)

    tracker = DINOTrackTracker(model, cfg, device=device)
    seqs = ds_loaders.LOADERS[args.dataset](args.data_root)
    if args.seqs:
        wanted = {l.strip() for l in open(args.seqs) if l.strip()}
        seqs = [s for s in seqs if s.name in wanted]
    print(f"[eval] {args.dataset}: {len(seqs)} sequences", flush=True)

    fps_count, fps_time = 0, 0.0

    def run_sequence(seq):
        nonlocal fps_count, fps_time
        first = load_image(seq.image_paths[0])
        if seq.gt is None:   # server-eval sets still need an init box
            raise RuntimeError(f"{args.dataset} test needs the first-frame GT for "
                               f"initialization but none is available for {seq.name}")
        tracker.initialize(first, seq.gt[0])
        preds = [seq.gt[0]]
        t0 = time.time()
        for ip in seq.image_paths[1:]:
            img = load_image(ip)
            preds.append(tracker.track(img)["state"])
        fps_time += time.time() - t0
        fps_count += len(seq.image_paths) - 1
        preds = np.asarray(preds, dtype=np.float64)
        np.savetxt(os.path.join(boxes_dir, seq.name + ".txt"), preds,
                   delimiter=",", fmt="%.2f")
        return preds

    per_seq = []
    attrs_by_seq = {}
    submit_rows = []
    skipped = []
    for si, seq in enumerate(seqs):
        if not seq.image_paths:
            print(f"[warn] no frames: {seq.name}")
            continue
        try:
            preds = run_sequence(seq)
        except Exception as e:
            skipped.append(seq.name)
            print(f"[warn] {seq.name}: {type(e).__name__}: {e} — sequence skipped",
                  flush=True)
            continue
        if args.dataset in ("got10k_test", "trackingnet"):
            for fi, b in enumerate(preds):
                submit_rows.append(f"{seq.name},{fi + 1 if args.dataset == 'got10k_test' else fi},"
                                   f"{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}")
        if seq.gt is not None:
            if args.dataset == "got10k_val":
                m = metrics.got10k_metrics(preds, seq.gt)
            elif len(seq.gt) == len(preds):
                m = metrics.compute_sequence_metrics(preds, seq.gt, seq.target_visible)
            else:
                m = (float("nan"),) * 3
            if args.dataset in UAV_SA_DATASETS and seq.target_visible is not None \
                    and len(seq.gt) == len(preds):
                sa = metrics.state_accuracy_scores(preds, seq.gt, seq.target_visible)
                per_seq.append((seq.name, *m, sa))
            else:
                per_seq.append((seq.name, *m))
            if getattr(seq, "attributes", None):
                attrs_by_seq[seq.name] = seq.attributes
        if (si + 1) % 20 == 0:
            print(f"[eval] {si + 1}/{len(seqs)} {seq.name} | {fps_count / max(fps_time, 1e-6):.1f} fps",
                  flush=True)

    if fps_count:
        print(f"[speed] {fps_count / fps_time:.1f} fps (backbone+neck+head, {device}; "
              f"Table 1 reports 61.4 fps on LaSOT)")
    if skipped:
        print(f"[eval] {args.dataset}: {len(skipped)}/{len(seqs)} sequences skipped "
              f"on per-sequence errors (see [warn] lines above)")

    if args.dataset in ("got10k_test", "trackingnet"):
        sub = os.path.join(out_root, f"submit_{args.dataset}.csv")
        with open(sub, "w") as f:
            f.write("\n".join(submit_rows) + "\n")
        print(f"[submit] wrote {sub} (upload to the official server, §4.2)")

    if per_seq:
        arr = np.array([[v for v in row[1:]] for row in per_seq], dtype=np.float64)
        if args.dataset == "got10k_val":
            ao, sr50, sr70 = np.nanmean(arr, axis=0)
            print(f"\n{args.dataset}: AO {ao:.2f}  SR0.5 {sr50:.2f}  SR0.7 {sr70:.2f} "
                  f"(paper GOT-10k test: 70.4 / 81.1 / 65.9)")
        elif args.dataset in UAV_SA_DATASETS and arr.shape[1] >= 4:
            auc, pn, pr, sa = np.nanmean(arr[:, :4], axis=0)
            print(f"\n{args.dataset}: AUC {auc:.2f}  P_norm {pn:.2f}  "
                  f"P {pr:.2f}  SA {sa:.2f}")
        else:
            auc, pn, pr = np.nanmean(arr, axis=0)
            print(f"\n{args.dataset}: AUC {auc:.2f}  P_norm {pn:.2f}  P {pr:.2f}")
        if attrs_by_seq:
            nmc = arr.shape[1]
            brk = metrics.attribute_breakdown(per_seq, attrs_by_seq, n_metric_cols=nmc)
            head = (f"[{args.dataset}] attribute breakdown "
                    f"(AUC / P_norm / P{' / SA' if nmc >= 4 else ''}):")
            lines = [head]
            for att, vals in sorted(brk.items()):
                cnt = vals[0]
                body = "  ".join(f"{vals[1 + i] * 100:.2f}" for i in range(nmc))
                lines.append(f"  {att:>26} n={cnt:4d}  {body}")
            print("\n" + "\n".join(lines))
            with open(os.path.join(out_root, "attribute_summary.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        with open(os.path.join(out_root, "per_seq_summary.txt"), "w") as f:
            for row in per_seq:
                f.write(",".join([row[0]] + [f"{v:.4f}" if v == v else "nan"
                                             for v in row[1:]]) + "\n")


if __name__ == "__main__":
    main()
