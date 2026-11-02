"""Package per-sequence prediction boxes into the official submission zips.

GOT-10k test / TrackingNet TEST have no public full GT: the paper (§4.2)
submits the test results to the official servers. The trackers' per-frame
boxes (x,y,w,h, one line per frame, first line = first-frame init box) are
written by lib.test.dinotrack_eval.run to <out>/per_seq/<seq>.txt (0-based
pixels from the tracker state; the first line echoes the published init box,
which for GOT-10k is already 1-based).

Formats (verified against the live submission pages, 2026-10-09):

  GOT-10k (per official exemplar SiamFC.zip: no tracker wrapper dir):
    GOT-10k_Test_000001/GOT-10k_Test_000001_001.txt
    GOT-10k_Test_000001/GOT-10k_Test_000001_time.txt   (one value per row)
    - box rows: "<x>,<y>,<w>,<h>", ALL coordinates 1-based: rows 2..N get
      +1 on x/y (tracker 0-based -> official 1-based); row 1 stays as the
      published (already 1-based) init box;
    - repetition files _00i: the live exemplar contains only _001 (R=1);
      --repetitions >1 replicates deterministically if ever required;

  TrackingNet: flat <video>.txt at zip root (validated accepted by the
  codabench re-hosted server — Coverage 100 on our prior submission).

Usage:
  PYTHONPATH=. python -m lib.test.dinotrack_eval.package_submission \
      --per-seq output/test/<tag>/got10k_test/per_seq \
      --dataset got10k --tracker DINOTrack \
      --out output/test/<tag>/got10k_test/submit_got10k_test.zip --fps 30 [--repetitions 3]
"""

from __future__ import annotations

import argparse
import os
import zipfile


def _write_got10k(z: zipfile.ZipFile, per_seq_dir: str, frame_time: float,
                  repetitions: int) -> int:
    """Exactly mirrors the official exemplar (SiamFC.zip, checked 2026-10-09):
    sequence dirs AT ZIP ROOT (no <tracker> wrapper), one
    <seq>/<seq>_001.txt per sequence (the published init box stays verbatim on
    row 1 - it is already 1-based; prediction rows shift 0-based -> 1-based by
    +1 on x/y), %.4f comma format, and <seq>_time.txt with ONE per-frame
    runtime value per row. repetitions>1 additionally writes _002/.. (the
    page diagram shows 3; the live exemplar ships only _001 - default 1)."""
    names = sorted(f for f in os.listdir(per_seq_dir) if f.endswith(".txt"))
    assert names, f"no .txt files under {per_seq_dir}"
    for name in names:
        seq = name[:-4]
        rows = [r for r in open(os.path.join(per_seq_dir, name)).read().splitlines() if r.strip()]
        out_rows = []
        for i, row in enumerate(rows):
            x, y, w, h = (float(v) for v in row.split(","))
            if i > 0:                                       # 0-based -> 1-based (row 1 = published init)
                x, y = x + 1.0, y + 1.0
            out_rows.append(f"{x:.4f},{y:.4f},{w:.4f},{h:.4f}")
        body = "\n".join(out_rows) + "\n"
        for r in range(1, max(1, repetitions) + 1):
            z.writestr(f"{seq}/{seq}_{r:03d}.txt", body)
        z.writestr(f"{seq}/{seq}_time.txt",
                   "\n".join([f"{frame_time:.6g}"] * len(out_rows)) + "\n")
    return len(names)


def _write_trackingnet(z: zipfile.ZipFile, per_seq_dir: str) -> int:
    names = sorted(f for f in os.listdir(per_seq_dir) if f.endswith(".txt"))
    assert len(names) == 511, f"TrackingNet TEST needs 511 sequences, got {len(names)}"
    for name in names:
        z.write(os.path.join(per_seq_dir, name), arcname=name)
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-seq", required=True, help="dir of <seq>.txt box files")
    ap.add_argument("--dataset", required=True, choices=("got10k", "trackingnet"))
    ap.add_argument("--out", required=True, help="output .zip path")
    ap.add_argument("--tracker", default="DINOTrack", help="GOT-10k top-level tracker dir name")
    ap.add_argument("--repetitions", type=int, default=1,
                    help="GOT-10k repetition files per sequence (exemplar ships 1; "
                         "page diagram shows 3 - only raise if the server asks)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="measured tracking fps; used for the GOT-10k N x R time matrix")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame_time = 1.0 / args.fps if args.fps > 0 else 0.016667
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        n = (_write_got10k(z, args.per_seq, frame_time, args.repetitions)
             if args.dataset == "got10k" else _write_trackingnet(z, args.per_seq))
    print(f"[submit] {n} sequences (dataset={args.dataset}, reps={args.repetitions if args.dataset == 'got10k' else '-'}) -> {args.out}")


if __name__ == "__main__":
    main()
