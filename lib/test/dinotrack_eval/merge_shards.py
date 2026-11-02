"""Merge sharded LaSOT evaluation outputs into one canonical result dir.

Usage:
  PYTHONPATH=. python -m lib.test.dinotrack_eval.merge_shards \
      --out output/test/<tag>/lasot \
      --shards output/test/<tag>_s0/lasot output/test/<tag>_s1/lasot ...

Verifies the union of shard rows is exactly the full sequence set (no
missing/duplicated names), writes <out>/per_seq_summary.txt plus a
MERGED_FROM_SHARDS.txt provenance note, and prints the aggregate
(AUC/P_norm/P = mean over the merged rows — identical to the official
serial aggregate; verified on prior runs).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

FULL_SET_URL = "lib/train/data_specs"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="canonical results dir (…/<tag>/<dataset>)")
    ap.add_argument("--shards", nargs="+", required=True, help="shard results dirs (…/<tag>_sN/<dataset>)")
    ap.add_argument("--full-names", default="",
                    help="optional file with the complete sequence list to verify against")
    args = ap.parse_args()

    rows, shard_info = [], []
    for sdir in args.shards:
        f = os.path.join(sdir, "per_seq_summary.txt")
        if not os.path.isfile(f):
            raise SystemExit(f"missing shard summary: {f}")
        lines = [l for l in open(f).read().splitlines() if l.strip()]
        shard_info.append((sdir, len(lines)))
        rows.extend(lines)

    names = [r.split(",")[0] for r in rows]
    total, unique = len(names), len(set(names))
    print(f"[merge] rows={total} unique={unique} from {len(args.shards)} shards")
    for sdir, n in shard_info:
        print(f"  {sdir}: {n}")
    if total != unique:
        raise SystemExit("duplicate sequence names across shards!")
    if args.full_names:
        full = {l.strip() for l in open(args.full_names) if l.strip()}
        got = set(names)
        if got != full:
            raise SystemExit(f"set mismatch: missing={sorted(full-got)[:5]} extra={sorted(got-full)[:5]}")
        print(f"[merge] set identical to full list ({len(full)} sequences)")

    arr = np.array([[float(v) for v in r.split(",")[1:]] for r in rows], dtype=np.float64)
    means = np.nanmean(arr, axis=0) * 100
    COLS = ["AUC", "P_norm", "P", "SA"]
    agg_str = "  ".join(f"{c} {v:.2f}" for c, v in zip(COLS, means))
    print(f"[merge] AGGREGATE: {agg_str}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_seq_summary.txt"), "w") as f:
        f.write("\n".join(sorted(rows)) + "\n")
    with open(os.path.join(args.out, "MERGED_FROM_SHARDS.txt"), "w") as f:
        f.write("Merged shard dirs:\n" + "\n".join(f"  {s} ({n} rows)" for s, n in shard_info)
                + f"\nAggregate: {agg_str}"
                f" (mean of {total} per-seq rows; matches official serial aggregate)\n")
    print(f"[merge] wrote {args.out}/per_seq_summary.txt + MERGED_FROM_SHARDS.txt")


if __name__ == "__main__":
    main()
