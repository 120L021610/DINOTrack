"""Evaluation sequence loaders for the five benchmarks (§4.2).

All datasets are consumed in the per-sequence layout produced by
scripts/preprocess/ (OSTrack devkit layout):
  root/lasot/<class>/<class>-<id>/{img/00000001.jpg, groundtruth.txt, ...}
  root/tnl2k/test/<seq>/{img/, groundtruth.txt}
  root/got10k/{val,test}/<seq>/{img/, groundtruth.txt}   (+ list.txt)
  root/trackingnet/TEST/{frames/<vid>/*.jpg}             (no public GT)
  root/lasot_extension_subset/<seq>/{img/, groundtruth.txt}

Boxes on disk are (x, y, w, h). GOT-10k / TrackingNet test have no public GT:
run.py only exports the official submission files for the server (paper
§4.2 "submit the test results to the official server").
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TestSequence:
    name: str
    image_paths: List[str]
    gt: Optional[np.ndarray]      # (N,4) xywh or None (server-side eval only)
    target_visible: Optional[np.ndarray] = None  # (N,), LaSOT occ/oov protocol
    attributes: Optional[dict] = None  # {attr_name: bool} per sequence (UAV sets)


def _load_gt(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64).reshape(-1, 4)


def _list_images(img_dir: str) -> List[str]:
    files = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not files:
        files = glob.glob(os.path.join(img_dir, "*.png"))
    # TEMPORAL order, not lexicographic: TrackingNet frames are unpadded
    # (0.jpg, 10.jpg, 100.jpg) — plain sorted() shuffles video order and
    # silently wrecks online tracking (server-side symptom: Coverage 100 but
    # Success ~33). Padded names (LaSOT/GOT-10k %08d) are order-identical
    # under both rules.
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if stems and all(s.isdigit() for s in stems):
        files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    else:
        files.sort()
    return files


def _seq_img_dir(seq_dir: str) -> str:
    """TNL2K uses 'imgs/', the other benchmarks 'img/'."""
    for name in ("img", "imgs"):
        p = os.path.join(seq_dir, name)
        if os.path.isdir(p):
            return p
    return os.path.join(seq_dir, "img")


def load_lasot(root: str) -> List[TestSequence]:
    """LaSOT test = all sequences except those in the training split file."""
    specs = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))), "lib", "train", "data_specs")
    with open(os.path.join(specs, "lasot_train_split.txt")) as f:
        train = {line.strip() for line in f if line.strip()}
    out = []
    for cls in sorted(os.listdir(root)):
        cd = os.path.join(root, cls)
        if not os.path.isdir(cd):
            continue
        for name in sorted(os.listdir(cd)):
            if name in train:
                continue
            sd = os.path.join(cd, name)
            gt_p = os.path.join(sd, "groundtruth.txt")
            imgs = _list_images(os.path.join(sd, "img"))
            if imgs and os.path.exists(gt_p):
                occ_p = os.path.join(sd, "full_occlusion.txt")
                oov_p = os.path.join(sd, "out_of_view.txt")
                visible = None
                if os.path.exists(occ_p) and os.path.exists(oov_p):
                    occ = np.loadtxt(occ_p, delimiter=",", dtype=np.uint8).reshape(-1)
                    oov = np.loadtxt(oov_p, delimiter=",", dtype=np.uint8).reshape(-1)
                    if len(occ) != len(imgs) or len(oov) != len(imgs):
                        raise ValueError(f"visibility/frame length mismatch in {sd}")
                    visible = ~(occ.astype(bool) | oov.astype(bool))
                out.append(TestSequence(name, imgs, _load_gt(gt_p), visible))
    return out


def load_tnl2k(root: str) -> List[TestSequence]:
    out = []
    base = os.path.join(root, "test")
    for name in sorted(os.listdir(base)):
        sd = os.path.join(base, name)
        gt_p = os.path.join(sd, "groundtruth.txt")
        imgs = _list_images(_seq_img_dir(sd))
        if imgs and os.path.exists(gt_p):
            out.append(TestSequence(name, imgs, _load_gt(gt_p)))
    return out


def load_got10k(root: str, split: str) -> List[TestSequence]:
    base = os.path.join(root, split)
    list_file = os.path.join(base, "list.txt")
    if os.path.exists(list_file):
        with open(list_file) as f:
            import csv as _csv
            names = [r[0] for r in _csv.reader(f) if r]
    else:
        names = sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))
    out = []
    for name in names:
        sd = os.path.join(base, name)
        gt_p = os.path.join(sd, "groundtruth.txt")
        # devkit layout keeps frames directly in the sequence dir; some
        # extractions nest them under img/ — accept both
        imgs = _list_images(os.path.join(sd, "img")) or _list_images(sd)
        out.append(TestSequence(name, imgs,
                                _load_gt(gt_p) if os.path.exists(gt_p) else None))
    return out


def load_trackingnet(root: str) -> List[TestSequence]:
    base = os.path.join(root, "TEST")
    frames = os.path.join(base, "frames")
    out = []
    for name in sorted(os.listdir(frames)):
        imgs = _list_images(os.path.join(frames, name))
        # official protocol: TEST anno contains ONLY the first-frame box, used
        # for initialization; full-GT scoring happens on the eval server
        gt: Optional[np.ndarray] = None
        anno_p = os.path.join(base, "anno", name + ".txt")
        if os.path.exists(anno_p):
            gt = _load_gt(anno_p)
        out.append(TestSequence(name, imgs, gt))
    return out


def load_lasot_ext(root: str) -> List[TestSequence]:
    """Supports both a flat root/<seq>/ layout and the nested official
    LaSOT-Extension layout root/<class>/<class>-<id>/ (as shipped in archives)."""
    out = []

    def _add_seq(name: str, sd: str) -> None:
        gt_p = os.path.join(sd, "groundtruth.txt")
        imgs = _list_images(os.path.join(sd, "img"))
        if imgs and os.path.exists(gt_p):
            out.append(TestSequence(name, imgs, _load_gt(gt_p)))

    for name in sorted(os.listdir(root)):
        sd = os.path.join(root, name)
        if not os.path.isdir(sd):
            continue
        if os.path.exists(os.path.join(sd, "groundtruth.txt")):
            _add_seq(name, sd)                      # flat: root/<seq>/
        else:                                       # nested: root/<class>/<seq>/
            for sub in sorted(os.listdir(sd)):
                ssd = os.path.join(sd, sub)
                if os.path.isdir(ssd):
                    _add_seq(sub, ssd)
    return out


# --- UAV / anti-drone benchmarks -------------------------------------------------

UAV_ANTI_UAV_ATT_NAMES = [
    'Camera_Motion', 'Viewpoint_Changes', 'Partial_Occlusion', 'Full_Occlusion',
    'Out_of_View', 'Rotation', 'Similar_Distractors', 'Illumination_Variations',
    'Motion_Blur', 'Partial_Target_Information', 'Small_Object', 'Fast_Motion',
    'Scale_Variation', 'Aspect_Ratio_Variation',
]   # + Length_short/medium/long from row 14 (0/1/2 scalar), MMOT toolkit mapping
ANTIUAV410_ATT_NAMES = ['TC', 'OV', 'SV', 'FM', 'OC', 'DBC', 'TS', 'SS', 'MS', 'NS']


def _parse_uav_anti_uav_attrs(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    v = np.loadtxt(path, delimiter=',', dtype=np.float64).reshape(-1)
    d = {name: bool(v[i]) for i, name in enumerate(UAV_ANTI_UAV_ATT_NAMES) if i < len(v)}
    if len(v) > 14:
        length_cls = int(v[14])
        d['Length_short'] = length_cls == 0
        d['Length_medium'] = length_cls == 1
        d['Length_long'] = length_cls == 2
    return d


def load_uav_anti_uav(root: str, split: str = "test",
                      max_seqs: Optional[int] = None) -> List[TestSequence]:
    base = os.path.join(root, "Train" if split == "train" else "Test")
    out = []
    names = sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))
    if max_seqs:
        names = names[:max_seqs]
    for name in names:
        sd = os.path.join(base, name)
        gt_p = os.path.join(sd, "groundtruth_rect.txt")
        imgs = (_list_images(os.path.join(sd, "imgs")) or _list_images(sd))
        if not imgs or not os.path.exists(gt_p):
            continue
        gt = np.loadtxt(gt_p, delimiter=',', dtype=np.float64).reshape(-1, 4)
        visible = None
        ab = os.path.join(sd, "absent.txt")
        if os.path.exists(ab):
            absent = np.loadtxt(ab, dtype=np.int64).reshape(-1)
            if len(absent) == len(gt):
                visible = absent == 0
        attrs = _parse_uav_anti_uav_attrs(os.path.join(sd, "attributes.txt"))
        out.append(TestSequence(name, imgs, gt, visible, attrs))
    return out


def load_antiuav318(root: str, split: str = "test",
                    max_seqs: Optional[int] = None) -> List[TestSequence]:
    """AntiUAV318-RGB: frames <split>/<seq>/000001.jpg…, GT
    annos/<seq>.json {gt_rect(N,4 xywh), exist(N)}."""
    base = os.path.join(root, split)
    annos = os.path.join(root, "annos")
    import json as _json
    out = []
    names = sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))
    if max_seqs:
        names = names[:max_seqs]
    for name in names:
        jp = os.path.join(annos, name + ".json")
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            lbl = _json.load(f)
        rects = [r if r else [0, 0, 0, 0] for r in lbl['gt_rect']]
        gt = np.array(rects, dtype=np.float64).reshape(-1, 4)
        imgs = [os.path.join(base, name, '{:06d}.jpg'.format(k))
                for k in range(1, len(gt) + 1)]
        if imgs and not os.path.exists(imgs[-1]):
            imgs = _list_images(os.path.join(base, name))[:len(gt)]
        visible = np.asarray(lbl['exist'], dtype=bool)
        if len(visible) != len(gt):
            print(f"[warn] exist/gt length mismatch in {name}")
            continue
        out.append(TestSequence(name, imgs, gt, visible))   # no attribute files
    return out


def load_antiuav410(root: str, split: str = "test",
                    max_seqs: Optional[int] = None) -> List[TestSequence]:
    """AntiUAV410: frames <split>/<seq>/000001.jpg…, GT per-seq
    IR_label.json {gt_rect, exist}; per-sequence 10-column attributes at
    <root>/annos/<split>/att/<seq>.txt. NOTE: frames are thermal IR content."""
    import json as _json
    base = os.path.join(root, split)
    att_base = os.path.join(root, "annos", split, "att")
    out = []
    names = sorted(d for d in os.listdir(base)
                   if os.path.isfile(os.path.join(base, d, "IR_label.json")))
    if max_seqs:
        names = names[:max_seqs]
    for name in names:
        sd = os.path.join(base, name)
        with open(os.path.join(sd, "IR_label.json")) as f:
            lbl = _json.load(f)
        rects = [r if r else [0, 0, 0, 0] for r in lbl['gt_rect']]
        gt = np.array(rects, dtype=np.float64).reshape(-1, 4)
        imgs = _list_images(sd)
        if len(imgs) != len(gt):
            print(f"[warn] frame/gt mismatch in {name} ({len(imgs)} vs {len(gt)})")
            m = min(len(imgs), len(gt))
            imgs, gt = imgs[:m], gt[:m]
        visible = np.asarray(lbl['exist'], dtype=bool)[:len(gt)]
        attrs = None
        ap = os.path.join(att_base, name + ".txt")
        if os.path.exists(ap):
            v = np.loadtxt(ap, delimiter=',', dtype=np.float64).reshape(-1)
            attrs = {ANTIUAV410_ATT_NAMES[i]: bool(v[i]) for i in range(min(len(v), 10))}
        out.append(TestSequence(name, imgs, gt, visible, attrs))
    return out


def load_ard100(root: str, split: str = "test",
                max_seqs: Optional[int] = None) -> List[TestSequence]:
    """ARD100 (RGB stream as extracted here): frames <split>/<seq>/<seq>_0001.jpg…,
    GT annotations/<seq>_gt.txt (space x,y,w,h; absent rows = '-100 -100 -100 -100')."""
    base = os.path.join(root, split)
    annos = os.path.join(root, "annotations")
    out = []
    names = sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))
    if max_seqs:
        names = names[:max_seqs]
    for name in names:
        gt_p = os.path.join(annos, name + "_gt.txt")
        if not os.path.exists(gt_p):
            continue
        gt = np.loadtxt(gt_p, dtype=np.float64).reshape(-1, 4)
        sd = os.path.join(base, name)
        imgs = sorted([os.path.join(sd, f) for f in os.listdir(sd) if f.endswith('.jpg')],
                      key=lambda p: int(os.path.splitext(p)[0].rsplit('_', 1)[-1]))
        m = min(len(imgs), len(gt))
        imgs, gt = imgs[:m], gt[:m]
        visible = (gt[:, 2] > 0) & (gt[:, 3] > 0) & (gt[:, 0] >= 0) & (gt[:, 1] >= 0)
        out.append(TestSequence(name, imgs, gt, visible))   # no attribute files
    return out


LOADERS = {"lasot": lambda r: load_lasot(os.path.join(r, "lasot")),
           "tnl2k": lambda r: load_tnl2k(os.path.join(r, "tnl2k")),
           "got10k_val": lambda r: load_got10k(os.path.join(r, "got10k"), "val"),
           "got10k_test": lambda r: load_got10k(os.path.join(r, "got10k"), "test"),
           "trackingnet": lambda r: load_trackingnet(os.path.join(r, "trackingnet")),
           "lasot_ext": lambda r: load_lasot_ext(os.path.join(r, "lasot_extension_subset")),
           "uav_anti_uav": lambda r: load_uav_anti_uav(os.path.join(r, "UAV-Anti-UAV")),
           "antiuav318_rgb": lambda r: load_antiuav318(os.path.join(r, "AntiUAV318-RGB")),
           "antiuav410": lambda r: load_antiuav410(os.path.join(r, "antiuav410")),
           "ard100": lambda r: load_ard100(os.path.join(r, "ARD100"))}
