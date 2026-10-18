#!/usr/bin/env python3
"""Compare a pipeline's per-frame detections against the naive baseline's.

Correctness before speed: a faster pipeline only counts if it finds the same boxes.
Boxes are matched greedily per frame by IoU (same class only), highest score first.

    python tools/compare_detections.py \
        --ref bench/results/detections/baseline_yolox_s_fp32-tf32_barn022.npz \
        --test bench/results/detections/cpp_yolox_s_fp16_barn022.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[int, np.ndarray]:
    """frame -> [N, 6] array (x1, y1, x2, y2, score, cls)."""
    if path.suffix == ".npz":
        z = np.load(path)
        frames, dets = z["frame"], z["dets"]
    else:
        a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        frames, dets = a[:, 0].astype(int), a[:, 1:]
    out = defaultdict(lambda: np.zeros((0, 6)))
    for f in np.unique(frames):
        out[int(f)] = dets[frames == f]
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def compare(ref: dict, test: dict, frames: list[int], iou_thresh: float) -> dict:
    matched, ref_total, test_total = 0, 0, 0
    ious, score_diffs = [], []
    for f in frames:
        r, t = ref[f], test[f]
        ref_total += len(r)
        test_total += len(t)
        if not len(r) or not len(t):
            continue
        m = iou_matrix(r, t)
        m[r[:, None, 5] != t[None, :, 5]] = 0  # different classes never match
        used = np.zeros(len(t), bool)
        for i in np.argsort(-r[:, 4]):
            cand = np.where(~used)[0]
            if not cand.size:
                break
            j = cand[np.argmax(m[i, cand])]
            if m[i, j] >= iou_thresh:
                used[j] = True
                matched += 1
                ious.append(m[i, j])
                score_diffs.append(abs(r[i, 4] - t[j, 4]))
    return {
        "frames": len(frames),
        "ref_boxes": ref_total,
        "test_boxes": test_total,
        "matched": matched,
        "recall_vs_ref": matched / max(ref_total, 1),
        "precision_vs_ref": matched / max(test_total, 1),
        "iou_thresh": iou_thresh,
        "matched_iou_mean": float(np.mean(ious)) if ious else None,
        "matched_iou_p01": float(np.percentile(ious, 1)) if ious else None,
        "score_absdiff_mean": float(np.mean(score_diffs)) if score_diffs else None,
        "score_absdiff_max": float(np.max(score_diffs)) if score_diffs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-frame", type=int, help="only compare frames from this index on")
    ap.add_argument("--max-frame", type=int, help="only compare frames below this index")
    ap.add_argument("--out", type=Path, help="write the summary as JSON here")
    args = ap.parse_args()

    ref, test = load(args.ref), load(args.test)
    frames = sorted(set(ref) | set(test))
    if args.min_frame is not None:
        frames = [f for f in frames if f >= args.min_frame]
    if args.max_frame is not None:
        frames = [f for f in frames if f < args.max_frame]
    report = {"ref": args.ref.name, "test": args.test.name, **compare(ref, test, frames, args.iou)}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
