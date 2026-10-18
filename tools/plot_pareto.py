#!/usr/bin/env python3
"""Accuracy vs latency for every model size and precision.

x: end-to-end latency per frame on the M2 pipeline, one frame at a time (gpu_sync results).
y: COCO val2017 mAP@0.5:0.95 of the same TensorRT engine.

    python tools/plot_pareto.py docs/pareto.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "bench" / "results"
# YOLOX-M's int8 point is int8fp32: INT8 + FP16 is broken for M on TensorRT 11.3 (INTERVIEW H2).
PRECISIONS = ["fp32", "tf32", "fp16", "int8", "int8fp32"]
MARKERS = {"fp32": "o", "tf32": "s", "fp16": "D", "int8": "^", "int8fp32": "v"}
COLORS = {"yolox_s": "#3f7fbf", "yolox_m": "#c44e52"}
LABELS = {"int8fp32": "int8 (fp32 rest)"}


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--clip", default="barn022")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    points = []
    for model in COLORS:
        xs, ys = [], []
        for prec in PRECISIONS:
            if (model, prec) == ("yolox_m", "int8"):
                continue  # wrong output on TRT 11.3; plotted as int8fp32 instead
            lat = load(f"gpu_sync_{model}_{prec}_{args.clip}.json")
            acc = load(f"coco_{model}_trt_{prec}.json")
            if not (lat and acc):
                continue
            x, y = lat["host_ms"]["latency"]["mean"], 100 * acc["stats"]["AP"]
            xs.append(x)
            ys.append(y)
            points.append((x, y))
            ax.scatter(x, y, marker=MARKERS[prec], s=55, color=COLORS[model], zorder=3)
            ax.annotate(LABELS.get(prec, prec), (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
        ax.plot(xs, ys, color=COLORS[model], alpha=0.35, lw=1)
        ax.plot([], [], color=COLORS[model], label=model.replace("_", "-").upper())

    # Pareto frontier: points no other point beats on both axes. Differences under 0.05 mAP or
    # 2% latency are measurement noise and count as ties (fp32/tf32/fp16 differ by ~0.01 mAP).
    def dominates(q, p):
        no_slower = q[0] <= p[0] * 1.02
        no_worse = q[1] >= p[1] - 0.05
        better = q[0] < p[0] * 0.98 or q[1] > p[1] + 0.05
        return q != p and no_slower and no_worse and better

    front = sorted(p for p in points if not any(dominates(q, p) for q in points))
    if front:
        ax.plot(*zip(*front), "k--", lw=0.8, alpha=0.6, label="Pareto frontier")
    ax.set_xlabel("latency per frame, M2 pipeline (ms, mean)")
    ax.set_ylabel("COCO val2017 mAP@0.5:0.95")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Accuracy vs latency, RTX 5070 Ti, batch 1, 1080p input", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
