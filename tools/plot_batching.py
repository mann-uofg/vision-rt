#!/usr/bin/env python3
"""Cross-camera batching vs one batch-1 context per camera.

left:  unthrottled total throughput vs max batch size (bench/results/batched_fps0_*), with the
       per-camera-context design's capacity (multistream_*_fps0) as a reference line
right: 30 fps cameras, worst stream's p99 vs number of cameras, batched vs per-camera contexts

    python tools/plot_batching.py docs/batching.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "bench" / "results"
COLORS = {1: "#9aa0a6", 2: "#8c6bb1", 4: "#e6a33e", 8: "#3f7fbf", 16: "#4da36b"}


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in RESULTS.glob(pattern)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--engine", default="yolox_s_fp16")
    args = ap.parse_args()

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4), dpi=150)

    # Capacity vs max batch, one line per camera count.
    cap = load("batched_fps0_n*_b*.json")
    # The second line sits just above the first, so its labels go above and the first's below.
    for i, n in enumerate(sorted({r["streams"] for r in cap})):
        color = ["#3f7fbf", "#c44e52"][i % 2]
        rs = sorted((r for r in cap if r["streams"] == n), key=lambda r: r["max_batch"])
        xs = [r["max_batch"] for r in rs]
        a.plot(xs, [r["aggregate_fps"] for r in rs], "o-", color=color, label=f"{n} cameras, batched")
        for r in rs:
            a.annotate(f"{r['batch_size']['mean']:.1f}", (r["max_batch"], r["aggregate_fps"]),
                       textcoords="offset points", xytext=(0, 7 if i else -12), ha="center", fontsize=7,
                       color=color)
    per_ctx = load(f"multistream_{args.engine}_fps0_n*.json")
    if per_ctx:
        best = max(r["aggregate_fps"] for r in per_ctx)
        a.axhline(best, color="k", ls="--", lw=0.8, alpha=0.6)
        a.text(1, best * 1.01, f"one batch-1 context per camera (best: {best:.0f})", fontsize=7)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    a.set_xlabel("max batch size (frames from different cameras)")
    a.set_ylabel("total throughput, all cameras (fps)")
    a.set_ylim(bottom=0)
    a.set_title("Unthrottled capacity (labels: mean batch actually formed)", fontsize=9)
    a.legend(fontsize=8, frameon=False, loc="lower right")

    # Real-time cameras: worst stream p99 vs camera count.
    rt = load("batched_fps30_n*_b*.json")
    for mb in sorted({r["max_batch"] for r in rt}):
        rs = sorted((r for r in rt if r["max_batch"] == mb), key=lambda r: r["streams"])
        b.plot([r["streams"] for r in rs], [r["worst_stream_p99_ms"] for r in rs], "o-",
               color=COLORS.get(mb, "#333333"), label=f"batched, max batch {mb}")
    ctx = sorted(load(f"multistream_{args.engine}_fps30_n*.json"), key=lambda r: r["streams"])
    if ctx:
        b.plot([r["streams"] for r in ctx], [r["worst_stream_p99_ms"] for r in ctx], "s--",
               color="#c44e52", label="one batch-1 context per camera")
    b.axhline(1000 / 30, color="k", ls=":", lw=0.8)
    b.text(b.get_xlim()[0] + 1, 1000 / 30 * 1.15, "33 ms (one frame)", fontsize=7)
    b.set_yscale("log")
    b.set_xlabel("cameras at 30 fps")
    b.set_ylabel("worst camera's p99 latency (ms, log)")
    b.set_title("Live cameras: when does the tail break?", fontsize=9)
    b.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in (a, b):
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
