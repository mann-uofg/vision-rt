#!/usr/bin/env python3
"""Multi-stream scaling plots from bench/results/multistream_*.json.

Figure 1 (capacity): every stream as fast as it can go.
  left:  aggregate fps vs streams for each engine, with GPU and NVDEC utilisation of the first
  right: pooled p50/p99 latency vs streams for the first engine
Figure 2 (real time): every stream paced at 30 fps, phases staggered like real cameras.
  worst stream's p99 latency vs number of cameras, against the 33 ms frame budget.

    python tools/plot_scaling.py docs/scaling.png docs/realtime.png --engines yolox_s_fp16 yolox_m_fp16
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "bench" / "results"
COLORS = ["#3f7fbf", "#c44e52", "#4da36b", "#e6a33e"]


def runs(engine: str, fps: int) -> list[dict]:
    rs = [json.loads(p.read_text()) for p in RESULTS.glob(f"multistream_{engine}_fps{fps}_n*.json")]
    return sorted(rs, key=lambda r: r["streams"])


def pretty(engine: str) -> str:
    model, prec = engine.rsplit("_", 1)
    return f"{model.replace('_', '-').upper()} {prec.upper()}"


def capacity(engines: list[str], out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9), dpi=150)
    u = a.twinx()
    for k, e in enumerate(engines):
        rs = runs(e, 0)
        if not rs:
            continue
        n = [r["streams"] for r in rs]
        a.plot(n, [r["aggregate_fps"] for r in rs], "o-", color=COLORS[k], label=f"{pretty(e)} fps")
        if k == 0:
            u.plot(n, [r["gpu_util_pct"] for r in rs], "--", color=COLORS[k], alpha=0.5, lw=1,
                   label=f"{pretty(e)} GPU %")
            u.plot(n, [r["decoder_util_pct"] for r in rs], ":", color="#8c6bb1", lw=1.2,
                   label=f"{pretty(e)} NVDEC %")
            b.plot(n, [r["latency_ms"]["p50"] for r in rs], "o-", color="#e6a33e", label="p50")
            b.plot(n, [r["latency_ms"]["p99"] for r in rs], "o-", color="#c44e52", label="p99")
            b.set_title(f"{pretty(e)} latency per frame", fontsize=10)
    a.set_xlabel("concurrent streams")
    a.set_ylabel("aggregate throughput (fps)")
    a.set_ylim(bottom=0)
    u.set_ylabel("utilisation (%)")
    u.set_ylim(0, 105)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = u.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=7, frameon=False)
    a.set_title("Throughput, every stream unthrottled", fontsize=10)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
    b.set_xlabel("concurrent streams")
    b.set_ylabel("ms (decode start to results on host)")
    b.legend(fontsize=8, frameon=False)
    for ax in (a, b):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


def realtime(engines: list[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    for k, e in enumerate(engines):
        rs = runs(e, 30)
        if not rs:
            continue
        ax.plot([r["streams"] for r in rs], [r["worst_stream_p99_ms"] for r in rs], "o-",
                color=COLORS[k], label=f"{pretty(e)}: worst stream p99")
    ax.axhline(1000 / 30, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.text(ax.get_xlim()[0] + 1, 1000 / 30 * 1.15, "one frame interval (33 ms)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("cameras, each 1080p H.264 at 30 fps")
    ax.set_ylabel("latency p99 (ms, log scale)")
    ax.set_title("How many live cameras one RTX 5070 Ti keeps up with", fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capacity_out", type=Path)
    ap.add_argument("realtime_out", type=Path)
    ap.add_argument("--engines", nargs="+", default=["yolox_s_fp16", "yolox_m_fp16"])
    args = ap.parse_args()
    args.capacity_out.parent.mkdir(parents=True, exist_ok=True)
    capacity(args.engines, args.capacity_out)
    realtime(args.engines, args.realtime_out)


if __name__ == "__main__":
    main()
