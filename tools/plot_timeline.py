#!/usr/bin/env python3
"""Draw a few frames of an Nsight Systems trace as a timeline PNG for the README.

Reads the SQLite export of an nsys report (nsys export --type sqlite) and draws, on one time
axis: the host thread's NVTX ranges (decode, submit, collect, ...) and the GPU's kernels and
copies, coloured by pipeline stage. Kernel names are mapped to stages by substring.

    nsys profile -t cuda,nvtx --cuda-graph-trace=node -o run vrt_pipeline ...
    nsys export --type sqlite run.nsys-rep
    python tools/plot_timeline.py run.sqlite docs/timeline.png --frames 6 --title "..."
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Kernel name substring -> stage. First match wins; anything else counts as inference.
KERNEL_STAGES = [
    ("nv12_letterbox", "preprocess"),
    ("score_kernel", "nms"),
    ("gather_kernel", "nms"),
    ("nms_", "nms"),
    ("Radix", "nms"),
    ("radix", "nms"),
    ("DeviceRadixSort", "nms"),
]
COLORS = {
    "decode": "#8c6bb1",
    "preprocess": "#e6a33e",
    "inference": "#3f7fbf",
    "nms": "#4da36b",
    "copy": "#c44e52",
    "launch": "#c9ccd1",
    "wait": "#6b7078",
}

# NVTX range name (first word) -> what the host thread is doing.
HOST_STAGES = {
    "decode": "decode",
    "preprocess": "preprocess",
    "postprocess": "nms",
    "h2d": "wait",      # M1: copy in, enqueue, copy out, stream sync
    "submit": "launch",  # M2: queue the frame's GPU work
    "collect": "wait",   # M2: wait for a frame's results
}


def rows(db: sqlite3.Connection, sql: str) -> list:
    try:
        return db.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []  # table absent (e.g. no memcpy in the trace)


def stage_of(kernel: str) -> str:
    for key, stage in KERNEL_STAGES:
        if key in kernel:
            return stage
    return "inference"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=6, help="how many 'frame'/'decode' ranges to show")
    ap.add_argument("--skip", type=int, default=200, help="skip this many frames (warm-up)")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    strings = dict(rows(db, "SELECT id, value FROM StringIds"))
    nvtx = [
        (s, e, text if text is not None else strings.get(tid, ""))
        for s, e, text, tid in rows(
            db, "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
    ]
    decodes = sorted(r for r in nvtx if r[2].startswith("decode"))
    if len(decodes) < args.skip + args.frames + 1:
        raise SystemExit(f"only {len(decodes)} decode ranges in the trace")
    t0 = decodes[args.skip][0]
    t1 = decodes[args.skip + args.frames][0]

    host = [(s, e, HOST_STAGES[name.split(" ")[0]]) for s, e, name in nvtx
            if s < t1 and e > t0 and name.split(" ")[0] in HOST_STAGES]
    kernels = [
        (s, e, strings.get(n, "")) for s, e, n in rows(
            db, f"SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
                f"WHERE start < {t1} AND end > {t0}")
    ]
    copies = rows(db, f"SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                      f"WHERE start < {t1} AND end > {t0}")

    ms = lambda t: (t - t0) / 1e6  # noqa: E731
    fig, ax = plt.subplots(figsize=(12, 2.6), dpi=150)
    lanes = {"CPU thread": 1.0, "GPU": 0.0}

    for s, e, stage in host:
        ax.broken_barh([(ms(s), ms(e) - ms(s))], (lanes["CPU thread"] - 0.35, 0.7),
                       facecolor=COLORS[stage], edgecolor="white", linewidth=0.3)
    for s, e, name in kernels:
        ax.broken_barh([(ms(s), ms(e) - ms(s))], (lanes["GPU"] - 0.35, 0.7),
                       facecolor=COLORS[stage_of(name)], linewidth=0)
    for s, e, _ in copies:
        ax.broken_barh([(ms(s), max(ms(e) - ms(s), 0.004))], (lanes["GPU"] - 0.35, 0.7),
                       facecolor=COLORS["copy"], linewidth=0)

    ax.set_yticks(list(lanes.values()), list(lanes.keys()))
    ax.set_xlim(0, ms(t1))
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("ms")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, ["decode", "preprocess", "inference", "NMS", "copy", "launch work",
                        "wait for GPU"],
              ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.32), frameon=False, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=10, pad=26)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"{args.out}: {args.frames} frames, {ms(t1):.2f} ms, {len(kernels)} kernels, "
          f"{len(copies)} copies")


if __name__ == "__main__":
    main()
