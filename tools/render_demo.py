#!/usr/bin/env python3
"""Render the demo video: a 2x2 barn-camera wall with the pipeline's detections drawn on it.

The detections come from the real C++ GPU pipeline (vrt_pipeline --dets, one CSV per clip;
bench/make_demo.sh runs it). This script only draws: cows in green and counted, anything else
the model reports in grey with its COCO label, so its mistakes stay visible. The latency
numbers in the footer are read from the results files, not made up.

    python tools/render_demo.py --seconds 30 --out docs/demo.mp4 --gif docs/demo.gif
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from vrt.coco import CLASS_NAMES, COW
from vrt.yolox import REPO_ROOT

CLIPS = ["barn022", "barn012", "barn023", "barn004"]
W, H = 1280, 720  # output canvas
HEADER, FOOTER = 32, 28  # rows reserved for the title and the measured numbers
TILE_H = (H - HEADER - FOOTER) // 2
TILE_W = TILE_H * 16 // 9
X0 = (W - 2 * TILE_W) // 2  # tiles centred, 16:9
GREEN = (80, 200, 60)
GREY = (170, 170, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_dets(path: Path) -> dict[int, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {int(f): a[a[:, 0] == f, 1:] for f in np.unique(a[:, 0])}


def text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_tile(frame: np.ndarray, dets: np.ndarray, cam: int, clip: str) -> tuple[np.ndarray, int]:
    sx, sy = TILE_W / frame.shape[1], TILE_H / frame.shape[0]
    tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cows = 0
    for x1, y1, x2, y2, score, cls in dets:
        cls = int(cls)
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        is_cow = cls == COW
        cows += is_cow
        color = GREEN if is_cow else GREY
        cv2.rectangle(tile, p1, p2, color, 2 if is_cow else 1, cv2.LINE_AA)
        text(tile, f"{CLASS_NAMES[cls]} {score:.2f}", (p1[0] + 3, max(p1[1] - 5, 12)), 0.4, color)
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, TILE_H - 26), (TILE_W, TILE_H), (0, 0, 0), -1)
    tile = cv2.addWeighted(overlay, 0.55, tile, 0.45, 0)
    text(tile, f"CAM {cam}  {clip}", (8, TILE_H - 8), 0.5)
    text(tile, f"cows: {cows}", (TILE_W - 90, TILE_H - 8), 0.55, GREEN, 1)
    return tile, cows


def measured_numbers() -> dict:
    r = REPO_ROOT / "bench" / "results"
    sync = json.loads((r / "gpu_sync_yolox_s_fp16_barn022.json").read_text())
    rt = json.loads((r / "multistream_yolox_s_fp16_fps30_n32.json").read_text())
    # Round the bound up so "p99 < x" stays true after rounding.
    return {"latency": sync["host_ms"]["latency"]["mean"], "cams": rt["streams"],
            "p99_bound": math.ceil(rt["worst_stream_p99_ms"] * 10) / 10}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets-dir", type=Path, default=REPO_ROOT / "bench/results/detections")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "demo.mp4")
    ap.add_argument("--gif", type=Path, help="also write a short GIF preview here")
    ap.add_argument("--gif-seconds", type=float, default=8)
    args = ap.parse_args()

    caps = [cv2.VideoCapture(str(REPO_ROOT / "data" / "footage" / f"{c}.mp4")) for c in CLIPS]
    dets = [load_dets(args.dets_dir / f"demo_{c}.csv") for c in CLIPS]
    m = measured_numbers()
    n_frames = int(args.seconds * args.fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
         "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE)
    empty = np.zeros((0, 6))
    for i in range(n_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        total = 0
        for k, (cap, d, clip) in enumerate(zip(caps, dets, CLIPS)):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"{clip} ended at frame {i}")
            tile, cows = draw_tile(frame, d.get(i, empty), k + 1, clip)
            total += cows
            y, x = HEADER + (k // 2) * TILE_H, X0 + (k % 2) * TILE_W
            canvas[y : y + TILE_H, x : x + TILE_W] = tile

        text(canvas, "vision-rt  |  barn monitoring, 4 cameras  |  YOLOX-S FP16 on an RTX 5070 Ti",
             (10, 22), 0.55)
        text(canvas, f"cows in view: {total:2d}", (W - 180, 22), 0.6, GREEN, 1)
        text(canvas,
             f"NVDEC -> fused CUDA preprocessing -> TensorRT -> GPU NMS  |  measured: "
             f"{m['latency']:.1f} ms per frame, {m['cams']} live 30 fps cameras on one GPU at "
             f"p99 < {m['p99_bound']:.1f} ms", (10, H - 9), 0.42, (220, 220, 220))
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"{args.out}: {n_frames} frames, {args.out.stat().st_size / 1e6:.1f} MB")

    if args.gif:
        palette = args.gif.with_suffix(".palette.png")
        vf = "fps=8,scale=640:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t",
                        str(args.gif_seconds), "-i", str(args.out), "-vf",
                        f"{vf},palettegen=stats_mode=diff", str(palette)], check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-t",
                        str(args.gif_seconds), "-i", str(args.out), "-i", str(palette),
                        "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4",
                        str(args.gif)], check=True)
        palette.unlink()
        print(f"{args.gif}: {args.gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
