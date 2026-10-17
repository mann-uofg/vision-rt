#!/usr/bin/env python3
"""Naive pipeline: the "before" number.

    OpenCV CPU decode -> CPU letterbox -> pageable H2D copy -> PyTorch eager inference
    -> D2H copy -> CPU score filter + NMS

Everything runs one frame at a time on one thread, each stage waiting for the previous one.
That is how most first versions of a detector-on-video script look.

Timing:
- Every stage gets host wall-clock time (perf_counter_ns), and the stages add up to the
  end-to-end time: from asking the decoder for a frame to having detections in host memory.
- The GPU part (H2D, inference, D2H) also gets CUDA events, so the GPU timeline can be
  compared with what the host saw. A gap between the two is time the GPU spent waiting on
  the CPU (kernel launches, Python).

    python tools/baseline.py --video data/footage/barn022.mp4 --size s
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vrt import yolox
from vrt.postprocess import DEPLOY_CONF, DEPLOY_NMS_IOU, postprocess
from vrt.preprocess import letterbox
from vrt.stats import environment, summarize_ms

HOST_STAGES = ["decode", "preprocess", "gpu", "postprocess", "e2e"]
GPU_STAGES = ["h2d", "infer", "d2h"]


def video_info(cap: cv2.VideoCapture, path: Path) -> dict:
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        "file": path.name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 so FP32 convolutions are true FP32 (PyTorch allows it by default)")
    ap.add_argument("--warmup", type=int, default=100, help="frames run before timing starts")
    ap.add_argument("--frames", type=int, help="cap on timed frames (default: rest of video)")
    ap.add_argument("--conf", type=float, default=DEPLOY_CONF)
    ap.add_argument("--nms", type=float, default=DEPLOY_NMS_IOU)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    vinfo = video_info(cap, args.video)

    dtype = torch.float16 if args.fp16 else torch.float32
    model = yolox.build(args.size, yolox.default_weights(args.size)).to(dtype)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    host = {k: [] for k in HOST_STAGES}
    gpu = {k: [] for k in GPU_STAGES}
    dets_per_frame = []
    frame_idx = 0
    cpu_begin = 0.0

    with torch.inference_mode():
        while args.frames is None or frame_idx < args.warmup + args.frames:
            if frame_idx == args.warmup:
                cpu_begin = time.process_time()  # all threads of this process
            t0 = time.perf_counter_ns()
            ok, frame = cap.read()
            if not ok:
                break
            t1 = time.perf_counter_ns()
            chw, scale = letterbox(frame, yolox.INPUT_HW)
            t2 = time.perf_counter_ns()

            events[0].record()
            x = torch.from_numpy(chw).unsqueeze(0).cuda().to(dtype)
            events[1].record()
            out = model(x)
            events[2].record()
            pred = out[0].float().cpu()
            events[3].record()
            t3 = time.perf_counter_ns()

            dets = postprocess(pred, scale, args.conf, args.nms)
            t4 = time.perf_counter_ns()

            if frame_idx >= args.warmup:
                events[3].synchronize()
                for name, a, b in [("decode", t0, t1), ("preprocess", t1, t2), ("gpu", t2, t3),
                                   ("postprocess", t3, t4), ("e2e", t0, t4)]:
                    host[name].append((b - a) / 1e6)
                for i, name in enumerate(GPU_STAGES):
                    gpu[name].append(events[i].elapsed_time(events[i + 1]))
                dets_per_frame.append(dets)
            frame_idx += 1

    cpu_ms = (time.process_time() - cpu_begin) * 1000
    timed = len(host["e2e"])
    if timed == 0:
        raise SystemExit("no frames timed; video shorter than --warmup?")

    precision = "fp16" if args.fp16 else ("fp32" if args.no_tf32 else "fp32-tf32")
    tag = f"baseline_yolox_{args.size}_{precision}_{args.video.stem}"
    e2e = summarize_ms(host["e2e"])
    report = {
        "pipeline": "naive-pytorch",
        "model": f"yolox_{args.size}",
        "precision": precision,
        "input": list(yolox.INPUT_HW),
        "batch": 1,
        "streams": 1,
        "conf_thresh": args.conf,
        "nms_iou": args.nms,
        "video": vinfo,
        "warmup_frames": args.warmup,
        "timed_frames": timed,
        "throughput_fps": 1000.0 / e2e["mean"],
        "cpu_ms_per_frame": cpu_ms / timed,
        "host_ms": {k: summarize_ms(v) for k, v in host.items()},
        "gpu_ms": {k: summarize_ms(v) for k, v in gpu.items()},
        "detections_per_frame": float(np.mean([len(d) for d in dets_per_frame])),
        "threads": {"opencv": cv2.getNumThreads(), "torch": torch.get_num_threads()},
        "environment": environment(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")

    # Per-frame detections, for checking later pipelines against this one.
    det_dir = args.out / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        det_dir / f"{tag}.npz",
        frame=np.concatenate([np.full(len(d), args.warmup + i) for i, d in enumerate(dets_per_frame)]),
        dets=np.concatenate(dets_per_frame) if dets_per_frame else np.zeros((0, 6)),
    )

    print(f"{tag}: {timed} frames timed, {vinfo['width']}x{vinfo['height']} {vinfo['codec']}")
    print(f"{'stage':<14}{'mean':>9}{'p50':>9}{'p99':>9}  (ms)")
    rows = [(f"host {k}", report["host_ms"][k]) for k in HOST_STAGES]
    rows += [(f"  gpu {k}", report["gpu_ms"][k]) for k in GPU_STAGES]
    for name, s in rows:
        print(f"{name:<14}{s['mean']:>9.2f}{s['p50']:>9.2f}{s['p99']:>9.2f}")
    print(f"throughput {report['throughput_fps']:.1f} fps, "
          f"{report['detections_per_frame']:.1f} detections/frame")


if __name__ == "__main__":
    main()
