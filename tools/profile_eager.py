#!/usr/bin/env python3
"""Why the naive pipeline's "infer" stage is slow: count kernels and compare GPU busy time
with the GPU timeline.

Model-only diagnostic on a fixed input. Not an end-to-end number. It answers one question:
how much of eager PyTorch's inference time is the GPU actually computing?

    python tools/profile_eager.py --size s
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import torch

from vrt import yolox
from vrt.stats import environment, summarize_ms


def time_forward(model, x, iters: int) -> tuple[list[float], list[float]]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    host, gpu = [], []
    for _ in range(iters):
        t = time.perf_counter_ns()
        start.record()
        model(x)
        end.record()
        end.synchronize()
        host.append((time.perf_counter_ns() - t) / 1e6)
        gpu.append(start.elapsed_time(end))
    return host, gpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--video", type=Path, default=Path("data/footage/barn022.mp4"),
                    help="decoded in a background thread for the 'with decode' run")
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    model = yolox.build(args.size, yolox.default_weights(args.size))
    x = torch.rand(1, 3, *yolox.INPUT_HW, device="cuda") * 255
    report = {"model": f"yolox_{args.size}", "precision": "fp32-tf32", "batch": 1,
              "note": "model-only diagnostic on a fixed input; not end-to-end"}

    with torch.inference_mode():
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()

        # Kernels per forward and their summed device time (GPU busy time).
        acts = [torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts) as prof:
            model(x)
            torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        report["kernels_per_forward"] = len(kernels)
        report["kernel_time_ms"] = sum(e.device_time for e in kernels) / 1e3

        host, gpu = time_forward(model, x, args.iters)
        report["idle_cpu"] = {"host_ms": summarize_ms(host), "gpu_timeline_ms": summarize_ms(gpu)}

        # Same, with OpenCV decoding the clip in another thread, as in the baseline.
        stop = threading.Event()

        def decode() -> None:
            cap = cv2.VideoCapture(str(args.video))
            while not stop.is_set():
                ok, _ = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        th = threading.Thread(target=decode)
        th.start()
        try:
            host, gpu = time_forward(model, x, args.iters)
        finally:
            stop.set()
            th.join()
        report["with_decode"] = {"host_ms": summarize_ms(host), "gpu_timeline_ms": summarize_ms(gpu)}

    busy = report["kernel_time_ms"] / report["idle_cpu"]["gpu_timeline_ms"]["p50"]
    report["gpu_busy_fraction_p50"] = float(busy)
    report["environment"] = environment()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eager_profile_yolox_{args.size}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{report['kernels_per_forward']} kernels, {report['kernel_time_ms']:.2f} ms of kernel time")
    for k in ("idle_cpu", "with_decode"):
        g = report[k]["gpu_timeline_ms"]
        print(f"{k:<12} GPU timeline p50 {g['p50']:.2f}  p99 {g['p99']:.2f} ms")
    print(f"GPU busy {100 * busy:.0f}% of the forward pass (p50)  -> {out}")


if __name__ == "__main__":
    main()
