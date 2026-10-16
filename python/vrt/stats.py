"""Latency summaries and the environment block every results file carries."""

import platform
import subprocess

import numpy as np
import torch


def summarize_ms(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def environment() -> dict:
    env = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "python": platform.python_version(),
        "kernel": platform.release(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        env["driver"] = out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        env["driver"] = None
    try:
        import cv2

        env["opencv"] = cv2.__version__
    except ImportError:
        pass
    return env
