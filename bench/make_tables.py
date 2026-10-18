#!/usr/bin/env python3
"""Build the README results tables from bench/results/*.json and splice them in.

The README holds each table between <!-- name:begin --> / <!-- name:end --> markers, so
every number in it comes straight from a results file.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
README = ROOT / "README.md"
MODELS = ["yolox_s", "yolox_m"]
PRECISIONS = ["fp32", "tf32", "fp16"]
# Every engine variant with an accuracy number, in table order. YOLOX-M int8 (INT8 with FP16
# elsewhere) is broken on TensorRT 11.3 (see INTERVIEW H2); int8fp32 keeps FP32 elsewhere.
VARIANTS = {
    "yolox_s": ["fp32", "tf32", "fp16", "int8", "int8max"],
    "yolox_m": ["fp32", "tf32", "fp16", "int8", "int8fp32"],
}
LATENCY_VARIANTS = {"yolox_s": ["fp32", "tf32", "fp16", "int8"],
                    "yolox_m": ["fp32", "tf32", "fp16", "int8fp32"]}
LABELS = {"int8": "int8 (entropy)", "int8max": "int8 (max calib.)",
          "int8fp32": "int8, fp32 elsewhere"}
CLIP = "barn022"


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def pretty(model: str) -> str:
    return model.replace("_", "-").upper()


def coco_ap(model: str, backend: str, precision: str) -> float | None:
    for r in load("coco_*.json"):
        if (r["model"], r["backend"], r["precision"], r["conf_thresh"], r["images"]) == (
            model, backend, precision, 0.01, 5000
        ):
            return 100 * r["stats"]["AP"]
    return None


def trtexec_gpu_ms(model: str, precision: str) -> float | None:
    p = RESULTS / "engines" / f"{model}_{precision}.trtexec.txt"
    if not p.exists():
        return None
    m = re.search(r"GPU Compute Time: .*?mean = ([\d.]+) ms", p.read_text())
    return float(m.group(1)) if m else None


def fmt(v: float | None, spec: str = ".1f") -> str:
    return "—" if v is None else format(v, spec)


STEPS = [
    # (results file prefix, label)
    ("baseline", "M0: PyTorch eager, CPU decode/pre/NMS"),
    ("cpp_sync", "M1: C++ + TensorRT, CPU decode/pre/NMS"),
    ("cpp_sync_graph", "M1 + CUDA graph for the engine"),
    ("gpu_sync", "M2: NVDEC + GPU pre/NMS, one frame at a time"),
    ("gpu_pipelined", "M2, decode overlapped with GPU work"),
]


def load_run(prefix: str, model: str, precision: str) -> dict | None:
    name = f"{prefix}_{model}_{'fp32-tf32' if prefix == 'baseline' else precision}_{CLIP}.json"
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def stages(r: dict) -> dict:
    """Per-stage mean ms, whichever pipeline produced the file."""
    h, g = r["host_ms"], r["gpu_ms"]
    if r["pipeline"] == "gpu-nvdec":
        return {
            "decode": h["decode"]["mean"], "preprocess": g["preprocess"]["mean"],
            "infer": g["infer"]["mean"], "nms": g["postprocess"]["mean"],
            "copies": g["d2h"]["mean"], "latency": h["latency"],
        }
    return {
        "decode": h["decode"]["mean"], "preprocess": h["preprocess"]["mean"],
        "infer": g["infer"]["mean"], "nms": h["postprocess"]["mean"],
        "copies": g["h2d"]["mean"] + g["d2h"]["mean"], "latency": h["e2e"],
    }


def step_row(label: str, r: dict, base: dict | None) -> str:
    s = stages(r)
    lat = s["latency"]
    gain = r["throughput_fps"] / base["throughput_fps"] if base else None
    cpu = r.get("cpu_ms_per_frame")
    return (
        f"| {label} | {r['precision']} | {s['decode']:.2f} | {s['preprocess']:.2f} "
        f"| {s['infer']:.2f} | {s['nms']:.2f} | {s['copies']:.2f} "
        f"| **{lat['mean']:.2f}** / {lat['p50']:.2f} / {lat['p99']:.2f} "
        f"| **{r['throughput_fps']:.0f}** | {fmt(gain, '.1f')}x | {fmt(cpu, '.1f')} |"
    )


HEADER = [
    "| Pipeline | Precision | decode | preprocess | inference | NMS | copies "
    "| **latency** mean / p50 / p99 | **fps** | vs M0 | CPU ms/frame |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]


def journey_table(model: str = "yolox_s", precision: str = "fp16") -> str:
    base = load_run("baseline", model, precision)
    rows = list(HEADER)
    for prefix, label in STEPS:
        r = load_run(prefix, model, precision)
        if r:
            rows.append(step_row(label, r, base))
    v = (base or {}).get("video", {})
    rows += ["", NOTE.format(file=v.get("file", "?"), w=v.get("width"), h=v.get("height"),
                             n=(base or {}).get("timed_frames"))]
    return "\n".join(rows)


def m2_table() -> str:
    rows = [
        "| Model | Precision | mAP (TensorRT) | M2 sync latency mean / p99 | M2 sync fps "
        "| M2 pipelined fps | GPU time per frame | vs M0 (fps) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        base = load_run("baseline", model, "")
        for prec in LATENCY_VARIANTS[model]:
            s, pl = load_run("gpu_sync", model, prec), load_run("gpu_pipelined", model, prec)
            if not (s and pl):
                continue
            gain = pl["throughput_fps"] / base["throughput_fps"] if base else None
            rows.append(
                f"| {pretty(model)} | {LABELS.get(prec, prec)} "
                f"| {fmt(coco_ap(model, 'tensorrt', prec), '.2f')} "
                f"| {s['host_ms']['latency']['mean']:.2f} / {s['host_ms']['latency']['p99']:.2f} ms "
                f"| {s['throughput_fps']:.0f} | {pl['throughput_fps']:.0f} "
                f"| {s['gpu_ms']['total']['mean']:.2f} ms | {fmt(gain, '.1f')}x |"
            )
    return "\n".join(rows)


NOTE = (
    "Mean ms per frame on `{file}` ({w}x{h} H.264), batch 1, one stream, {n} timed frames after "
    "100 warm-up. CPU stages are host wall-clock; GPU stages (inference everywhere, and "
    "preprocess/NMS/copies in M2) are CUDA event times. Latency is from asking for a frame to "
    "having its detections in host memory. In the pipelined run stages overlap, so they don't "
    "add up to latency, and latency includes waiting behind the next frame's decode. CPU ms/frame "
    "is process CPU time (all threads) over the timed frames."
)


def accuracy_table() -> str:
    rows = [
        "| Model | Precision | mAP, PyTorch | mAP, TensorRT | mAP, C++ app | "
        "TensorRT GPU compute (model only) |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for prec in VARIANTS[model]:
            torch_ap = coco_ap(model, "pytorch", {"fp32": "fp32", "tf32": "fp32-tf32"}.get(prec, ""))
            label = LABELS.get(prec, prec)
            if (model, prec) == ("yolox_m", "int8"):
                label += " ⚠"
            rows.append(
                f"| {pretty(model)} | {label} | {fmt(torch_ap, '.2f')} "
                f"| {fmt(coco_ap(model, 'tensorrt', prec), '.2f')} "
                f"| {fmt(coco_ap(model, 'external', 'cpp-' + prec), '.2f')} "
                f"| {fmt(trtexec_gpu_ms(model, prec), '.2f')} ms |"
            )
    rows += [
        "",
        "COCO val2017 mAP@0.5:0.95, all 5,000 images, score threshold 0.01, NMS IoU 0.65. "
        "\"C++ app\" runs the whole C++ path, including its own FFmpeg JPEG decode. The last "
        "column is trtexec's mean GPU time for the engine alone at batch 1: no decode, no "
        "pre/postprocessing, no copies. It is not an end-to-end number. INT8 is post-training "
        "quantization with explicit Q/DQ (Model Optimizer, entropy or max calibration on 512 "
        "COCO train images). ⚠ YOLOX-M with INT8 + FP16 gives wrong output on TensorRT 11.3; "
        "keeping the non-quantized layers in FP32 restores accuracy but is no faster than plain "
        "FP16, so YOLOX-M deploys as FP16.",
    ]
    return "\n".join(rows)


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {name}:begin -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no {name} markers")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    text = README.read_text()
    text = splice(text, "latency", journey_table())
    text = splice(text, "m2", m2_table())
    text = splice(text, "accuracy", accuracy_table())
    README.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
