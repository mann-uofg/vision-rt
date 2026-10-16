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


def latency_table() -> str:
    runs = []
    for model in MODELS:
        base = RESULTS / f"baseline_{model}_fp32-tf32_{CLIP}.json"
        if base.exists():
            runs.append(("PyTorch eager (M0)", json.loads(base.read_text())))
        for prec in PRECISIONS:
            p = RESULTS / f"cpp_sync_{model}_{prec}_{CLIP}.json"
            if p.exists():
                runs.append(("C++ + TensorRT (M1)", json.loads(p.read_text())))

    base_mean = {r["model"]: r["host_ms"]["e2e"]["mean"]
                 for name, r in runs if name.startswith("PyTorch")}
    rows = [
        "| Model | Pipeline | Precision | decode | preprocess | H2D + infer + D2H | NMS "
        "| **end-to-end mean / p50 / p99** | throughput | vs M0 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs:
        h = r["host_ms"]
        e = h["e2e"]
        speedup = base_mean[r["model"]] / e["mean"] if r["model"] in base_mean else None
        rows.append(
            f"| {pretty(r['model'])} | {name} | {r['precision']} "
            f"| {h['decode']['mean']:.2f} | {h['preprocess']['mean']:.2f} | {h['gpu']['mean']:.2f} "
            f"| {h['postprocess']['mean']:.2f} "
            f"| **{e['mean']:.2f} / {e['p50']:.2f} / {e['p99']:.2f}** "
            f"| {r['throughput_fps']:.0f} fps | {fmt(speedup, '.2f')}x |"
        )
    v = runs[0][1]["video"]
    rows += [
        "",
        f"Stage columns are mean ms per frame, host wall-clock; they add up to end-to-end. "
        f"Clip `{v['file']}`, {v['width']}x{v['height']} {v['codec']}, "
        f"{runs[0][1]['timed_frames']} timed frames after "
        f"{runs[0][1]['warmup_frames']} warm-up; batch 1, one stream.",
    ]
    return "\n".join(rows)


def accuracy_table() -> str:
    rows = [
        "| Model | Precision | mAP, PyTorch | mAP, TensorRT | mAP, C++ app | "
        "TensorRT GPU compute (model only) |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        for prec in PRECISIONS:
            torch_ap = coco_ap(model, "pytorch", {"fp32": "fp32", "tf32": "fp32-tf32"}.get(prec, ""))
            rows.append(
                f"| {pretty(model)} | {prec} | {fmt(torch_ap, '.2f')} "
                f"| {fmt(coco_ap(model, 'tensorrt', prec), '.2f')} "
                f"| {fmt(coco_ap(model, 'external', 'cpp-' + prec), '.2f')} "
                f"| {fmt(trtexec_gpu_ms(model, prec), '.2f')} ms |"
            )
    rows += [
        "",
        "COCO val2017 mAP@0.5:0.95, all 5,000 images, score threshold 0.01, NMS IoU 0.65. "
        "\"C++ app\" runs the whole C++ path, including its own FFmpeg JPEG decode. The last "
        "column is trtexec's mean GPU time for the engine alone at batch 1: no decode, no "
        "pre/postprocessing, no copies. It is not an end-to-end number.",
    ]
    return "\n".join(rows)


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {name}:begin -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no {name} markers")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    text = README.read_text()
    text = splice(text, "latency", latency_table())
    text = splice(text, "accuracy", accuracy_table())
    README.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
