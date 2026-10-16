#!/usr/bin/env python3
"""Build the README results tables from bench/results/*.json and splice them in.

The README holds the tables between <!-- name:begin --> / <!-- name:end --> markers, so
every number in it comes straight from a results file.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
README = ROOT / "README.md"


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob(pattern))]


def coco_ap(model: str, precision: str, conf: float = 0.01) -> str:
    for r in load("coco_*.json"):
        if (r["model"], r["precision"], r["backend"], r["conf_thresh"], r["images"]) == (
            model, precision, "pytorch", conf, 5000
        ):
            return f"{100 * r['stats']['AP']:.1f}"
    return "—"


def baseline_table() -> str:
    rows = [
        "| Model | Precision | mAP | decode | preprocess | H2D + infer + D2H | NMS | **end-to-end p50 / p99** | throughput |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = ["yolox_s", "yolox_m"]
    for r in sorted(load("baseline_*.json"), key=lambda r: (order.index(r["model"]), r["precision"])):
        h = r["host_ms"]
        rows.append(
            f"| {r['model'].replace('_', '-').upper()} "
            f"| {r['precision']} | {coco_ap(r['model'], r['precision'])} "
            f"| {h['decode']['mean']:.2f} | {h['preprocess']['mean']:.2f} | {h['gpu']['mean']:.2f} "
            f"| {h['postprocess']['mean']:.2f} "
            f"| **{h['e2e']['p50']:.1f} / {h['e2e']['p99']:.1f}** | {r['throughput_fps']:.0f} fps |"
        )
    v = load("baseline_*.json")[0]["video"]
    rows.append("")
    rows.append(
        f"Stage columns are mean ms per frame (host wall-clock). Clip: `{v['file']}`, "
        f"{v['width']}x{v['height']} {v['codec']}, {v['fps']:.0f} fps; batch 1, one stream. "
        "mAP is COCO val2017 mAP@0.5:0.95 with the same pre/postprocessing."
    )
    return "\n".join(rows)


def splice(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {name}:begin -->\n).*?(<!-- {name}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no {name} markers")
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    text = README.read_text()
    text = splice(text, "baseline", baseline_table())
    README.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
