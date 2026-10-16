#!/usr/bin/env python3
"""COCO val2017 mAP, with the same pre/postprocessing as the pipelines.

Three sources of detections:
    python tools/eval_coco.py --size s                                   # PyTorch model
    python tools/eval_coco.py --size s --engine models/yolox_s_fp16.engine   # TensorRT engine
    python tools/eval_coco.py --size s --detections out.json --label cpp-fp16  # scored as-is

The --detections form scores a COCO results file written by something else (the C++ app).
--limit N evaluates the first N images by id, for quick checks.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vrt import coco, yolox
from vrt.postprocess import EVAL_CONF, EVAL_NMS_IOU, postprocess
from vrt.preprocess import letterbox
from vrt.stats import environment


def run_model(predict, dtype, gt, image_ids, conf) -> list[dict]:
    cats = coco.class_to_category(gt)
    results = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for n, img_id in enumerate(image_ids, 1):
            info = gt.loadImgs(img_id)[0]
            bgr = cv2.imread(str(coco.COCO_DIR / "val2017" / info["file_name"]))
            chw, scale = letterbox(bgr, yolox.INPUT_HW)
            x = torch.from_numpy(chw).unsqueeze(0).cuda().to(dtype)
            pred = predict(x)[0].float().cpu()
            for x1, y1, x2, y2, score, cls in postprocess(pred, scale, conf, EVAL_NMS_IOU):
                results.append({
                    "image_id": img_id,
                    "category_id": cats[int(cls)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
            if n % 1000 == 0:
                print(f"{n}/{len(image_ids)} images, {time.perf_counter() - t0:.0f} s")
    return results


def csv_to_coco(path: Path, cats: list[int]) -> list[dict]:
    """C++ app output (image_id, x1, y1, x2, y2, score, cls) -> COCO results."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return [
        {
            "image_id": int(r[0]),
            "category_id": cats[int(r[6])],
            "bbox": [float(r[1]), float(r[2]), float(r[3] - r[1]), float(r[4] - r[2])],
            "score": float(r[5]),
        }
        for r in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--fp16", action="store_true", help="PyTorch: run the model in half precision")
    ap.add_argument("--no-tf32", action="store_true",
                    help="PyTorch: disable TF32 so FP32 convolutions are true FP32")
    ap.add_argument("--engine", type=Path, help="evaluate this TensorRT engine instead")
    ap.add_argument("--detections", type=Path, help="score this COCO results file instead")
    ap.add_argument("--label", help="precision label for --detections (e.g. cpp-fp16)")
    ap.add_argument("--conf", type=float, default=EVAL_CONF, help="score threshold before NMS")
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (sorted by id)")
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    gt = coco.load_val()
    image_ids = sorted(gt.getImgIds())[: args.limit]

    if args.detections:
        if not args.label:
            ap.error("--detections needs --label")
        backend, precision = "external", args.label
        wanted = set(image_ids)
        if args.detections.suffix == ".csv":
            results = csv_to_coco(args.detections, coco.class_to_category(gt))
        else:
            results = json.loads(args.detections.read_text())
        results = [r for r in results if r["image_id"] in wanted]
    elif args.engine:
        from vrt.trt import TrtModel

        engine = TrtModel(args.engine)
        backend = "tensorrt"
        precision = args.engine.stem.split("_")[-1]  # yolox_s_fp16.engine -> fp16
        results = run_model(engine, engine.input_dtype, gt, image_ids, args.conf)
    else:
        if args.no_tf32:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        dtype = torch.float16 if args.fp16 else torch.float32
        model = yolox.build(args.size, args.weights or yolox.default_weights(args.size)).to(dtype)
        backend = "pytorch"
        precision = "fp16" if args.fp16 else ("fp32" if args.no_tf32 else "fp32-tf32")
        results = run_model(model, dtype, gt, image_ids, args.conf)

    short = {"pytorch": "torch", "tensorrt": "trt", "external": "ext"}[backend]
    tag = f"yolox_{args.size}_{short}_{precision}"
    if args.conf != EVAL_CONF:
        tag += f"_conf{args.conf:g}"
    if args.limit:
        tag += f"_first{args.limit}"
    stats = coco.evaluate(gt, results, image_ids, args.out / "detections" / f"{tag}.json")
    report = {
        "model": f"yolox_{args.size}",
        "backend": backend,
        "precision": precision,
        "input": list(yolox.INPUT_HW),
        "conf_thresh": args.conf,
        "nms_iou": EVAL_NMS_IOU,
        "images": len(image_ids),
        "stats": stats,
        "environment": environment(),
    }
    if args.engine:
        import tensorrt

        report["engine"] = args.engine.name
        report["environment"]["tensorrt"] = tensorrt.__version__
    if args.detections:
        report["detections_file"] = args.detections.name
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"coco_{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"AP {stats['AP']:.4f}  AP50 {stats['AP50']:.4f}  -> {args.out / f'coco_{tag}.json'}")


if __name__ == "__main__":
    main()
