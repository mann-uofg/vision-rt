#!/usr/bin/env python3
"""COCO val2017 mAP for the PyTorch model, using the same pre/postprocessing as the baseline.

    python tools/eval_coco.py --size s
    python tools/eval_coco.py --size m --limit 500     # quick check on a subset
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from vrt import coco, yolox
from vrt.postprocess import EVAL_CONF, EVAL_NMS_IOU, postprocess
from vrt.preprocess import letterbox
from vrt.stats import environment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--fp16", action="store_true", help="run the model in half precision")
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 in cuDNN/cuBLAS so FP32 convolutions are true FP32")
    ap.add_argument("--conf", type=float, default=EVAL_CONF, help="score threshold before NMS")
    ap.add_argument("--limit", type=int, help="evaluate only the first N images (sorted by id)")
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    gt = coco.load_val()
    cats = coco.class_to_category(gt)
    image_ids = sorted(gt.getImgIds())[: args.limit]

    model = yolox.build(args.size, args.weights or yolox.default_weights(args.size))
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(dtype)

    results = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for n, img_id in enumerate(image_ids, 1):
            info = gt.loadImgs(img_id)[0]
            bgr = cv2.imread(str(coco.COCO_DIR / "val2017" / info["file_name"]))
            chw, scale = letterbox(bgr, yolox.INPUT_HW)
            x = torch.from_numpy(chw).unsqueeze(0).cuda().to(dtype)
            pred = model(x)[0].float().cpu()
            for x1, y1, x2, y2, score, cls in postprocess(pred, scale, args.conf, EVAL_NMS_IOU):
                results.append({
                    "image_id": img_id,
                    "category_id": cats[int(cls)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })
            if n % 1000 == 0:
                print(f"{n}/{len(image_ids)} images, {time.perf_counter() - t0:.0f} s")

    precision = "fp16" if args.fp16 else ("fp32" if args.no_tf32 else "fp32-tf32")
    tag = f"yolox_{args.size}_torch_{precision}"
    if args.conf != EVAL_CONF:
        tag += f"_conf{args.conf:g}"
    if args.limit:
        tag += f"_first{args.limit}"
    stats = coco.evaluate(gt, results, image_ids, args.out / "detections" / f"{tag}.json")
    report = {
        "model": f"yolox_{args.size}",
        "backend": "pytorch",
        "precision": precision,
        "input": list(yolox.INPUT_HW),
        "conf_thresh": args.conf,
        "nms_iou": EVAL_NMS_IOU,
        "images": len(image_ids),
        "stats": stats,
        "environment": environment(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"coco_{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"AP {stats['AP']:.4f}  AP50 {stats['AP50']:.4f}  -> {args.out / f'coco_{tag}.json'}")


if __name__ == "__main__":
    main()
