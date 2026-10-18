#!/usr/bin/env python3
"""TensorRT's built-in NMS layer vs my NMS, on the same raw predictions.

The engine (built from tools/add_nms.py output) returns both the head's predictions and the
NMS layer's selected indices, so for every image the two NMS implementations see exactly the
same boxes and scores. Reports how often the kept sets agree, and COCO mAP for each.

    python tools/compare_nms_layer.py --size s \
        --engine models/yolox_s_fp16_nms_c0.01_i0.65.engine --conf 0.01 --iou 0.65
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from vrt import coco, yolox
from vrt.postprocess import cxcywh_to_xyxy, postprocess
from vrt.preprocess import letterbox
from vrt.trt import TrtModel


def layer_dets(pred: torch.Tensor, selected: torch.Tensor, scale: float) -> np.ndarray:
    """NMS-layer output -> [N, 6] like postprocess(), sorted by score."""
    if selected.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
    cls = selected[:, 1].long()
    idx = selected[:, 2].long()
    p = pred[idx]
    score = p[:, 4] * p.gather(1, (5 + cls).unsqueeze(1)).squeeze(1)  # objectness * class score
    boxes = cxcywh_to_xyxy(p[:, :4]) / scale
    out = torch.cat([boxes, score[:, None], cls[:, None].float()], 1)
    return out[torch.argsort(out[:, 4], descending=True, stable=True)].numpy()


def same_set(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    key = lambda d: np.lexsort((d[:, 0], d[:, 5], -d[:, 4]))  # noqa: E731
    return np.allclose(a[key(a)], b[key(b)], atol=1e-3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--plain-engine", type=Path, help="same model without NMS, for timing")
    ap.add_argument("--conf", type=float, required=True)
    ap.add_argument("--iou", type=float, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("bench/results"))
    args = ap.parse_args()

    gt = coco.load_val()
    cats = coco.class_to_category(gt)
    image_ids = sorted(gt.getImgIds())[: args.limit]
    model = TrtModel(args.engine)
    plain = TrtModel(args.plain_engine) if args.plain_engine else None
    t_nms, t_plain = [], []

    results = {"layer": [], "mine": []}
    agree, count_diff, n_layer, n_mine = 0, [], 0, 0
    with torch.inference_mode():
        for img_id in image_ids:
            info = gt.loadImgs(img_id)[0]
            bgr = cv2.imread(str(coco.COCO_DIR / "val2017" / info["file_name"]))
            chw, scale = letterbox(bgr, yolox.INPUT_HW)
            x = torch.from_numpy(chw).unsqueeze(0).cuda().to(model.input_dtype)
            # Time both engines the same way on the same real image: run, then wait for the
            # result (the NMS layer's data-dependent output forces a sync inside the run).
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.run(x)
            torch.cuda.synchronize()
            t_nms.append((time.perf_counter() - t0) * 1e3)
            if plain is not None:
                t0 = time.perf_counter()
                plain.run(x)
                torch.cuda.synchronize()
                t_plain.append((time.perf_counter() - t0) * 1e3)
            pred = out["preds"][0].float().cpu()
            sel = out["selected"].cpu()
            dets = {"layer": layer_dets(pred, sel, scale),
                    "mine": postprocess(pred.clone(), scale, args.conf, args.iou)}
            agree += same_set(dets["layer"], dets["mine"])
            count_diff.append(len(dets["layer"]) - len(dets["mine"]))
            n_layer += len(dets["layer"])
            n_mine += len(dets["mine"])
            for k, d in dets.items():
                for x1, y1, x2, y2, s, c in d:
                    results[k].append({"image_id": img_id, "category_id": cats[int(c)],
                                       "bbox": [float(x1), float(y1), float(x2 - x1),
                                                float(y2 - y1)], "score": float(s)})

    stats = {k: coco.evaluate(gt, v, image_ids, None) for k, v in results.items()}
    cd = np.array(count_diff)
    report = {
        "engine": args.engine.name,
        "conf_thresh": args.conf,
        "nms_iou": args.iou,
        "images": len(image_ids),
        "images_identical": agree,
        "boxes_layer": n_layer,
        "boxes_mine": n_mine,
        "images_with_more_boxes_from_layer": int((cd > 0).sum()),
        "images_with_fewer_boxes_from_layer": int((cd < 0).sum()),
        "ap_layer": stats["layer"]["AP"],
        "ap_mine": stats["mine"]["AP"],
        "ap50_layer": stats["layer"]["AP50"],
        "ap50_mine": stats["mine"]["AP50"],
    }
    pct = lambda v, q: float(np.percentile(v, q))  # noqa: E731
    skip = 20  # warm-up images
    report["host_ms_engine_with_nms"] = {"mean": float(np.mean(t_nms[skip:])),
                                          "p50": pct(t_nms[skip:], 50), "p99": pct(t_nms[skip:], 99)}
    if t_plain:
        report["host_ms_engine_without_nms"] = {"mean": float(np.mean(t_plain[skip:])),
                                                 "p50": pct(t_plain[skip:], 50),
                                                 "p99": pct(t_plain[skip:], 99)}
        report["note"] = ("host wall time per image (run + sync) from Python, same harness for "
                          "both engines; the difference is what the NMS layer adds")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"nms_layer_compare_{args.engine.stem}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
