"""COCO val2017 mAP with pycocotools."""

import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .yolox import REPO_ROOT

COCO_DIR = REPO_ROOT / "data" / "coco"

# COCOeval.stats order.
STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def load_val(coco_dir: Path = COCO_DIR) -> COCO:
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(coco_dir / "annotations" / "instances_val2017.json"))


def class_to_category(gt: COCO) -> list[int]:
    """Model class index 0..79 -> COCO category id (ids have gaps, 1..90)."""
    return sorted(gt.getCatIds())


def evaluate(gt: COCO, results: list[dict], image_ids: list[int], out_json: Path | None) -> dict:
    """Run bbox COCOeval and return {stat name: value}. Prints the standard summary table."""
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results))
    if not results:
        return dict.fromkeys(STAT_NAMES, 0.0)
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = image_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return dict(zip(STAT_NAMES, (float(s) for s in ev.stats)))


# COCO class names in the model's class-index order (0..79).
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
COW = CLASS_NAMES.index("cow")
