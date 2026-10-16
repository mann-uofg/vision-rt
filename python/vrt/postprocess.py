"""CPU postprocessing: score filter, class-aware NMS, map boxes back to the original frame."""

import numpy as np
import torch
import torchvision

# Thresholds YOLOX uses for COCO evaluation (yolox_base.py: test_conf, nmsthre).
EVAL_CONF = 0.01
EVAL_NMS_IOU = 0.65

# Thresholds for running on video, where only confident boxes are wanted. The score cut is
# YOLOX's demo default; 0.45 IoU is the usual deployment value.
DEPLOY_CONF = 0.3
DEPLOY_NMS_IOU = 0.45


def postprocess(
    pred: torch.Tensor, scale: float, conf_thresh: float, iou_thresh: float
) -> np.ndarray:
    """One image's head output [8400, 85] -> detections [N, 6]: x1, y1, x2, y2, score, class.

    Boxes come back in original-frame pixels. Score is objectness * best class score, the same
    as upstream's postprocess().
    """
    class_score, class_id = pred[:, 5:].max(dim=1)
    score = pred[:, 4] * class_score
    keep = score >= conf_thresh
    boxes, score, class_id = cxcywh_to_xyxy(pred[keep, :4]), score[keep], class_id[keep]

    kept = torchvision.ops.batched_nms(boxes, score, class_id, iou_thresh)
    boxes = boxes[kept] / scale
    return torch.cat(
        [boxes, score[kept, None], class_id[kept, None].to(boxes.dtype)], dim=1
    ).numpy()


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    half = b[:, 2:4] / 2
    return torch.cat([b[:, 0:2] - half, b[:, 0:2] + half], dim=1)


def nms_reference(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Plain greedy NMS in numpy, for testing other implementations against.

    Sort by score, keep the best box, drop everything overlapping it by more than iou_thresh,
    repeat. Returns kept indices in descending score order. Ties keep the lower index.
    """
    order = np.argsort(-scores, kind="stable")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)
