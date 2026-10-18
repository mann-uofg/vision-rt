#!/usr/bin/env python3
"""Append ONNX NonMaxSuppression to an exported YOLOX graph, so TensorRT does NMS itself.

TensorRT 11 dropped the EfficientNMS plugin; ONNX NonMaxSuppression now maps to the built-in
NMS layer (INetworkDefinition::addNMS). This builds the engine variant used to compare that
layer against my GPU NMS kernels.

Semantics match vrt::postprocess: each anchor competes only in its best class, with score =
objectness * best class score. ONNX NMS scores every (class, box) pair, so the score matrix
is zero except at each anchor's best class, and the score threshold drops the zeros.

Outputs: "preds" (unchanged, [1, 8400, 85]) and "selected" ([K, 3] int64: batch, class, anchor),
K data-dependent.

    python tools/add_nms.py models/yolox_s_fp16.onnx --conf 0.3 --iou 0.45 --max-per-class 100
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def const(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    model = onnx.load(str(args.onnx))
    g = model.graph
    preds = g.output[0].name
    g.initializer.extend([
        const("nms_s0", [0], np.int64), const("nms_s4", [4], np.int64),
        const("nms_s5", [5], np.int64), const("nms_s85", [85], np.int64),
        const("nms_axis2", [2], np.int64), const("nms_half", [0.5], np.float32),
        const("nms_zero", [0.0], np.float32),
        const("nms_max", [args.max_per_class], np.int64),
        const("nms_iou", [args.iou], np.float32), const("nms_conf", [args.conf], np.float32),
    ])
    n = helper.make_node
    g.node.extend([
        # boxes: cx, cy, w, h -> x1, y1, x2, y2 (same arithmetic as the CPU path)
        n("Slice", [preds, "nms_s0", "nms_s4", "nms_axis2"], ["nms_cxcywh"]),
        n("Split", ["nms_cxcywh"], ["nms_cxcy", "nms_wh"], axis=2, num_outputs=2),
        n("Mul", ["nms_wh", "nms_half"], ["nms_halfwh"]),
        n("Sub", ["nms_cxcy", "nms_halfwh"], ["nms_x1y1"]),
        n("Add", ["nms_cxcy", "nms_halfwh"], ["nms_x2y2"]),
        n("Concat", ["nms_x1y1", "nms_x2y2"], ["nms_boxes"], axis=2),
        # scores: objectness * class score, kept only at each anchor's best class
        n("Slice", [preds, "nms_s4", "nms_s5", "nms_axis2"], ["nms_obj"]),
        n("Slice", [preds, "nms_s5", "nms_s85", "nms_axis2"], ["nms_cls"]),
        n("ReduceMax", ["nms_cls", "nms_axis2"], ["nms_best"], keepdims=1),
        n("Equal", ["nms_cls", "nms_best"], ["nms_is_best"]),
        n("Mul", ["nms_cls", "nms_obj"], ["nms_all_scores"]),
        n("Where", ["nms_is_best", "nms_all_scores", "nms_zero"], ["nms_scores_nc"]),
        n("Transpose", ["nms_scores_nc"], ["nms_scores"], perm=[0, 2, 1]),  # [1, 80, 8400]
        n("NonMaxSuppression", ["nms_boxes", "nms_scores", "nms_max", "nms_iou", "nms_conf"],
          ["selected"], center_point_box=0),
    ])
    g.output.append(helper.make_tensor_value_info("selected", TensorProto.INT64, ["K", 3]))

    onnx.checker.check_model(model)
    tag = f"nms_c{args.conf:g}_i{args.iou:g}"
    out = args.out or args.onnx.with_name(f"{args.onnx.stem}_{tag}.onnx")
    onnx.save(model, str(out))
    print(out)


if __name__ == "__main__":
    main()
