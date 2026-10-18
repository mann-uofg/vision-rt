#!/usr/bin/env python3
"""INT8 post-training quantization with TensorRT Model Optimizer (explicit Q/DQ ONNX).

TensorRT 11 has no INT8 calibrator anymore: INT8 means QuantizeLinear/DequantizeLinear pairs
in the graph, carrying the scales. Model Optimizer calibrates on real images (here 512 COCO
train2017 images, from tools/make_calib.py), inserts the Q/DQ pairs around the convolutions,
and converts everything it didn't quantize to FP16.

One fix-up afterwards: Model Optimizer's FP16 pass also turns the box decode at the end of
the graph (grid offset, exp, x stride) into FP16 and casts the result back to FP32. That
undoes the reason the decode is in FP32 (see export_onnx.py), so this script moves it back:
a Cast to FP32 on the head output, FP32 decode constants, and no trailing cast.

    python tools/quantize_int8.py --size s                       # entropy calibration
    python tools/quantize_int8.py --size s --calibration max     # max (min-max) calibration
    python tools/quantize_int8.py --size m --exclude-head --tag int8nohead
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from vrt import yolox

DECODE_START = "Transpose"  # the head's [1, 85, 8400] -> [1, 8400, 85] permute starts the tail


def restore_fp32_decode(model: onnx.ModelProto) -> int:
    """Run everything after the head's final Transpose in FP32. Returns nodes moved."""
    g = model.graph
    transposes = [n for n in g.node if n.op_type == DECODE_START]
    if len(transposes) != 1:
        raise SystemExit(f"expected one Transpose before the decode, found {len(transposes)}")
    head_out = transposes[0].output[0]

    # Nodes downstream of the transpose.
    tail, frontier = [], {head_out}
    for n in g.node:
        if any(i in frontier for i in n.input):
            tail.append(n)
            frontier.update(n.output)

    # A trailing Cast to FP32 on the output becomes redundant: let its producer write the output.
    out_name = g.output[0].name
    final_casts = [n for n in tail if n.op_type == "Cast" and n.output[0] == out_name]
    for c in final_casts:
        for n in tail:
            for k, o in enumerate(n.output):
                if o == c.input[0]:
                    n.output[k] = out_name
        g.node.remove(c)
        tail.remove(c)

    # Cast the head output to FP32 and feed that to the tail.
    cast_out = head_out + "_fp32"
    for n in tail:
        for k, i in enumerate(n.input):
            if i == head_out:
                n.input[k] = cast_out
    cast = helper.make_node("Cast", [head_out], [cast_out], to=TensorProto.FLOAT,
                            name="decode_cast_to_fp32")
    g.node.insert(list(g.node).index(transposes[0]) + 1, cast)

    # Decode constants back to FP32 (grid offsets and strides are small integers: exact).
    tail_inputs = {i for n in tail for i in n.input}
    for k, init in enumerate(g.initializer):
        if init.name in tail_inputs and init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            g.initializer[k].CopyFrom(numpy_helper.from_array(arr, init.name))

    # Stale FP16 type annotations on tail tensors would contradict the new types.
    stale = {o for n in tail for o in n.output} | {head_out}
    keep = [v for v in g.value_info if v.name not in stale]
    del g.value_info[:]
    g.value_info.extend(keep)
    return len(tail)


def head_output_convs(model: onnx.ModelProto) -> list[str]:
    """The detection head's final convolutions: box regression (4 channels), objectness (1)
    and class scores (80), one of each per scale. Found by their weight shapes."""
    shapes = {i.name: list(i.dims) for i in model.graph.initializer}
    names = []
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and shapes.get(n.input[1], [0])[0] in (1, 4, 80):
            names.append(n.name)
    return names


def main() -> None:
    from modelopt.onnx.quantization import quantize

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--calibration", choices=["entropy", "max"], default="entropy")
    ap.add_argument("--high-precision", choices=["fp16", "fp32"], default="fp16",
                    help="dtype for everything that isn't quantized")
    ap.add_argument("--exclude-head", action="store_true",
                    help="keep the head's 9 output convolutions out of INT8")
    ap.add_argument("--tag", help="output name suffix (default: int8 / int8max)")
    ap.add_argument("--calib-data", type=Path,
                    default=yolox.REPO_ROOT / "data" / "calib" / "coco_train_512.npy")
    args = ap.parse_args()

    models = yolox.REPO_ROOT / "models"
    src = models / f"yolox_{args.size}_fp32.onnx"
    suffix = args.tag or ("int8" if args.calibration == "entropy" else "int8max")
    raw = models / f"yolox_{args.size}_{suffix}_modelopt.onnx"
    out = models / f"yolox_{args.size}_{suffix}.onnx"

    exclude = head_output_convs(onnx.load(str(src))) if args.exclude_head else None
    if exclude:
        print(f"excluding {len(exclude)} head output convs: {exclude}")
    quantize(
        onnx_path=str(src),
        nodes_to_exclude=exclude,
        quantize_mode="int8",
        calibration_method=args.calibration,
        calibration_data=np.load(args.calib_data),
        calibration_eps=["cpu"],
        high_precision_dtype=args.high_precision,
        output_path=str(raw),
    )
    model = onnx.load(str(raw))
    moved = restore_fp32_decode(model) if args.high_precision == "fp16" else 0
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(out))
    qdq = sum(n.op_type == "QuantizeLinear" for n in model.graph.node)
    print(f"{out}: {qdq} QuantizeLinear nodes, {moved} decode nodes back in FP32")


if __name__ == "__main__":
    main()
