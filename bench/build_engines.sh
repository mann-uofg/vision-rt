#!/usr/bin/env bash
# Export ONNX graphs and build TensorRT engines for every model size and precision.
#
#   fp32  FP32 graph, TF32 disabled (true FP32)
#   tf32  FP32 graph, TF32 allowed (TensorRT's default for FP32 layers)
#   fp16  FP16 graph (TensorRT 11 is strongly typed, so precision comes from the graph)
#   int8     INT8 Q/DQ graph from Model Optimizer, entropy calibration (tools/quantize_int8.py)
#   int8max  same, max calibration
#   int8fp32 INT8 with FP32 (not FP16) for the non-quantized layers (see INTERVIEW H2)
#
# Engines go to models/ (gitignored). Each build also writes the engine's layer list to
# bench/results/engines/ and keeps trtexec's model-only timing summary next to it.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES="${SIZES:-s m}"
PRECISIONS="${PRECISIONS:-fp32 tf32 fp16}"
OUT=bench/results/engines
mkdir -p "$OUT"

for size in $SIZES; do
  [ -f "models/yolox_${size}_fp32.onnx" ] || python tools/export_onnx.py --size "$size"
  [ -f "models/yolox_${size}_fp16.onnx" ] || python tools/export_onnx.py --size "$size" --fp16
  for prec in $PRECISIONS; do
    name="yolox_${size}_${prec}"
    case $prec in
      fp32) onnx=models/yolox_${size}_fp32.onnx; extra=(--noTF32) ;;
      tf32) onnx=models/yolox_${size}_fp32.onnx; extra=() ;;
      fp16) onnx=models/yolox_${size}_fp16.onnx; extra=() ;;
      int8 | int8max)
        onnx=models/yolox_${size}_${prec}.onnx; extra=()
        cal=entropy; [ "$prec" = int8max ] && cal=max
        [ -f "$onnx" ] || python tools/quantize_int8.py --size "$size" --calibration "$cal" ;;
      int8fp32)
        onnx=models/yolox_${size}_int8fp32.onnx; extra=()
        [ -f "$onnx" ] || python tools/quantize_int8.py --size "$size" --high-precision fp32 --tag int8fp32 ;;
      *) echo "unknown precision $prec" >&2; exit 1 ;;
    esac
    trtexec --onnx="$onnx" --saveEngine="models/$name.engine" "${extra[@]}" \
      --profilingVerbosity=detailed --timingCacheFile=models/timing.cache \
      --exportLayerInfo="$OUT/$name.layers.json" --duration=10 \
      > "models/$name.build.log" 2>&1 || { tail -20 "models/$name.build.log"; exit 1; }
    # Model-only numbers from trtexec (fixed input, no pre/post, no copies). Not end-to-end.
    grep -E "TensorRT version|Selected Device|Compute Capability|Throughput:|GPU Compute Time:|Latency:|Engine built in|Total Activation Memory" \
      "models/$name.build.log" | sed 's/^\[[^]]*\] *\[I\] *//' > "$OUT/$name.trtexec.txt"
    echo "== $name"; grep -E "Throughput|GPU Compute Time" "$OUT/$name.trtexec.txt"
  done
done
