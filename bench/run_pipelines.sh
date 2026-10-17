#!/usr/bin/env bash
# Every C++ pipeline variant for every engine on the benchmark clip, each followed by a
# per-frame detection comparison against the naive baseline:
#   cpp_sync        M1: CPU decode/letterbox/NMS around TensorRT (enqueueV3), synchronous
#   cpp_sync_graph  M1 + the engine replayed from a CUDA graph
#   gpu_sync        M2: NVDEC + GPU letterbox + TensorRT graph + GPU NMS, one frame at a time
#   gpu_pipelined   M2 with decode of frame n+1 overlapped with the GPU work for frame n
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${VIDEO:-data/footage/barn022.mp4}"
clip=$(basename "$VIDEO" .mp4)
R=bench/results
D=$R/detections

compare() {  # compare <size> <run name>
  python tools/compare_detections.py \
    --ref "$D/baseline_yolox_$1_fp32-tf32_${clip}.npz" --test "$D/$2_${clip}.csv" \
    --out "$R/compare_$2_${clip}.json" | grep -E '"recall_vs_ref"' | sed "s/^/  $2 /"
}

for size in ${SIZES:-s m}; do
  for prec in ${PRECISIONS:-fp32 tf32 fp16}; do
    name="yolox_${size}_${prec}"
    engine="models/$name.engine"
    for variant in cpp_sync cpp_sync_graph; do
      extra=()
      [ "$variant" = cpp_sync_graph ] && extra=(--cuda-graph)
      build/release/vrt_detect --engine "$engine" --video "$VIDEO" "${extra[@]}" \
        --json "$R/${variant}_${name}_${clip}.json" --dets "$D/${variant}_${name}_${clip}.csv" |
        grep throughput | sed "s/^/$variant $name: /"
      compare "$size" "${variant}_${name}"
    done
    for mode in sync pipelined; do
      build/release/vrt_pipeline --engine "$engine" --video "$VIDEO" --mode "$mode" \
        --json "$R/gpu_${mode}_${name}_${clip}.json" --dets "$D/gpu_${mode}_${name}_${clip}.csv" |
        grep throughput | sed "s/^/gpu_$mode $name: /"
      compare "$size" "gpu_${mode}_${name}"
    done
  done
done
