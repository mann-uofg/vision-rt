#!/usr/bin/env bash
# M1 C++ pipeline (CPU pre/post around TensorRT) for every engine on the benchmark clip,
# then a per-frame detection comparison against the naive baseline.
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${VIDEO:-data/footage/barn022.mp4}"
clip=$(basename "$VIDEO" .mp4)
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$busy" -gt 10 ] && [ -z "${ALLOW_BUSY_GPU:-}" ]; then
  echo "GPU is ${busy}% busy; close other GPU work or set ALLOW_BUSY_GPU=1" >&2
  exit 1
fi

for size in ${SIZES:-s m}; do
  for prec in ${PRECISIONS:-fp32 tf32 fp16}; do
    name="yolox_${size}_${prec}"
    build/release/vrt_detect --engine "models/$name.engine" --video "$VIDEO" \
      --json "bench/results/cpp_sync_${name}_${clip}.json" \
      --dets "bench/results/detections/cpp_sync_${name}_${clip}.csv"
    python tools/compare_detections.py \
      --ref "bench/results/detections/baseline_yolox_${size}_fp32-tf32_${clip}.npz" \
      --test "bench/results/detections/cpp_sync_${name}_${clip}.csv" \
      --out "bench/results/compare_cpp_sync_${name}_${clip}.json" |
      grep -E '"(recall|precision)_vs_ref"|matched_iou_mean'
  done
done
