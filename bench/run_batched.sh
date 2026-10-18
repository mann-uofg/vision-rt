#!/usr/bin/env bash
# Cross-camera batching sweep (apps/vrt_batched.cpp), YOLOX-S FP16 engines of batch 1-16.
#   capacity:  N cameras unthrottled, for each max batch size
#   real time: N cameras at 30 fps (staggered), for the larger max batch sizes
# Compare with bench/run_multistream.sh, where every camera has its own batch-1 context.
set -euo pipefail
cd "$(dirname "$0")/.."

E=models/yolox_s_fp16.engine
for b in 2 4 8 16; do E="$E,models/yolox_s_fp16_b$b.engine"; done
VIDEOS=$(printf "%s," data/footage/barn*.mp4)
VIDEOS=${VIDEOS%,}
R=bench/results
CAPACITY_N="${CAPACITY_N-16 32}"
CAPACITY_B="${CAPACITY_B-1 2 4 8 16}"
REALTIME_N="${REALTIME_N-32 40 48 56 64 72}"
REALTIME_B="${REALTIME_B-8 16}"

run() {  # run <streams> <max batch> <args...>; killed if it hangs
  timeout 900 build/release/vrt_batched --engines "$E" --videos "$VIDEOS" --streams "$1" \
    --max-batch "$2" "${@:3}"
}

for n in $CAPACITY_N; do
  for b in $CAPACITY_B; do
    run "$n" "$b" --frames 1000 --warmup 100 --json "$R/batched_fps0_n${n}_b${b}.json" ||
      echo "capacity n=$n b=$b failed"
  done
done
for b in $REALTIME_B; do
  for n in $REALTIME_N; do
    run "$n" "$b" --fps 30 --frames 600 --warmup 60 --json "$R/batched_fps30_n${n}_b${b}.json" ||
      echo "real-time n=$n b=$b failed"
  done
done

# Correctness: one camera per clip, batches of up to 4, and the barn022 camera's detections
# compared with the unbatched GPU path's (bench/run_pipelines.sh), which skips the warmup frames.
D=$R/detections
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
clips=(data/footage/barn*.mp4)
run "${#clips[@]}" 4 --frames 1000 --warmup 100 --dets-dir "$tmp" >/dev/null
cam=0
for v in "${clips[@]}"; do
  [ "$(basename "$v")" = barn022.mp4 ] && break
  cam=$((cam + 1))
done
python tools/compare_detections.py --ref "$D/gpu_sync_yolox_s_fp16_barn022.csv" \
  --test "$tmp/cam$cam.csv" --min-frame 100 --max-frame 1100 \
  --out "$R/compare_batched_b4_barn022.json"
