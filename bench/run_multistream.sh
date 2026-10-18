#!/usr/bin/env bash
# Multi-stream scaling: N camera streams on one GPU, each with its own decoder, CUDA stream and
# TensorRT execution context.
#   capacity:  every stream as fast as it can go, N = 1..32 -> aggregate fps and p99 vs N
#   real time: every stream paced at 30 fps like a camera -> how many cameras keep up. These
#              runs use sync mode: with frames arriving every 33 ms, pipelined mode would hold
#              each frame's results until the next frame arrives, adding a frame of latency.
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINES="${ENGINES:-yolox_s_fp16}"
CAPACITY_N="${CAPACITY_N-1 2 3 4 6 8 12 16 24 32}"
REALTIME_N="${REALTIME_N-8 16 24 32 40 48 56}"
VIDEOS=$(printf "%s," data/footage/barn*.mp4)
VIDEOS=${VIDEOS%,}
R=bench/results

run() {  # one run, killed if it hangs, so the sweep always finishes
  timeout 900 build/release/vrt_multistream --engine "models/$1.engine" --videos "$VIDEOS" \
    --streams "$2" "${@:3}"
}

for e in $ENGINES; do
  for n in $CAPACITY_N; do
    run "$e" "$n" --frames 1000 --warmup 100 --json "$R/multistream_${e}_fps0_n${n}.json" ||
      echo "capacity n=$n failed"
  done
  for n in $REALTIME_N; do
    run "$e" "$n" --fps 30 --mode sync --frames 600 --warmup 60 \
      --json "$R/multistream_${e}_fps30_n${n}.json" || echo "real-time n=$n failed"
  done
done
