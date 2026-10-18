#!/usr/bin/env bash
# The demo video: run the GPU pipeline on each demo clip, then draw its detections.
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINE="${ENGINE:-models/yolox_s_fp16.engine}"
D=bench/results/detections
mkdir -p "$D"
for clip in barn022 barn012 barn023 barn004; do
  build/release/vrt_pipeline --engine "$ENGINE" --video "data/footage/$clip.mp4" --mode sync \
    --warmup 0 --dets "$D/demo_$clip.csv" | tail -1
done
python tools/render_demo.py --seconds 30 --out docs/demo.mp4 --gif docs/demo.gif
