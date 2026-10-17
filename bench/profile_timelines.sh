#!/usr/bin/env bash
# Nsight Systems traces of the M1 and M2 pipelines, drawn as the README timelines.
# Traces (large) go to $PROFILES, outside the repo; only the PNGs land in docs/.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILES="${PROFILES:-$HOME/vrt-assets/profiles}"
VIDEO="${VIDEO:-data/footage/barn022.mp4}"
ENGINE="${ENGINE:-models/yolox_s_fp16.engine}"
mkdir -p "$PROFILES"

trace() {  # trace <name> <command...>
  local name=$1
  shift
  nsys profile -t cuda,nvtx --cuda-graph-trace=node --force-overwrite true \
    -o "$PROFILES/$name" "$@" > /dev/null
  nsys export --type sqlite --force-overwrite true -o "$PROFILES/$name.sqlite" \
    "$PROFILES/$name.nsys-rep" > /dev/null
}

trace m1 build/release/vrt_detect --engine "$ENGINE" --video "$VIDEO" --frames 300
trace m2 build/release/vrt_pipeline --engine "$ENGINE" --video "$VIDEO" --frames 300

python tools/plot_timeline.py "$PROFILES/m1.sqlite" docs/timeline_m1.png --frames 4 \
  --title "M1: CPU decode, CPU letterbox, TensorRT via enqueueV3, CPU NMS (YOLOX-S FP16, 4 frames)"
python tools/plot_timeline.py "$PROFILES/m2.sqlite" docs/timeline_m2.png --frames 4 \
  --title "M2: NVDEC, fused GPU letterbox, TensorRT + NMS as CUDA graphs, pipelined (YOLOX-S FP16, 4 frames)"
