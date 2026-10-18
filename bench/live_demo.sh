#!/usr/bin/env bash
# The live demo: four barn cameras through the real GPU pipeline (NVDEC -> fused preprocessing
# -> TensorRT FP16 from CUDA graphs -> GPU NMS), paced like 30 fps cameras, in a 2x2 window
# with the boxes, cow counts, and the fps and latency the pipeline measures as it runs.
#
#   bench/live_demo.sh                 # close the window or press q to stop
#   bench/live_demo.sh --frames 900    # extra arguments go to vrt_multistream
#
# Needs a display: WSLg on Windows 11, or any X11/Wayland desktop. The clips loop.
set -euo pipefail
cd "$(dirname "$0")/.."

ENGINE="${ENGINE:-models/yolox_s_fp16.engine}"
CLIPS=(barn022 barn012 barn023 barn004)  # same cameras as the offline demo video

if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  echo "live_demo.sh: no display (DISPLAY / WAYLAND_DISPLAY unset)" >&2
  exit 1
fi
videos=$(printf 'data/footage/%s.mp4,' "${CLIPS[@]}")
names=$(IFS=,; echo "${CLIPS[*]}")

# sync mode: each frame is collected before the next one is read, the lowest-latency mode for
# paced cameras (pipelined mode would hold every frame until the next one arrives, 33 ms later).
build/release/vrt_multistream --engine "$ENGINE" --streams 4 --videos "${videos%,}" \
  --fps 30 --mode sync --show "$@" | python tools/live_view.py --names "$names"
