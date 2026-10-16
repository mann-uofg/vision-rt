# shellcheck shell=bash
# Source this from WSL before building or running anything:  source scripts/env.sh
# Paths can be overridden by setting the variables first.

export TRT_ROOT="${TRT_ROOT:-$HOME/opt/TensorRT-11.3.0.99}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VENV="${VENV:-$HOME/venvs/vision-rt}"

export PATH="$TRT_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"

# FFmpeg shared build (BtbN, GPL) with the CUDA/NVDEC decoders, used for footage and decode.
export FFMPEG_ROOT="${FFMPEG_ROOT:-$HOME/opt/ffmpeg-n8.1-latest-linux64-gpl-shared-8.1}"
export PATH="$FFMPEG_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$FFMPEG_ROOT/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
