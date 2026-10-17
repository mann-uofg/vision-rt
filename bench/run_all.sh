#!/usr/bin/env bash
# Every benchmark in one go, on an idle GPU: decode, the naive baseline, then all C++
# pipelines. Takes a few minutes. Rebuild the README tables afterwards with
# python bench/make_tables.py.
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${VIDEO:-data/footage/barn022.mp4}"
clip=$(basename "$VIDEO" .mp4)
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$busy" -gt 10 ] && [ -z "${ALLOW_BUSY_GPU:-}" ]; then
  echo "GPU is ${busy}% busy; close other GPU work or set ALLOW_BUSY_GPU=1" >&2
  exit 1
fi

for dec in cpu nvdec; do
  build/release/vrt_decode_bench --video "$VIDEO" --decoder "$dec" \
    --json "bench/results/decode_${dec}_${clip}.json"
done
ALLOW_BUSY_GPU=1 bench/run_baseline.sh
bench/run_pipelines.sh
