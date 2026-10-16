#!/usr/bin/env bash
# Run the naive baseline for both model sizes, one after the other, on an idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${VIDEO:-data/footage/barn022.mp4}"
busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$busy" -gt 10 ] && [ -z "${ALLOW_BUSY_GPU:-}" ]; then
  echo "GPU is ${busy}% busy; close other GPU work or set ALLOW_BUSY_GPU=1" >&2
  exit 1
fi

for size in s m; do
  python tools/baseline.py --video "$VIDEO" --size "$size"
done
