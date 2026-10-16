#!/usr/bin/env bash
# Download the test footage and transcode it to the format the benchmarks use.
#
# Source: USDA "Dairy Cows (20220209-FPAC-RockHillDairy0NN)" clips on Wikimedia Commons,
# public domain (work of the U.S. federal government). Originals are 1080p VP9 WebM.
#
# Output: data/footage/barnNNN.mp4 — H.264 High, 1920x1080, yuv420p, original frame rate,
# GOP of 30 frames and no B-frames, which is what a typical IP camera stream looks like.
# The originals are checked against pinned hashes; the transcodes depend on the ffmpeg build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/footage}"
FFMPEG="${FFMPEG:-ffmpeg}"
BASE=https://upload.wikimedia.org/wikipedia/commons
mkdir -p "$DEST/src"

# clip id -> Commons hash directory
declare -A DIR=(
  [004]=b/b1 [007]=2/2d [012]=0/0a [019]=3/38 [022]=0/06 [023]=3/3b
)
declare -A SHA256=(
  [004]=61dc64d9551347ebbaf38f2668a700cf3626c333ad61a3e2b0f224d253cb5497
  [007]=e9558345ed98b724578ca780f472981af2878876946941053a22c71d3804e526
  [012]=d4a25cb03f79647f0e1cb7f41898d2ef8810e0d1a974e838912b57f3d50b55bf
  [019]=f3c8e562e0c8016fbe46bb16d46dd3815140e964d969ed68647ecf2bba6db5ee
  [022]=bf77aae20154b20639ab9f9505130c635ebc32389d8604c16bac0d9199aff8c1
  [023]=0520ec29cbd728003c84d62a386447dccea8779ac315f8e68a9bf144a6697a3b
)

for id in $(printf '%s\n' "${!DIR[@]}" | sort); do
  src="$DEST/src/RockHillDairy$id.webm"
  out="$DEST/barn$id.mp4"
  if [ ! -f "$src" ]; then
    curl -fsSL --retry 3 -A "vision-rt/0.1 (footage fetch)" -o "$src.part" \
      "$BASE/${DIR[$id]}/Dairy_Cows_%2820220209-FPAC-RockHillDairy$id%29.webm"
    mv "$src.part" "$src"
  fi
  echo "${SHA256[$id]}  $src" | sha256sum -c --quiet -
  if [ ! -f "$out" ]; then
    "$FFMPEG" -hide_banner -loglevel error -i "$src" -an \
      -c:v libx264 -profile:v high -preset medium -crf 20 -pix_fmt yuv420p \
      -g 30 -keyint_min 30 -sc_threshold 0 -bf 0 \
      "$out"
  fi
  echo "$out"
done
