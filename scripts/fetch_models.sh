#!/usr/bin/env bash
# Download the COCO-pretrained YOLOX-S and YOLOX-M checkpoints from the official release
# and check them against pinned hashes.
set -euo pipefail

DEST="${1:-$(dirname "$0")/../models}"
URL=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
mkdir -p "$DEST"

declare -A SHA256=(
  [yolox_s.pth]=f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30
  [yolox_m.pth]=60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6
)

for f in "${!SHA256[@]}"; do
  if [ ! -f "$DEST/$f" ]; then
    curl -fL --retry 3 -o "$DEST/$f.part" "$URL/$f"
    mv "$DEST/$f.part" "$DEST/$f"
  fi
  echo "${SHA256[$f]}  $DEST/$f" | sha256sum -c -
done
