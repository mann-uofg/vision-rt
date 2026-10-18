#!/usr/bin/env bash
# Download COCO 2017 val images and annotations into data/coco.
#
# images.cocodataset.org only works over plain HTTP (its HTTPS certificate doesn't match the
# host name), so both archives are checked against pinned hashes before anything is extracted.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/coco}"
PY="${PYTHON:-python3}"
mkdir -p "$DEST"
cd "$DEST"

declare -A SHA256=(
  [val2017.zip]=4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05
  [annotations_trainval2017.zip]=113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268
)

for z in zips/val2017.zip annotations/annotations_trainval2017.zip; do
  f=$(basename "$z")
  if [ ! -f "$f" ]; then
    curl -fL --retry 3 -o "$f.part" "http://images.cocodataset.org/$z"
    mv "$f.part" "$f"
  fi
  echo "${SHA256[$f]}  $f" | sha256sum -c -
done

# unzip isn't always installed; Python's zipfile always is.
[ -d val2017 ] || "$PY" -c "import zipfile; zipfile.ZipFile('val2017.zip').extractall('.')"
[ -f annotations/instances_val2017.json ] ||
  "$PY" -c "import zipfile; zipfile.ZipFile('annotations_trainval2017.zip').extractall('.')"

echo "$(find val2017 -name '*.jpg' | wc -l) images in $DEST/val2017"
