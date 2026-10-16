#!/usr/bin/env bash
# Download COCO 2017 val images and annotations into data/coco.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/data/coco}"
PY="${PYTHON:-python3}"
mkdir -p "$DEST"
cd "$DEST"

for z in zips/val2017.zip annotations/annotations_trainval2017.zip; do
  f=$(basename "$z")
  if [ ! -f "$f" ]; then
    curl -fL --retry 3 -o "$f.part" "http://images.cocodataset.org/$z"
    mv "$f.part" "$f"
  fi
done

# unzip isn't always installed; Python's zipfile always is.
[ -d val2017 ] || "$PY" -c "import zipfile; zipfile.ZipFile('val2017.zip').extractall('.')"
[ -f annotations/instances_val2017.json ] ||
  "$PY" -c "import zipfile; zipfile.ZipFile('annotations_trainval2017.zip').extractall('.')"

echo "$(find val2017 -name '*.jpg' | wc -l) images in $DEST/val2017"
