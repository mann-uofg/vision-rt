#!/usr/bin/env python3
"""Build the INT8 calibration set: N random COCO train2017 images, preprocessed exactly like
the pipeline's input (letterbox to 640x640, BGR, 0-255, CHW float32), saved as one .npy.

Calibration uses train2017, never val2017, so the mAP measured afterwards on val is fair.
Images are downloaded one by one from the COCO server (no need for the 18 GB zip).

    python tools/make_calib.py --count 512      # -> data/calib/coco_train_512.npy
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from vrt import yolox
from vrt.preprocess import letterbox

DATA = yolox.REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = DATA / "coco" / "annotations" / "instances_train2017.json"
    images = json.loads(ann.read_text())["images"]
    random.Random(args.seed).shuffle(images)
    chosen = images[: args.count]

    img_dir = DATA / "calib" / "train2017"
    img_dir.mkdir(parents=True, exist_ok=True)
    batch = np.empty((len(chosen), 3, *yolox.INPUT_HW), dtype=np.float32)
    for i, info in enumerate(chosen):
        name = info["file_name"]
        if Path(name).name != name or not name.endswith(".jpg"):
            raise SystemExit(f"unexpected file name in the annotations: {name!r}")
        path = img_dir / name
        if not path.exists():
            # Plain HTTP is the only way this server works (see scripts/fetch_coco.sh). These
            # images only feed calibration statistics; a half-written file must not be cached.
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{name}", part)
            part.rename(path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"cannot read {path}")
        batch[i], _ = letterbox(bgr, yolox.INPUT_HW)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(chosen)}")

    out = DATA / "calib" / f"coco_train_{args.count}.npy"
    np.save(out, batch)
    (out.with_suffix(".json")).write_text(
        json.dumps({"seed": args.seed, "images": [c["file_name"] for c in chosen]}, indent=1))
    print(f"{out}: {batch.shape}, {batch.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
