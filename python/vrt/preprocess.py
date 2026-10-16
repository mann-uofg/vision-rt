"""CPU preprocessing, matching YOLOX's own eval transform (ValTransform, legacy=False).

This is the naive baseline's preprocessing and the reference the GPU kernel gets tested
against later. What YOLOX expects is easy to get wrong:

- channel order stays BGR (the released weights were trained on cv2 images)
- no mean/std normalization: values are raw 0-255 as float32
- letterbox: scale by min(H/h, W/w), resize with bilinear, pad with 114, image at top-left
"""

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(bgr: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    """HWC uint8 BGR image -> (CHW float32 tensor of shape [3, H, W], scale factor)."""
    in_h, in_w = input_hw
    h, w = bgr.shape[:2]
    r = min(in_h / h, in_w / w)
    new_w, new_h = int(w * r), int(h * r)

    padded = np.full((in_h, in_w, 3), PAD_VALUE, dtype=np.uint8)
    padded[:new_h, :new_w] = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return chw, r
