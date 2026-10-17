"""The C++ CPU core against the Python reference, through the C API (ctypes).

Skipped when the C++ library hasn't been built (cmake --preset release, or ci in CI).
"""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vrt.postprocess import postprocess
from vrt.preprocess import letterbox
from vrt.yolox import REPO_ROOT


def _find_lib() -> Path | None:
    candidates = [Path(os.environ["VRT_CAPI"])] if "VRT_CAPI" in os.environ else []
    candidates += sorted((REPO_ROOT / "build").glob("*/libvrt_capi.so"))
    return next((p for p in candidates if p.exists()), None)


LIB_PATH = _find_lib()
pytestmark = pytest.mark.skipif(LIB_PATH is None, reason="libvrt_capi.so not built")

F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
U8P = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.vrt_letterbox.argtypes = [U8P] + [ctypes.c_int] * 5 + [F32P]
    lib.vrt_letterbox.restype = None
    lib.vrt_postprocess.argtypes = [F32P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                    ctypes.c_float, ctypes.c_float, F32P, ctypes.c_int]
    lib.vrt_postprocess.restype = ctypes.c_int
    return lib


def cpp_letterbox(lib, img: np.ndarray, hw=(640, 640)) -> np.ndarray:
    out = np.empty((3, *hw), dtype=np.float32)
    h, w = img.shape[:2]
    lib.vrt_letterbox(np.ascontiguousarray(img), w, h, w * 3, hw[1], hw[0], out)
    return out


@pytest.mark.parametrize("hw", [(1080, 1920), (480, 640), (333, 517), (427, 640), (640, 640)])
def test_letterbox_within_one_of_reference(lib, hw):
    # Smooth content (like real images) plus noise. cv2 resizes uint8 with fixed-point
    # arithmetic and the C++ code uses float, so single pixels may differ by 1.
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0 : hw[0], 0 : hw[1]]
    base = 127 + 100 * np.sin(xx / 37.0)[..., None] * np.cos(yy / 23.0)[..., None]
    img = np.clip(base + rng.normal(0, 20, size=(*hw, 3)), 0, 255).astype(np.uint8)

    ours = cpp_letterbox(lib, img)
    ref, _ = letterbox(img, (640, 640))
    diff = np.abs(ours - ref)
    assert diff.max() <= 1
    # cv2 uses fixed-point weights for uint8 resizes. Identical when the scale is a whole ratio
    # or 1 (1080p, 480p and 427x640 here); ~8% of values off by one for 333x517, where the
    # weights are arbitrary fractions.
    assert (diff > 0).mean() < 0.12


def test_letterbox_half_equals_float(lib):
    lib.vrt_letterbox_half.argtypes = [U8P] + [ctypes.c_int] * 5 + [
        np.ctypeslib.ndpointer(np.uint16, flags="C_CONTIGUOUS")
    ]
    img = np.random.default_rng(3).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    half = np.empty((3, 640, 640), dtype=np.uint16)
    lib.vrt_letterbox_half(img, 1920, 1080, 1920 * 3, 640, 640, half)
    np.testing.assert_array_equal(half.view(np.float16).astype(np.float32), cpp_letterbox(lib, img))


def test_letterbox_same_size_is_exact(lib):
    img = np.random.default_rng(2).integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    ref, _ = letterbox(img, (640, 640))
    np.testing.assert_array_equal(cpp_letterbox(lib, img), ref)


@pytest.mark.parametrize("seed", range(3))
def test_postprocess_matches_reference(lib, seed):
    rng = np.random.default_rng(seed)
    n = 8400
    preds = np.zeros((n, 85), dtype=np.float32)
    # Clustered boxes so NMS has real work: 40 objects, ~200 jittered proposals each.
    centers = rng.uniform(50, 590, size=(40, 2))
    idx = rng.integers(0, 40, size=n)
    preds[:, 0:2] = centers[idx] + rng.normal(0, 6, size=(n, 2))
    preds[:, 2:4] = rng.uniform(20, 120, size=(n, 2))
    preds[:, 4] = rng.uniform(0, 1, size=n)
    preds[:, 5:] = rng.uniform(0, 0.2, size=(n, 80))
    preds[np.arange(n), 5 + idx % 80] = rng.uniform(0.3, 1, size=n)
    scale = 1 / 3

    ref = postprocess(torch.from_numpy(preds.copy()), scale, 0.3, 0.45)
    out = np.empty((n, 6), dtype=np.float32)
    count = lib.vrt_postprocess(preds, n, 80, scale, 0.3, 0.45, out, n)
    ours = out[:count]

    assert count == len(ref)
    np.testing.assert_array_equal(ours[:, 5], ref[:, 5])
    np.testing.assert_allclose(ours[:, 4], ref[:, 4], rtol=1e-6)
    np.testing.assert_allclose(ours[:, :4], ref[:, :4], atol=1e-3)
