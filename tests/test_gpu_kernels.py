"""The CUDA kernels against the Python reference, through libvrt_gpu_capi.so (ctypes).

Needs a GPU build (cmake --preset release); skipped otherwise, so CI skips it.
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
from vrt.yuv import nv12_to_bgr


def _find_lib() -> Path | None:
    candidates = [Path(os.environ["VRT_GPU_CAPI"])] if "VRT_GPU_CAPI" in os.environ else []
    candidates += sorted((REPO_ROOT / "build").glob("*/libvrt_gpu_capi.so"))
    return next((p for p in candidates if p.exists()), None)


LIB_PATH = _find_lib()
CPU_LIB_PATH = next(iter(sorted((REPO_ROOT / "build").glob("*/libvrt_capi.so"))), None)
pytestmark = pytest.mark.skipif(
    LIB_PATH is None or not torch.cuda.is_available(), reason="GPU build or GPU not available"
)

U8P = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
F32P = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")


@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.vrt_gpu_last_error.restype = ctypes.c_char_p
    lib.vrt_gpu_nv12_letterbox.argtypes = [U8P] + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    lib.vrt_gpu_nv12_thumbnail.argtypes = [U8P] + [ctypes.c_int] * 6 + [U8P]
    lib.vrt_gpu_postprocess.argtypes = [F32P, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                        ctypes.c_float, ctypes.c_float, ctypes.c_int, F32P]
    lib.vrt_gpu_postprocess_batched.argtypes = [
        F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_int, F32P, np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")]
    return lib


@pytest.fixture(scope="module")
def cpu_lib():
    if CPU_LIB_PATH is None:
        pytest.skip("libvrt_capi.so not built")
    lib = ctypes.CDLL(str(CPU_LIB_PATH))
    lib.vrt_letterbox.argtypes = [U8P] + [ctypes.c_int] * 5 + [F32P]
    return lib


def synthetic_nv12(w: int, h: int, seed: int) -> np.ndarray:
    """Smooth gradients plus noise, covering the full 0-255 range in Y, U and V."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    y = 128 + 120 * np.sin(xx / 41.0) * np.cos(yy / 29.0) + rng.normal(0, 12, (h, w))
    cy, cx = np.mgrid[0 : h // 2, 0 : w // 2]
    u = 128 + 110 * np.sin(cx / 17.0 + cy / 23.0) + rng.normal(0, 8, (h // 2, w // 2))
    v = 128 + 110 * np.cos(cx / 19.0 - cy / 13.0) + rng.normal(0, 8, (h // 2, w // 2))
    uv = np.stack([u, v], axis=-1)
    to_u8 = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)  # noqa: E731
    return np.concatenate([to_u8(y).ravel(), to_u8(uv).ravel()])


def gpu_letterbox(lib, nv12, w, h, bt709, full_range, fp16):
    out = np.empty((3, 640, 640), dtype=np.float16 if fp16 else np.float32)
    rc = lib.vrt_gpu_nv12_letterbox(nv12, w, h, int(bt709), int(full_range), 640, 640, int(fp16),
                                    out.ctypes.data)
    assert rc == 0, lib.vrt_gpu_last_error().decode()
    return out.astype(np.float32)


@pytest.mark.parametrize("w,h", [(1920, 1080), (1280, 720), (640, 480), (854, 480), (1000, 562)])
@pytest.mark.parametrize("bt709,full_range", [(True, False), (False, True)])
def test_letterbox_matches_cpu_exactly(lib, cpu_lib, w, h, bt709, full_range):
    # Same taps and the same float maths as the C++ letterbox: must be bit-identical.
    nv12 = synthetic_nv12(w, h, seed=w + h)
    bgr = nv12_to_bgr(nv12, w, h, "bt709" if bt709 else "bt601", full_range)
    cpu = np.empty((3, 640, 640), dtype=np.float32)
    cpu_lib.vrt_letterbox(bgr, w, h, w * 3, 640, 640, cpu)
    np.testing.assert_array_equal(gpu_letterbox(lib, nv12, w, h, bt709, full_range, False), cpu)


@pytest.mark.parametrize("w,h", [(1920, 1080), (854, 480)])
def test_letterbox_within_one_of_cv2(lib, w, h):
    # cv2.resize uses fixed-point weights for uint8: identical for integer scale ratios
    # (1080p -> 640), off by one on a few percent of values otherwise.
    nv12 = synthetic_nv12(w, h, seed=w + h)
    ref, _ = letterbox(nv12_to_bgr(nv12, w, h), (640, 640))
    diff = np.abs(gpu_letterbox(lib, nv12, w, h, True, False, False) - ref)
    assert diff.max() <= 1
    assert (diff > 0).mean() < 0.06
    if w == 1920:
        assert diff.max() == 0


def test_letterbox_fp16_equals_fp32(lib):
    nv12 = synthetic_nv12(1920, 1080, seed=7)
    a = gpu_letterbox(lib, nv12, 1920, 1080, True, False, fp16=True)
    b = gpu_letterbox(lib, nv12, 1920, 1080, True, False, fp16=False)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("w,h,dw,dh", [(1920, 1080, 640, 360), (1280, 720, 640, 360),
                                       (854, 480, 300, 170)])
@pytest.mark.parametrize("bt709,full_range", [(True, False), (False, True)])
def test_thumbnail_is_box_mean_of_converted_pixels(lib, w, h, dw, dh, bt709, full_range):
    # The live-demo thumbnail: each output pixel is the rounded mean of the BGR pixels in its
    # integer box [x*w//dw, (x+1)*w//dw). Box sums are exact in float and a mean is never
    # within float error of a rounding boundary unless it is on one, so this is bit-exact.
    nv12 = synthetic_nv12(w, h, seed=w + dw)
    bgr = nv12_to_bgr(nv12, w, h, "bt709" if bt709 else "bt601", full_range).astype(np.int64)
    x0 = np.arange(dw) * w // dw  # downscaling: the boxes tile the frame
    y0 = np.arange(dh) * h // dh
    sums = np.add.reduceat(np.add.reduceat(bgr, x0, axis=1), y0, axis=0)
    counts = np.outer(np.diff(np.append(y0, h)), np.diff(np.append(x0, w)))[..., None]
    ref = np.clip(np.floor(sums / counts + 0.5), 0, 255).astype(np.uint8)
    out = np.empty((dh, dw, 3), dtype=np.uint8)
    rc = lib.vrt_gpu_nv12_thumbnail(nv12, w, h, int(bt709), int(full_range), dw, dh, out)
    assert rc == 0, lib.vrt_gpu_last_error().decode()
    np.testing.assert_array_equal(out, ref)


def clustered_preds(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    preds = np.zeros((n, 85), dtype=np.float32)
    centers = rng.uniform(50, 590, size=(60, 2))
    idx = rng.integers(0, 60, size=n)
    preds[:, 0:2] = centers[idx] + rng.normal(0, 6, size=(n, 2))
    preds[:, 2:4] = rng.uniform(20, 120, size=(n, 2))
    preds[:, 4] = rng.uniform(0, 1, size=n)
    preds[:, 5:] = rng.uniform(0, 0.2, size=(n, 80))
    preds[np.arange(n), 5 + idx % 7] = rng.uniform(0.2, 1, size=n)  # few classes: more overlap
    return preds


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("conf,iou", [(0.3, 0.45), (0.01, 0.65)])
def test_postprocess_matches_reference(lib, seed, conf, iou):
    preds = clustered_preds(8400, seed)
    scale = 1 / 3
    ref = postprocess(torch.from_numpy(preds.copy()), scale, conf, iou)
    max_det = 8400
    out = np.empty((max_det, 6), dtype=np.float32)
    count = lib.vrt_gpu_postprocess(preds, 8400, 80, scale, conf, iou, max_det, out)
    assert count >= 0, lib.vrt_gpu_last_error().decode()
    ours = out[:count]
    assert count == len(ref)
    np.testing.assert_array_equal(ours[:, 5], ref[:, 5])
    np.testing.assert_array_equal(ours[:, 4], ref[:, 4])
    np.testing.assert_allclose(ours[:, :4], ref[:, :4], rtol=0, atol=1e-3)


def test_postprocess_respects_max_det(lib):
    preds = clustered_preds(8400, 0)
    out = np.empty((5, 6), dtype=np.float32)
    count = lib.vrt_gpu_postprocess(preds, 8400, 80, 1.0, 0.01, 0.65, 5, out)
    ref = postprocess(torch.from_numpy(preds.copy()), 1.0, 0.01, 0.65)
    assert count == 5
    np.testing.assert_array_equal(out[:, 4], ref[:5, 4])


def test_postprocess_no_candidates(lib):
    preds = np.zeros((8400, 85), dtype=np.float32)
    out = np.empty((10, 6), dtype=np.float32)
    assert lib.vrt_gpu_postprocess(preds, 8400, 80, 1.0, 0.3, 0.45, 10, out) == 0


def gpu_postprocess(lib, preds, scale, conf, iou, max_det):
    out = np.empty((max_det, 6), dtype=np.float32)
    count = lib.vrt_gpu_postprocess(preds, preds.shape[0], 80, scale, conf, iou, max_det, out)
    assert count >= 0, lib.vrt_gpu_last_error().decode()
    return out[:count]


def gpu_postprocess_batched(lib, batch, scale, conf, iou, max_det):
    b, n, _ = batch.shape
    out = np.empty((b, max_det, 6), dtype=np.float32)
    counts = np.empty(b, dtype=np.int32)
    rc = lib.vrt_gpu_postprocess_batched(np.ascontiguousarray(batch), b, n, 80, scale, conf, iou,
                                         max_det, out, counts)
    assert rc == 0, lib.vrt_gpu_last_error().decode()
    return [out[i, : counts[i]] for i in range(b)]


def mixed_batch(size: int, seed: int) -> np.ndarray:
    """Images that differ in candidate count: clustered scenes, one with no candidates, one with
    scores rounded to 2 decimals (many exact ties, so the anchor tiebreak matters), and a
    repeat of image 0 (identical boxes in two images must not suppress each other)."""
    images = []
    for i in range(size):
        if i % 6 == 3:
            images.append(np.zeros((8400, 85), dtype=np.float32))
        elif i % 6 == 4:
            p = clustered_preds(8400, seed * 100 + i)
            p[:, 4:] = np.round(p[:, 4:], 2)
            images.append(p)
        elif i % 6 == 5:
            images.append(images[0].copy())
        else:
            images.append(clustered_preds(8400, seed * 100 + i))
    return np.stack(images)


@pytest.mark.parametrize("size", [1, 2, 5, 8, 16])
@pytest.mark.parametrize("conf,iou,max_det", [(0.3, 0.45, 100), (0.01, 0.65, 8400), (0.01, 0.65, 7)])
def test_batched_postprocess_bit_identical_to_per_image(lib, size, conf, iou, max_det):
    # One sort over the whole batch, tiles strided over a fixed grid, one walk block per image:
    # every image's detections must be exactly the single-image kernels' (all six columns).
    batch = mixed_batch(size, seed=size)
    scale = 1 / 3
    ours = gpu_postprocess_batched(lib, batch, scale, conf, iou, max_det)
    for i in range(size):
        ref = gpu_postprocess(lib, batch[i], scale, conf, iou, max_det)
        assert len(ours[i]) == len(ref), f"image {i}"
        np.testing.assert_array_equal(ours[i], ref, err_msg=f"image {i}")
    if size > 3:
        assert len(ours[3]) == 0  # the all-zero image


def test_batched_postprocess_matches_reference(lib):
    batch = mixed_batch(6, seed=11)
    ours = gpu_postprocess_batched(lib, batch, 0.5, 0.3, 0.45, 8400)
    for i in range(6):
        if i == 4:
            continue  # exact score ties: torchvision's batched_nms breaks them its own way
        ref = postprocess(torch.from_numpy(batch[i].copy()), 0.5, 0.3, 0.45)
        assert len(ours[i]) == len(ref)
        np.testing.assert_array_equal(ours[i][:, 4:], ref[:, 4:])
        np.testing.assert_allclose(ours[i][:, :4], ref[:, :4], rtol=0, atol=1e-3)
