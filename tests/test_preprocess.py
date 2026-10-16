import importlib.util

import numpy as np
import pytest

from vrt import yolox
from vrt.preprocess import PAD_VALUE, letterbox

# Load upstream's data_augment.py on its own: importing the yolox.data package drags in the
# whole training stack (datasets, tqdm, ...).
_spec = importlib.util.spec_from_file_location(
    "yolox_data_augment", yolox.REPO_ROOT / "third_party/YOLOX/yolox/data/data_augment.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
upstream_preproc = _mod.preproc


@pytest.mark.parametrize("hw", [(1080, 1920), (480, 640), (640, 480), (333, 517), (640, 640)])
def test_matches_upstream_exactly(hw):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(*hw, 3), dtype=np.uint8)
    ours, r = letterbox(img, (640, 640))
    theirs, r_up = upstream_preproc(img, (640, 640))
    assert r == r_up
    assert ours.dtype == np.float32 and ours.shape == (3, 640, 640)
    np.testing.assert_array_equal(ours, theirs)


def test_padding_is_bottom_right():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    chw, r = letterbox(img, (640, 640))
    assert r == pytest.approx(1 / 3)
    content_h = int(1080 * r)  # 360
    assert np.all(chw[:, :content_h, :] == 0)
    assert np.all(chw[:, content_h:, :] == PAD_VALUE)


def test_keeps_bgr_order():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 2] = 200  # R
    chw, _ = letterbox(img, (64, 64))
    assert chw[0, 0, 0] == 10 and chw[2, 0, 0] == 200
