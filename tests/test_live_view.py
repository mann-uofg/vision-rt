"""The live viewer's side of the vrt_multistream --show stream (format in apps/live_feed.h).

CPU only: builds messages the way the C++ writer lays them out, then parses and draws them.
"""

import io
import struct
import sys

import numpy as np
import pytest

from vrt.coco import COW
from vrt.yolox import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
import live_view  # noqa: E402
import render_demo as rd  # noqa: E402


def message(cam: int, dets: list[list[float]], w: int = 640, h: int = 360, frame: int = 7) -> bytes:
    head = struct.pack("<4s5i4f", b"VRT1", cam, w, h, len(dets), frame, 6.5, 30.0, 6.1, 8.2)
    assert len(head) == 40  # static_assert(sizeof(Header) == 40) on the C++ side
    image = np.full((h, w, 3), 40 + cam, dtype=np.uint8)
    return head + np.asarray(dets, dtype=np.float32).tobytes() + image.tobytes()


def test_reads_messages_back_to_back():
    cow = [10, 20, 110, 90, 0.9, COW]
    person = [200, 50, 240, 150, 0.6, 0]
    stream = io.BytesIO(message(0, [cow, person]) + message(3, [], frame=8))

    a = live_view.read_message(stream)
    assert (a.cam, a.frame, a.image.shape) == (0, 7, (360, 640, 3))
    np.testing.assert_array_equal(a.dets, np.asarray([cow, person], dtype=np.float32))
    assert a.latency_ms == 6.5 and a.p99_ms == pytest.approx(8.2)
    assert a.image[0, 0, 0] == 40

    b = live_view.read_message(stream)
    assert (b.cam, b.frame, b.dets.shape) == (3, 8, (0, 6))
    assert live_view.read_message(stream) is None


def test_truncated_stream_ends_cleanly():
    data = message(1, [[0, 0, 5, 5, 0.5, COW]])
    assert live_view.read_message(io.BytesIO(data[:-100])) is None


def test_rejects_other_input():
    with pytest.raises(ValueError, match="bad message header"):
        live_view.read_message(io.BytesIO(b"frame,x1,y1,x2,y2,score,cls\n" + bytes(64)))


def test_compose_draws_every_camera():
    frames = [live_view.read_message(io.BytesIO(message(k, [[10, 20, 110, 90, 0.9, COW]])))
              for k in range(3)]
    canvas = live_view.compose(frames + [None], ["a", "b", "c", "d"], viewer_fps=60)
    assert canvas.shape == (rd.H, rd.W, 3) and canvas.dtype == np.uint8
    # Tiles 1-3 hold their camera's grey; tile 4 has no frame yet and stays mostly black.
    for k in range(4):
        y = rd.HEADER + (k // 2) * rd.TILE_H + rd.TILE_H // 2
        x = rd.X0 + (k % 2) * rd.TILE_W + rd.TILE_W - 20
        assert canvas[y, x, 0] == (40 + k if k < 3 else 0)
