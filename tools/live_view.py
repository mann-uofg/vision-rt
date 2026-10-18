#!/usr/bin/env python3
"""Live 2x2 camera wall for `vrt_multistream --show` (bench/live_demo.sh runs both).

    build/release/vrt_multistream ... --show | python tools/live_view.py --names a,b,c,d

Reads the message stream on stdin (format in apps/live_feed.h: per camera frame, the
detections and a 640x360 BGR copy made on the GPU) and draws each camera's newest frame with the
same tiles as the offline demo (tools/render_demo.py). Every number on screen is measured by the
C++ pipeline, per camera over a rolling window; this viewer only draws, and it may skip frames
if it falls behind, since the pipeline never waits for it.

The window is Tk (the pip OpenCV build has no GUI); WSLg or any X/Wayland display works.
Close it or press q / Esc to stop; vrt_multistream then prints its run summary.
"""

import argparse
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

# The drawing code (render_demo, which pulls in torch through vrt.coco) is imported only once
# stdin is being read, so the ~3 s import doesn't stall the stream at startup.

MAGIC = b"VRT1"
HEAD = struct.Struct("<4s5i4f")  # magic, cam, width, height, ndet, frame, latency, fps, p50, p99


@dataclass
class CamFrame:
    cam: int
    frame: int
    latency_ms: float
    fps: float
    p50_ms: float
    p99_ms: float
    dets: np.ndarray  # (ndet, 6): x1, y1, x2, y2, score, cls in image pixels
    image: np.ndarray  # (height, width, 3) BGR


def read_exact(f: BinaryIO, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_message(f: BinaryIO) -> CamFrame | None:
    """The next message, or None at end of stream."""
    head = read_exact(f, HEAD.size)
    if head is None:
        return None
    magic, cam, w, h, ndet, frame, lat, fps, p50, p99 = HEAD.unpack(head)
    if magic != MAGIC:
        raise ValueError(f"bad message header {magic!r}: is stdin vrt_multistream --show?")
    body = read_exact(f, ndet * 24 + w * h * 3)
    if body is None:
        return None
    dets = np.frombuffer(body, np.float32, ndet * 6).reshape(ndet, 6)
    image = np.frombuffer(body, np.uint8, offset=ndet * 24).reshape(h, w, 3)
    return CamFrame(cam, frame, lat, fps, p50, p99, dets, image)


def compose(latest: list[CamFrame | None], names: list[str], viewer_fps: float) -> np.ndarray:
    """The 2x2 canvas (BGR) for the newest frame of each camera."""
    from render_demo import GREEN, HEADER, TILE_H, TILE_W, H, W, X0, draw_tile, text

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    total = 0
    for k, (f, name) in enumerate(zip(latest, names)):
        y, x = HEADER + (k // 2) * TILE_H, X0 + (k % 2) * TILE_W
        if f is None:
            text(canvas, f"CAM {k + 1}  {name}: starting", (x + 12, y + TILE_H // 2), 0.5)
            continue
        tile, cows = draw_tile(f.image, f.dets, k + 1, name)
        tile[:24] = tile[:24] * 0.45  # dark band under the numbers, like the caption bar
        p99 = f"p99 {f.p99_ms:4.1f} ms" if f.p99_ms >= 0 else "warming up"
        text(tile, f"{f.fps:4.1f} fps   latency {f.latency_ms:4.1f} ms   {p99}", (8, 17), 0.45)
        total += cows
        canvas[y : y + TILE_H, x : x + TILE_W] = tile

    text(canvas, "vision-rt LIVE  |  barn monitoring, 4 cameras  |  YOLOX-S FP16 on an RTX 5070 Ti",
         (10, 22), 0.55)
    text(canvas, f"cows in view: {total:2d}", (W - 180, 22), 0.6, GREEN, 1)
    live = [f for f in latest if f is not None]
    if live:
        warm = [f.p99_ms for f in live if f.p99_ms >= 0]
        p99 = f"worst camera p99 {max(warm):4.1f} ms (last 10 s)" if warm else "warming up"
        footer = (f"measured live by the pipeline: {sum(f.fps for f in live):5.1f} fps total  |  "
                  f"{p99}  |  NVDEC -> CUDA preprocess -> TensorRT -> GPU NMS  |  "
                  f"window {viewer_fps:2.0f} fps")
    else:
        footer = "loading the engine and capturing CUDA graphs..."
    text(canvas, footer, (10, H - 9), 0.42, (220, 220, 220))
    return canvas


class Reader(threading.Thread):
    """Reads stdin on its own thread and keeps the newest frame per camera."""

    def __init__(self, f: BinaryIO, cams: int):
        super().__init__(daemon=True)
        self.f = f
        self.lock = threading.Lock()
        self.latest: list[CamFrame | None] = [None] * cams
        self.version = 0
        self.ended = False
        self.error: Exception | None = None
        self.received = 0
        self.skipped = 0  # frames the pipeline finished but the writer replaced before sending

    def run(self) -> None:
        try:
            while (m := read_message(self.f)) is not None:
                if 0 <= m.cam < len(self.latest):
                    prev = self.latest[m.cam]
                    self.received += 1
                    if prev is not None and m.frame > prev.frame:
                        self.skipped += m.frame - prev.frame - 1
                    with self.lock:
                        self.latest[m.cam] = m
                        self.version += 1
        except Exception as e:  # noqa: BLE001 - reported by the window loop
            self.error = e
        self.ended = True

    def snapshot(self) -> tuple[int, list[CamFrame | None]]:
        with self.lock:
            return self.version, list(self.latest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="cam1,cam2,cam3,cam4", help="comma-separated, 4 cameras")
    args = ap.parse_args()
    names = (args.names.split(",") + ["", "", "", ""])[:4]

    reader = Reader(sys.stdin.buffer, 4)
    reader.start()
    import tkinter as tk

    from PIL import Image, ImageTk
    from render_demo import H, W

    root = tk.Tk()
    root.title("vision-rt live")
    root.configure(background="black")
    photo = ImageTk.PhotoImage(Image.new("RGB", (W, H)))
    label = tk.Label(root, image=photo, borderwidth=0)
    label.pack()
    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("q", lambda e: root.destroy())

    shown = -1
    painted = 0
    paints: list[float] = []  # repaint times over the last second

    def tick() -> None:
        nonlocal shown, painted
        if reader.ended:
            root.destroy()
            return
        version, latest = reader.snapshot()
        now = time.perf_counter()
        # Repaint when something changed, at most 60 times a second: the cameras are staggered
        # across the frame interval, so 60 Hz shows every camera's frames within 17 ms.
        if version != shown and (not paints or now - paints[-1] >= 1 / 60) or shown < 0:
            shown = version
            painted += 1
            paints.append(now)
            while paints and now - paints[0] > 1.0:
                paints.pop(0)
            canvas = compose(latest, names, len(paints) - 1 if len(paints) > 1 else 0)
            photo.paste(Image.fromarray(np.ascontiguousarray(canvas[..., ::-1])))
        root.after(5, tick)

    started = time.perf_counter()
    tick()
    root.mainloop()
    seconds = time.perf_counter() - started
    print(f"live_view: {reader.received} camera frames received, {reader.skipped} skipped; "
          f"window repainted {painted / seconds:.1f} times/s over {seconds:.0f} s", file=sys.stderr)
    if reader.error is not None:
        sys.exit(f"live_view: {reader.error}")


if __name__ == "__main__":
    main()
