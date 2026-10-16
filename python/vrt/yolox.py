"""Build YOLOX from the pinned upstream source and load the released COCO weights."""

import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
_YOLOX_SRC = REPO_ROOT / "third_party" / "YOLOX"
if str(_YOLOX_SRC) not in sys.path:
    sys.path.insert(0, str(_YOLOX_SRC))

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead  # noqa: E402

# (depth, width) multipliers, from exps/default/yolox_{s,m}.py upstream.
SIZES = {"s": (0.33, 0.50), "m": (0.67, 0.75)}
NUM_CLASSES = 80
INPUT_HW = (640, 640)


def build(size: str, weights: str | Path | None = None, device: str = "cuda") -> nn.Module:
    """Return a YOLOX model in eval mode.

    In eval mode the head decodes boxes itself, so the output is [batch, 8400, 85]:
    cx, cy, w, h in input pixels, objectness, then 80 class scores (both already sigmoided).
    """
    depth, width = SIZES[size]
    in_channels = [256, 512, 1024]
    model = YOLOX(
        YOLOPAFPN(depth, width, in_channels=in_channels, act="silu"),
        YOLOXHead(NUM_CLASSES, width, in_channels=in_channels, act="silu"),
    )
    # Upstream trains with these BatchNorm settings (yolox_base.py, get_model). eps matters at
    # inference too, so it has to match the checkpoint.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    if weights is not None:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt))
    return model.eval().to(device)


def default_weights(size: str) -> Path:
    return REPO_ROOT / "models" / f"yolox_{size}.pth"
