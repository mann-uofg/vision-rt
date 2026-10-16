#!/usr/bin/env python3
"""Export YOLOX to ONNX for TensorRT.

TensorRT 11 networks are strongly typed: the precision of every layer comes from the types in
the ONNX graph, not from builder flags. So there is one ONNX file per precision.

Graph contract (both precisions):
    input  "images": [batch, 3, 640, 640], BGR, 0-255. float32, or float16 for --fp16.
    output "preds":  [batch, 8400, 85] float32. cx, cy, w, h in input pixels, objectness,
                     80 class scores (sigmoided).

For --fp16 the network body runs in half precision, but box decoding (grid offset, exp,
stride) runs in float32. Near 640 px, float16 can only represent every 0.5 px, and exp() of
a half-precision regression output loses more. Decoding is a handful of elementwise ops, so
doing it in fp32 costs nothing.

    python tools/export_onnx.py --size s            # models/yolox_s_fp32.onnx
    python tools/export_onnx.py --size s --fp16     # models/yolox_s_fp16.onnx
"""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from vrt import yolox

STRIDES = (8, 16, 32)


def anchor_grid(input_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid cell offsets [1, N, 2] and strides [1, N, 1], in the head's output order."""
    grids, strides = [], []
    for s in STRIDES:
        h, w = input_hw[0] // s, input_hw[1] // s
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grids.append(torch.stack((xv, yv), 2).view(1, -1, 2))
        strides.append(torch.full((1, h * w, 1), float(s)))
    return torch.cat(grids, 1).float(), torch.cat(strides, 1)


class Exportable(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype):
        super().__init__()
        model.head.decode_in_inference = False
        self.model = model.to(dtype)
        grid, stride = anchor_grid(yolox.INPUT_HW)
        self.register_buffer("grid", grid)
        self.register_buffer("stride", stride)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.model(images).float()
        xy = (raw[..., 0:2] + self.grid) * self.stride
        wh = torch.exp(raw[..., 2:4]) * self.stride
        return torch.cat([xy, wh, raw[..., 4:]], dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(yolox.SIZES), default="s")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    precision = "fp16" if args.fp16 else "fp32"
    out = args.out or yolox.REPO_ROOT / "models" / f"yolox_{args.size}_{precision}.onnx"

    model = Exportable(yolox.build(args.size, yolox.default_weights(args.size)), dtype)
    model = model.cuda().eval()
    x = torch.rand(args.batch, 3, *yolox.INPUT_HW, device="cuda").mul(255).to(dtype)

    with torch.inference_mode():
        reference = yolox.build(args.size, yolox.default_weights(args.size))(x.float())
        mine = model(x)
    print(f"wrapper vs upstream decode, max abs diff: {(mine - reference).abs().max().item():.4g}")

    torch.onnx.export(
        model, (x,), str(out),
        input_names=["images"], output_names=["preds"],
        opset_version=args.opset, dynamo=True, external_data=False,
    )
    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(m.graph.node)} nodes)")


if __name__ == "__main__":
    main()
