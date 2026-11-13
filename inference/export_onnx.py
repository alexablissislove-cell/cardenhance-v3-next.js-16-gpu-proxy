#!/usr/bin/env python3
"""Export a trained Chrome RaGAN G_AB checkpoint to ONNX for onnxruntime-web."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from engine import load_generator


class NormalizedGeneratorWrapper(nn.Module):
    """Wraps [-1, 1] generator to accept/emit [0, 1] NCHW float32 tensors in browser workers."""

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, rgb_01: torch.Tensor) -> torch.Tensor:
        x = rgb_01 * 2.0 - 1.0
        out = self.generator(x)
        return ((out + 1.0) * 0.5).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to last.pth / epoch_*.pth")
    parser.add_argument("--output", default="chrome_restore_g_ab.onnx", help="Output .onnx path")
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    generator, meta = load_generator(args.checkpoint, device="cpu")
    wrapped = NormalizedGeneratorWrapper(generator).eval()
    dummy = torch.rand(1, 3, args.tile_size, args.tile_size, dtype=torch.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["input_rgb"],
        output_names=["restored_rgb"],
        dynamic_axes={
            "input_rgb": {2: "height", 3: "width"},
            "restored_rgb": {2: "height", 3: "width"},
        },
        dynamo=False,
    )
    print(f"Exported {meta['mode']} G_AB (epoch {meta['epoch']}) -> {out_path}")


if __name__ == "__main__":
    main()