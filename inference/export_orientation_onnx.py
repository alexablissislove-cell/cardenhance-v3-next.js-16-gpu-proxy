#!/usr/bin/env python3
"""Export ternaus/check_orientation (swsl_resnext50_32x4d) to ONNX for onnxruntime-web.

In ternaus/check_orientation/dataloaders.py:
  orientation = random.randint(0, 3)
  image = np.rot90(image, orientation)  # k * 90 deg counter-clockwise (CCW)

Therefore, when the model predicts class k in {0, 1, 2, 3}, the input image has been
rotated k * 90 deg CCW from upright, so rotating the image by (k * 90) % 360 deg
CLOCKWISE (CW) restores it to upright (0 deg).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch
from torch import nn
from torch.utils import model_zoo

CHECKPOINT_URL = (
    "https://github.com/ternaus/check_orientation/releases/download/"
    "v0.0.3/2020-11-16_resnext50_32x4d.zip"
)


def load_ternaus_orientation_model(weights_path: str | None = None) -> nn.Module:
    model = timm.create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4)
    if weights_path:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        raw = model_zoo.load_url(CHECKPOINT_URL, progress=True, map_location="cpu")
    state_dict = raw["state_dict"] if "state_dict" in raw else raw
    cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    return nn.Sequential(model, nn.Softmax(dim=1)).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="Optional local checkpoint path")
    parser.add_argument(
        "--output",
        default="ternaus_check_orientation_224.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()

    model = load_ternaus_orientation_model(args.weights)
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["input_rgb_224"],
        output_names=["probs_ccw_k"],
        dynamo=False,
    )
    print(f"Exported ternaus/check_orientation ONNX -> {out_path}")


if __name__ == "__main__":
    main()