#!/usr/bin/env python3
"""Restore an image or directory with a paired/unpaired RaGAN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import IMAGE_EXTENSIONS, list_images
from engine import load_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input RGB image or directory")
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=0, help="Core tile size; 0 disables tiling")
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    args = parser.parse_args()
    if args.tile_size and args.tile_size < 16:
        parser.error("tile-size must be 0 or at least 16")
    if args.tile_overlap < 0 or (args.tile_size and args.tile_overlap >= args.tile_size):
        parser.error("tile-overlap must be non-negative and smaller than tile-size")
    return args


def pad_to_multiple(image: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = image.shape
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if not pad_height and not pad_width:
        return image, height, width
    mode = "reflect" if min(height, width) > max(pad_height, pad_width) else "replicate"
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
    return padded, height, width


@torch.no_grad()
def restore_tensor(
    image: torch.Tensor,
    generator: torch.nn.Module,
    tile_size: int = 0,
    tile_overlap: int = 32,
    amp: bool = False,
) -> torch.Tensor:
    device = next(generator.parameters()).device
    image = image.to(device)
    batch, channels, height, width = image.shape
    if batch != 1:
        raise ValueError("Inference expects one image at a time")
    if not tile_size or (height <= tile_size and width <= tile_size):
        padded, original_height, original_width = pad_to_multiple(image)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = generator(padded).float()
        return output[..., :original_height, :original_width].clamp(-1, 1)

    result = torch.empty_like(image, dtype=torch.float32)
    for top in range(0, height, tile_size):
        core_bottom = min(height, top + tile_size)
        extended_top = max(0, top - tile_overlap)
        extended_bottom = min(height, core_bottom + tile_overlap)
        for left in range(0, width, tile_size):
            core_right = min(width, left + tile_size)
            extended_left = max(0, left - tile_overlap)
            extended_right = min(width, core_right + tile_overlap)
            tile = image[
                ..., extended_top:extended_bottom, extended_left:extended_right
            ]
            restored = restore_tensor(tile, generator, 0, 0, amp)
            result[..., top:core_bottom, left:core_right] = restored[
                ...,
                top - extended_top:core_bottom - extended_top,
                left - extended_left:core_right - extended_left,
            ]
    return result.clamp(-1, 1)


def restore_file(
    source: Path,
    destination: Path,
    generator: torch.nn.Module,
    tile_size: int,
    tile_overlap: int,
    amp: bool,
) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32).copy() / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    restored = restore_tensor(tensor, generator, tile_size, tile_overlap, amp)
    output = restored[0].permute(1, 2, 0).cpu().numpy()
    output = np.clip((output + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(destination)


def main() -> None:
    args = parse_args()
    generator, checkpoint = load_generator(args.checkpoint, args.device)
    source = Path(args.input)
    destination = Path(args.output)
    amp = args.amp and next(generator.parameters()).device.type == "cuda"
    if source.is_dir():
        if destination.suffix.lower() in IMAGE_EXTENSIONS:
            raise ValueError("Directory input requires a directory output path")
        for image_path in list_images(source):
            target = destination / image_path.name
            restore_file(
                image_path, target, generator,
                args.tile_size, args.tile_overlap, amp,
            )
            print(f"Restored {image_path.name} -> {target}")
    else:
        target = destination
        if destination.suffix.lower() not in IMAGE_EXTENSIONS:
            target = destination / source.name
        restore_file(
            source, target, generator,
            args.tile_size, args.tile_overlap, amp,
        )
        print(f"Restored {source} -> {target}")
    print(f"Checkpoint mode: {checkpoint['mode']}; epoch: {checkpoint['epoch']}")


if __name__ == "__main__":
    main()
