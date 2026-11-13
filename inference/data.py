"""Paired and unpaired image datasets with native-DPI patch cropping."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(directory: Path, allow_empty: bool = False) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths and not allow_empty:
        raise ValueError(f"No supported images found in {directory}")
    return paths


def domain_directory(root: Path, split: str, domain: str) -> Path:
    candidates = [root / split / domain, root / f"{split}{domain}"]
    if split == "train":
        candidates.append(root / domain)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate domain {domain!r}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def prepare_canvas(image: Image.Image, image_size: int, native_dpi: bool = True) -> Image.Image:
    """Preserve 1:1 600-DPI resolution when larger than image_size; only upscale tiny inputs."""
    width, height = image.size
    if native_dpi and min(width, height) >= image_size:
        return image
    target_short = image_size if native_dpi else max(image_size, round(image_size * 1.12))
    scale = target_short / min(width, height)
    return image.resize(
        (max(image_size, round(width * scale)), max(image_size, round(height * scale))),
        Image.Resampling.BICUBIC,
    )


def sample_geometry(width: int, height: int, image_size: int, training: bool) -> tuple[int, int, bool]:
    left = random.randint(0, width - image_size) if training else (width - image_size) // 2
    top = random.randint(0, height - image_size) if training else (height - image_size) // 2
    flip = training and random.random() < 0.5
    return left, top, flip


def transform_image(
    image: Image.Image,
    image_size: int,
    training: bool,
    geometry: tuple[int, int, bool] | None = None,
    native_dpi: bool = True,
) -> torch.Tensor:
    canvas = prepare_canvas(image, image_size, native_dpi=native_dpi)
    if geometry is None:
        geometry = sample_geometry(canvas.width, canvas.height, image_size, training)
    left, top, flip = geometry
    cropped = canvas.crop((left, top, left + image_size, top + image_size))
    if flip:
        cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return pil_to_tensor(cropped)


class UnpairedImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        split: str = "train",
        native_dpi: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.training = split == "train"
        self.native_dpi = native_dpi
        self.paths_a = list_images(domain_directory(self.root, split, "A"))
        self.paths_b = list_images(domain_directory(self.root, split, "B"))

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index_a = index % len(self.paths_a)
        index_b = random.randrange(len(self.paths_b)) if self.training else index % len(self.paths_b)
        with Image.open(self.paths_a[index_a]) as image_a, Image.open(self.paths_b[index_b]) as image_b:
            return {
                "A": transform_image(
                    image_a.convert("RGB"), self.image_size, self.training, native_dpi=self.native_dpi
                ),
                "B": transform_image(
                    image_b.convert("RGB"), self.image_size, self.training, native_dpi=self.native_dpi
                ),
            }


class PairedImageDataset(Dataset[dict[str, torch.Tensor]]):
    """A is damaged chrome and B is clean chrome; filenames & dimensions must match."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        split: str = "train",
        native_dpi: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.training = split == "train"
        self.native_dpi = native_dpi
        directory_a = domain_directory(self.root, split, "A")
        directory_b = domain_directory(self.root, split, "B")
        self.paths_a = list_images(directory_a)
        paths_by_name = {path.name: path for path in list_images(directory_b, allow_empty=True)}
        missing = [path.name for path in self.paths_a if path.name not in paths_by_name]
        extra = sorted(set(paths_by_name) - {path.name for path in self.paths_a})
        if missing or extra:
            raise ValueError(
                "Paired datasets require identical filenames in A and B. "
                f"Missing in B: {missing[:5]}; extra in B: {extra[:5]}"
            )
        self.paths_b = [paths_by_name[path.name] for path in self.paths_a]

    def __len__(self) -> int:
        return len(self.paths_a)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with Image.open(self.paths_a[index]) as image_a, Image.open(self.paths_b[index]) as image_b:
            image_a = image_a.convert("RGB")
            image_b = image_b.convert("RGB")
            if image_a.size != image_b.size:
                raise ValueError(
                    f"Paired image dimensions differ for {self.paths_a[index].name}: "
                    f"{image_a.size} versus {image_b.size}"
                )
            canvas_a = prepare_canvas(image_a, self.image_size, native_dpi=self.native_dpi)
            geometry = sample_geometry(canvas_a.width, canvas_a.height, self.image_size, self.training)
            return {
                "A": transform_image(
                    image_a, self.image_size, self.training, geometry, native_dpi=self.native_dpi
                ),
                "B": transform_image(
                    image_b, self.image_size, self.training, geometry, native_dpi=self.native_dpi
                ),
            }


def build_dataset(
    mode: str,
    root: str | Path,
    image_size: int,
    split: str = "train",
    native_dpi: bool = True,
) -> Dataset[dict[str, torch.Tensor]]:
    dataset_type: Callable[..., Dataset[dict[str, torch.Tensor]]]
    dataset_type = PairedImageDataset if mode == "paired" else UnpairedImageDataset
    return dataset_type(root=root, image_size=image_size, split=split, native_dpi=native_dpi)
