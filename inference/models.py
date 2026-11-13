"""Fully convolutional generators and PatchGAN critics for chrome restoration.

All networks consume and produce RGB tensors in [-1, 1]. Discriminators return
raw logits; sigmoid is deliberately left to BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
from torch import nn


def weights_init_normal(module: nn.Module) -> None:
    """Initialize convolution and normalization layers without touching buffers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if module.affine:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.zeros_(module.bias)


class ResidualDenseBlock(nn.Module):
    """Five convolutions with dense feature reuse and local residual scaling."""

    def __init__(self, channels: int, growth_channels: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.activation(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.activation(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * self.residual_scale


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, channels: int, growth_channels: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualDenseBlock(channels, growth_channels, residual_scale),
            ResidualDenseBlock(channels, growth_channels, residual_scale),
            ResidualDenseBlock(channels, growth_channels, residual_scale),
        )
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.blocks(x) * self.residual_scale


class RRDBNet(nn.Module):
    """Same-resolution RRDB restoration generator (no super-resolution upsampler)."""

    def __init__(
        self,
        channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 9,
        growth_channels: int = 32,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.conv_first = nn.Conv2d(channels, base_channels, 3, padding=1)
        self.body = nn.Sequential(
            *(RRDB(base_channels, growth_channels) for _ in range(num_blocks))
        )
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_last = nn.Conv2d(base_channels, channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_first(x)
        body = self.conv_body(self.body(features))
        return torch.tanh(self.conv_last(self.activation(features + body)))


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    """CycleGAN-style ResNet generator retained for lightweight unpaired training."""

    def __init__(self, channels: int = 3, num_residual_blocks: int = 9, base_channels: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = base_channels
        for _ in range(2):
            out_channels = in_channels * 2
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.extend(ResnetBlock(in_channels) for _ in range(num_residual_blocks))
        for _ in range(2):
            out_channels = in_channels // 2
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [nn.ReflectionPad2d(3), nn.Conv2d(base_channels, channels, 7), nn.Tanh()]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70-style PatchGAN critic returning unbounded raw logits."""

    def __init__(self, channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        layers: list[nn.Module] = [
            nn.Conv2d(channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = base_channels
        for index in range(1, n_layers):
            out_channels = min(base_channels * (2**index), base_channels * 8)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_channels = out_channels
        layers.extend(
            [
                nn.Conv2d(in_channels, in_channels, 4, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels, 1, 4, padding=1),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


# Backwards-compatible name for the uploaded CycleGAN integration.
Discriminator = PatchDiscriminator


def build_generator(
    name: str,
    channels: int = 3,
    residual_blocks: int = 9,
    base_channels: int = 64,
) -> nn.Module:
    normalized = name.lower()
    if normalized == "rrdb":
        return RRDBNet(
            channels=channels,
            base_channels=base_channels,
            num_blocks=residual_blocks,
            growth_channels=max(4, base_channels // 2),
        )
    if normalized == "resnet":
        return GeneratorResNet(channels, residual_blocks, base_channels)
    raise ValueError(f"Unknown generator {name!r}; expected 'rrdb' or 'resnet'")
