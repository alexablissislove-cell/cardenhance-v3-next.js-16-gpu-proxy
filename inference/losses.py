"""Relativistic average GAN and restoration losses."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


def relativistic_average_d_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """RaGAN discriminator loss on raw logits.

    Real samples are trained to be more realistic than the fake mean, while
    fake samples are trained to be less realistic than the real mean.
    """
    if real_logits.shape != fake_logits.shape:
        raise ValueError("real_logits and fake_logits must have identical shapes")
    real_relative = real_logits - fake_logits.mean()
    fake_relative = fake_logits - real_logits.mean()
    real_loss = F.binary_cross_entropy_with_logits(
        real_relative, torch.ones_like(real_relative)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_relative, torch.zeros_like(fake_relative)
    )
    return 0.5 * (real_loss + fake_loss)


def relativistic_average_g_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Symmetric RaGAN generator loss; pass detached real logits."""
    if real_logits.shape != fake_logits.shape:
        raise ValueError("real_logits and fake_logits must have identical shapes")
    real_relative = real_logits - fake_logits.mean()
    fake_relative = fake_logits - real_logits.mean()
    real_loss = F.binary_cross_entropy_with_logits(
        real_relative, torch.zeros_like(real_relative)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_relative, torch.ones_like(fake_relative)
    )
    return 0.5 * (real_loss + fake_loss)


class VGGPerceptualLoss(nn.Module):
    """Multi-layer VGG19 feature L1 loss for [-1, 1] RGB images."""

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__()
        if weights_path is None:
            weights = VGG19_Weights.IMAGENET1K_V1
            network = vgg19(weights=weights)
        else:
            network = vgg19(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            network.load_state_dict(state)
        # ReLU1_2, ReLU2_2, ReLU3_4, ReLU4_4 in torchvision's VGG19 features.
        selected = {3, 8, 17, 26}
        self.blocks = nn.ModuleList()
        block: list[nn.Module] = []
        for index, layer in enumerate(network.features):
            block.append(layer)
            if index in selected:
                self.blocks.append(nn.Sequential(*block))
                block = []
        for module in self.blocks:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> "VGGPerceptualLoss":
        # Perceptual features must stay in evaluation mode even when GANs train.
        super().train(False)
        return self

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("Perceptual inputs must have identical shapes")
        if prediction.shape[1] != 3:
            raise ValueError("VGGPerceptualLoss requires three-channel RGB input")
        prediction = (prediction.float() + 1.0) * 0.5
        target = (target.float() + 1.0) * 0.5
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = prediction.new_zeros(())
        for block in self.blocks:
            prediction = block(prediction)
            with torch.no_grad():
                target = block(target)
            loss = loss + F.l1_loss(prediction, target)
        return loss / len(self.blocks)
