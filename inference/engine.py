"""Training engines and checkpointing for paired and unpaired chrome restoration."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from losses import (
    VGGPerceptualLoss,
    relativistic_average_d_loss,
    relativistic_average_g_loss,
)
from models import PatchDiscriminator, build_generator, weights_init_normal


@dataclass
class TrainerConfig:
    mode: str
    device: str = "auto"
    generator: str = "rrdb"
    channels: int = 3
    residual_blocks: int = 9
    generator_channels: int = 64
    discriminator_channels: int = 64
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    lambda_perceptual: float = 1.0
    lambda_identity: float = 5.0
    lambda_cycle: float = 10.0
    amp: bool = True
    replay_size: int = 50
    vgg_weights: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"paired", "unpaired"}:
            raise ValueError("mode must be 'paired' or 'unpaired'")
        if self.channels != 3:
            raise ValueError("This restoration pipeline currently requires RGB (channels=3)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name in (
            "lambda_l1", "lambda_perceptual", "lambda_identity", "lambda_cycle"
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.replay_size < 1:
            raise ValueError("replay_size must be at least 1")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def scale_backward_step(
    scaler: torch.amp.GradScaler,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def finite_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in metrics.items():
        scalar = float(value.detach().cpu())
        if not torch.isfinite(torch.tensor(scalar)):
            raise FloatingPointError(f"Non-finite training metric: {name}={scalar}")
        result[name] = scalar
    return result


class ReplayBuffer:
    """Image-history buffer that stabilizes the relativistic critics."""

    def __init__(self, max_size: int = 50):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self.images: list[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for image in batch.detach():
            image = image.detach().clone()
            if len(self.images) < self.max_size:
                self.images.append(image)
                outputs.append(image)
            elif random.random() < 0.5:
                outputs.append(image)
            else:
                index = random.randrange(self.max_size)
                outputs.append(self.images[index].clone())
                self.images[index] = image
        return torch.stack(outputs)


class BaseTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.perceptual: VGGPerceptualLoss | None = None
        if config.lambda_perceptual > 0:
            self.perceptual = VGGPerceptualLoss(config.vgg_weights).to(self.device)
            self.perceptual.eval()

    def autocast(self) -> torch.amp.autocast:
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def new_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)

    def perceptual_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.perceptual is None:
            return prediction.new_zeros(())
        # VGG feature extraction is kept in full precision for stable comparisons.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            return self.perceptual(prediction, target)

    def save_checkpoint(self, path: str | Path, epoch: int, global_step: int) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        raise NotImplementedError

    def _payload(
        self,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        scalers: dict[str, torch.amp.GradScaler],
        epoch: int,
        global_step: int,
    ) -> dict[str, Any]:
        return {
            "format_version": 1,
            "mode": self.config.mode,
            "config": asdict(self.config),
            "epoch": epoch,
            "global_step": global_step,
            "models": {name: model.state_dict() for name, model in models.items()},
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
        }

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def load_generator(
    checkpoint_path: str | Path, device: str = "auto"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the damaged->clean generator from either training mode."""
    target_device = resolve_device(device)
    checkpoint = read_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    generator = build_generator(
        config["generator"],
        channels=config["channels"],
        residual_blocks=config["residual_blocks"],
        base_channels=config["generator_channels"],
    )
    generator.load_state_dict(checkpoint["models"]["G_AB"])
    generator.to(target_device).eval()
    return generator, checkpoint


class PairedTrainer(BaseTrainer):
    """Supervised damaged->clean restoration with RaGAN, L1, and VGG losses."""

    def __init__(self, config: TrainerConfig):
        if config.mode != "paired":
            raise ValueError("PairedTrainer requires mode='paired'")
        super().__init__(config)
        self.generator = build_generator(
            config.generator,
            config.channels,
            config.residual_blocks,
            config.generator_channels,
        ).to(self.device)
        self.discriminator = PatchDiscriminator(
            config.channels, config.discriminator_channels
        ).to(self.device)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.scaler_g = self.new_scaler()
        self.scaler_d = self.new_scaler()

    def train_step(
        self, damaged: torch.Tensor, clean: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        damaged = damaged.to(self.device, non_blocking=True)
        clean = clean.to(self.device, non_blocking=True)
        self.generator.train()
        self.discriminator.train()

        set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad(set_to_none=True)
        with self.autocast():
            restored = self.generator(damaged)
            with torch.no_grad():
                real_logits = self.discriminator(clean)
            fake_logits = self.discriminator(restored)
            adversarial = relativistic_average_g_loss(real_logits, fake_logits)
            l1 = torch.nn.functional.l1_loss(restored, clean)
            identity = (
                torch.nn.functional.l1_loss(self.generator(clean), clean)
                if self.config.lambda_identity > 0
                else restored.new_zeros(())
            )
            perceptual = self.perceptual_loss(restored, clean)
            generator_loss = (
                adversarial
                + self.config.lambda_l1 * l1
                + self.config.lambda_identity * identity
                + self.config.lambda_perceptual * perceptual
            )
        scale_backward_step(self.scaler_g, generator_loss, self.optimizer_g)

        set_requires_grad(self.discriminator, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        with self.autocast():
            real_logits = self.discriminator(clean)
            fake_logits = self.discriminator(restored.detach())
            discriminator_loss = relativistic_average_d_loss(real_logits, fake_logits)
        scale_backward_step(self.scaler_d, discriminator_loss, self.optimizer_d)

        metrics = finite_metrics(
            {
                "total_g": generator_loss,
                "adv_g": adversarial,
                "l1": l1,
                "identity": identity,
                "perceptual": perceptual,
                "total_d": discriminator_loss,
            }
        )
        return metrics, {"A": damaged.detach(), "B": clean.detach(), "fake_B": restored.detach()}

    def save_checkpoint(self, path: str | Path, epoch: int, global_step: int) -> None:
        payload = self._payload(
            {"G_AB": self.generator, "D_B": self.discriminator},
            {"G": self.optimizer_g, "D": self.optimizer_d},
            {"G": self.scaler_g, "D": self.scaler_d},
            epoch,
            global_step,
        )
        self._atomic_save(payload, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = read_checkpoint(path, map_location=self.device)
        if checkpoint["mode"] != self.config.mode:
            raise ValueError(
                f"Checkpoint mode {checkpoint['mode']!r} does not match {self.config.mode!r}"
            )
        self.generator.load_state_dict(checkpoint["models"]["G_AB"])
        self.discriminator.load_state_dict(checkpoint["models"]["D_B"])
        self.optimizer_g.load_state_dict(checkpoint["optimizers"]["G"])
        self.optimizer_d.load_state_dict(checkpoint["optimizers"]["D"])
        self.scaler_g.load_state_dict(checkpoint["scalers"]["G"])
        self.scaler_d.load_state_dict(checkpoint["scalers"]["D"])
        return checkpoint


class UnpairedTrainer(BaseTrainer):
    """Bidirectional CycleGAN with symmetric RaGAN adversarial losses."""

    def __init__(self, config: TrainerConfig):
        if config.mode != "unpaired":
            raise ValueError("UnpairedTrainer requires mode='unpaired'")
        super().__init__(config)
        generator_arguments = (
            config.channels,
            config.residual_blocks,
            config.generator_channels,
        )
        self.g_ab = build_generator(config.generator, *generator_arguments).to(self.device)
        self.g_ba = build_generator(config.generator, *generator_arguments).to(self.device)
        self.d_a = PatchDiscriminator(config.channels, config.discriminator_channels).to(self.device)
        self.d_b = PatchDiscriminator(config.channels, config.discriminator_channels).to(self.device)
        for module in (self.g_ab, self.g_ba, self.d_a, self.d_b):
            module.apply(weights_init_normal)
        self.optimizer_g = torch.optim.Adam(
            list(self.g_ab.parameters()) + list(self.g_ba.parameters()),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_d = torch.optim.Adam(
            list(self.d_a.parameters()) + list(self.d_b.parameters()),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        self.scaler_g = self.new_scaler()
        self.scaler_d = self.new_scaler()
        self.fake_a_buffer = ReplayBuffer(config.replay_size)
        self.fake_b_buffer = ReplayBuffer(config.replay_size)

    def train_step(
        self, damaged: torch.Tensor, clean: torch.Tensor
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        damaged = damaged.to(self.device, non_blocking=True)
        clean = clean.to(self.device, non_blocking=True)
        self.g_ab.train()
        self.g_ba.train()
        self.d_a.train()
        self.d_b.train()

        set_requires_grad(self.d_a, False)
        set_requires_grad(self.d_b, False)
        self.optimizer_g.zero_grad(set_to_none=True)
        with self.autocast():
            fake_clean = self.g_ab(damaged)
            fake_damaged = self.g_ba(clean)
            recovered_damaged = self.g_ba(fake_clean)
            recovered_clean = self.g_ab(fake_damaged)

            with torch.no_grad():
                real_damaged_logits = self.d_a(damaged)
                real_clean_logits = self.d_b(clean)
            adversarial_a = relativistic_average_g_loss(
                real_damaged_logits, self.d_a(fake_damaged)
            )
            adversarial_b = relativistic_average_g_loss(
                real_clean_logits, self.d_b(fake_clean)
            )
            adversarial = 0.5 * (adversarial_a + adversarial_b)

            cycle_a = torch.nn.functional.l1_loss(recovered_damaged, damaged)
            cycle_b = torch.nn.functional.l1_loss(recovered_clean, clean)
            cycle = 0.5 * (cycle_a + cycle_b)
            identity_a = torch.nn.functional.l1_loss(self.g_ba(damaged), damaged)
            identity_b = torch.nn.functional.l1_loss(self.g_ab(clean), clean)
            identity = 0.5 * (identity_a + identity_b)
            perceptual = self.perceptual_loss(fake_clean, damaged)
            generator_loss = (
                adversarial
                + self.config.lambda_cycle * cycle
                + self.config.lambda_identity * identity
                + self.config.lambda_perceptual * perceptual
            )
        scale_backward_step(self.scaler_g, generator_loss, self.optimizer_g)

        set_requires_grad(self.d_a, True)
        set_requires_grad(self.d_b, True)
        self.optimizer_d.zero_grad(set_to_none=True)
        historical_damaged = self.fake_a_buffer.push_and_pop(fake_damaged)
        historical_clean = self.fake_b_buffer.push_and_pop(fake_clean)
        with self.autocast():
            loss_d_a = relativistic_average_d_loss(
                self.d_a(damaged), self.d_a(historical_damaged)
            )
            loss_d_b = relativistic_average_d_loss(
                self.d_b(clean), self.d_b(historical_clean)
            )
            discriminator_loss = 0.5 * (loss_d_a + loss_d_b)
        scale_backward_step(self.scaler_d, discriminator_loss, self.optimizer_d)

        metrics = finite_metrics(
            {
                "total_g": generator_loss,
                "adv_g": adversarial,
                "cycle": cycle,
                "identity": identity,
                "perceptual": perceptual,
                "total_d": discriminator_loss,
                "d_a": loss_d_a,
                "d_b": loss_d_b,
            }
        )
        outputs = {
            "A": damaged.detach(),
            "B": clean.detach(),
            "fake_A": fake_damaged.detach(),
            "fake_B": fake_clean.detach(),
            "recovered_A": recovered_damaged.detach(),
            "recovered_B": recovered_clean.detach(),
        }
        return metrics, outputs

    def save_checkpoint(self, path: str | Path, epoch: int, global_step: int) -> None:
        payload = self._payload(
            {"G_AB": self.g_ab, "G_BA": self.g_ba, "D_A": self.d_a, "D_B": self.d_b},
            {"G": self.optimizer_g, "D": self.optimizer_d},
            {"G": self.scaler_g, "D": self.scaler_d},
            epoch,
            global_step,
        )
        self._atomic_save(payload, path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = read_checkpoint(path, map_location=self.device)
        if checkpoint["mode"] != self.config.mode:
            raise ValueError(
                f"Checkpoint mode {checkpoint['mode']!r} does not match {self.config.mode!r}"
            )
        self.g_ab.load_state_dict(checkpoint["models"]["G_AB"])
        self.g_ba.load_state_dict(checkpoint["models"]["G_BA"])
        self.d_a.load_state_dict(checkpoint["models"]["D_A"])
        self.d_b.load_state_dict(checkpoint["models"]["D_B"])
        self.optimizer_g.load_state_dict(checkpoint["optimizers"]["G"])
        self.optimizer_d.load_state_dict(checkpoint["optimizers"]["D"])
        self.scaler_g.load_state_dict(checkpoint["scalers"]["G"])
        self.scaler_d.load_state_dict(checkpoint["scalers"]["D"])
        return checkpoint


def create_trainer(config: TrainerConfig) -> PairedTrainer | UnpairedTrainer:
    if config.mode == "paired":
        return PairedTrainer(config)
    return UnpairedTrainer(config)
