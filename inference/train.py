#!/usr/bin/env python3
"""Train paired RRDB/RaGAN or unpaired CycleGAN/RaGAN chrome restoration."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data import build_dataset
from engine import TrainerConfig, create_trainer, read_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Dataset containing train/A and train/B")
    parser.add_argument("--output", default="runs/chrome", help="Run output directory")
    parser.add_argument("--mode", choices=("paired", "unpaired"), default="unpaired")
    parser.add_argument("--generator", choices=("rrdb", "resnet"), default="rrdb")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--decay-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--residual-blocks", type=int, default=9)
    parser.add_argument("--generator-channels", type=int, default=64)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="Paired pixel loss weight")
    parser.add_argument("--lambda-perceptual", type=float, default=None)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--replay-size", type=int, default=50)
    parser.add_argument("--vgg-weights", default=None, help="Optional local VGG19 state_dict")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10, help="Steps between log messages")
    parser.add_argument("--sample-every", type=int, default=200, help="Steps between preview images")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Epochs between checkpoints")
    parser.add_argument("--resume", help="Checkpoint whose model/optimizer state should be resumed")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        parser.error("epochs and batch-size must be positive; image-size must be at least 16")
    return args


def make_config(args: argparse.Namespace) -> TrainerConfig:
    if args.resume:
        saved = read_checkpoint(args.resume)["config"]
        if saved["mode"] != args.mode:
            raise ValueError("--mode must match the resumed checkpoint")
        saved.update(device=args.device, amp=not args.no_amp)
        if args.vgg_weights:
            saved["vgg_weights"] = args.vgg_weights
        return TrainerConfig(**saved)
    perceptual = args.lambda_perceptual
    if perceptual is None:
        perceptual = 1.0 if args.mode == "paired" else 0.0
    return TrainerConfig(
        mode=args.mode,
        device=args.device,
        generator=args.generator,
        residual_blocks=args.residual_blocks,
        generator_channels=args.generator_channels,
        discriminator_channels=args.discriminator_channels,
        learning_rate=args.learning_rate,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=perceptual,
        lambda_identity=args.lambda_identity,
        lambda_cycle=args.lambda_cycle,
        amp=not args.no_amp,
        replay_size=args.replay_size,
        vgg_weights=args.vgg_weights,
    )


def learning_rate_factor(epoch: int, total_epochs: int, decay_epoch: int) -> float:
    if epoch < decay_epoch:
        return 1.0
    return max(0.0, 1.0 - (epoch - decay_epoch) / max(1, total_epochs - decay_epoch))


def save_preview(outputs: dict[str, torch.Tensor], path: Path) -> None:
    keys = ["A", "fake_B", "B"] if "fake_A" not in outputs else [
        "A", "fake_B", "recovered_A", "B", "fake_A", "recovered_B"
    ]
    batch = torch.cat([outputs[key].float().cpu() for key in keys], dim=0)
    grid = make_grid(
        batch,
        nrow=outputs["A"].shape[0],
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    save_image(grid, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = make_config(args)
    trainer = create_trainer(config)
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    output = Path(args.output)
    checkpoint_dir = output / "checkpoints"
    sample_dir = output / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    dataset = build_dataset(config.mode, args.data_root, args.image_size, "train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=trainer.device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    optimizers = [trainer.optimizer_g, trainer.optimizer_d]
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: learning_rate_factor(epoch, args.epochs, args.decay_epoch),
        )
        for optimizer in optimizers
    ]
    # Restore scheduler positions without changing model/optimizer state.
    for scheduler in schedulers:
        for _ in range(start_epoch):
            scheduler.step()

    print(json.dumps({"event": "start", "device": str(trainer.device), **asdict(config)}))
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(loader):
            metrics, outputs = trainer.train_step(batch["A"], batch["B"])
            global_step += 1
            record = {
                "epoch": epoch,
                "batch": batch_index,
                "global_step": global_step,
                **metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if global_step % args.log_every == 0 or global_step == 1:
                print(json.dumps(record), flush=True)
            if args.sample_every > 0 and global_step % args.sample_every == 0:
                save_preview(outputs, sample_dir / f"step_{global_step:08d}.png")
        for scheduler in schedulers:
            scheduler.step()
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            trainer.save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pth", epoch, global_step
            )
        trainer.save_checkpoint(checkpoint_dir / "last.pth", epoch, global_step)
        print(json.dumps({
            "event": "epoch_complete",
            "epoch": epoch,
            "seconds": round(time.monotonic() - epoch_started, 3),
        }), flush=True)


if __name__ == "__main__":
    main()
