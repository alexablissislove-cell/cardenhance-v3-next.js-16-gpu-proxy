"""Private FastAPI inference service for the Next.js/Vercel frontend."""

from __future__ import annotations

import asyncio
import io
import os
import secrets
import time
import warnings
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from engine import load_generator
from infer import restore_tensor

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "25000000"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
API_KEY = os.getenv("RESTORATION_API_KEY", "")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "")

model: torch.nn.Module | None = None
checkpoint: dict = {}
gpu_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global model, checkpoint
    if not CHECKPOINT_PATH:
        raise RuntimeError("CHECKPOINT_PATH must reference a trained .pth checkpoint")
    model, checkpoint = load_generator(CHECKPOINT_PATH, os.getenv("DEVICE", "auto"))
    yield
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Chrome RaGAN Inference",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


async def authorize(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme != "Bearer" or not secrets.compare_digest(supplied, API_KEY):
        raise HTTPException(status_code=401, detail="Valid inference credentials required")


@app.get("/health", dependencies=[Depends(authorize)])
async def health() -> dict:
    device = str(next(model.parameters()).device) if model is not None else None
    config = checkpoint.get("config", {})
    return {
        "status": "ready" if model is not None else "loading",
        "mode": checkpoint.get("mode"),
        "generator": config.get("generator"),
        "device": device,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def decode_image(payload: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit",
                    )
                rgb = image.convert("RGB")
                array = np.asarray(rgb, dtype=np.float32).copy() / 127.5 - 1.0
        return array, (width, height)
    except HTTPException:
        raise
    except (OSError, ValueError, Image.DecompressionBombWarning) as error:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image data") from error


def run_inference(
    array: np.ndarray,
    tile_size: int,
    tile_overlap: int,
    amp: bool,
) -> torch.Tensor:
    if model is None:
        raise RuntimeError("Model is not loaded")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return restore_tensor(
        tensor,
        model,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        amp=amp,
    )


@app.post("/restore", dependencies=[Depends(authorize)])
async def restore(
    file: UploadFile = File(...),
    tile_size: int = Form(default=0),
    tile_overlap: int = Form(default=32),
    amp: bool = Form(default=False),
    output_format: str = Form(default="png"),
) -> Response:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP are supported")
    if tile_size not in {0, 256, 512, 1024}:
        raise HTTPException(status_code=400, detail="Invalid tile size")
    if tile_overlap < 0 or tile_overlap > 256 or (tile_size and tile_overlap >= tile_size):
        raise HTTPException(status_code=400, detail="Invalid tile overlap")
    if output_format not in {"png", "jpeg"}:
        raise HTTPException(status_code=400, detail="Output format must be png or jpeg")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    array, (width, height) = decode_image(payload)

    started = time.perf_counter()
    async with gpu_lock:
        restored = await run_in_threadpool(
            run_inference, array, tile_size, tile_overlap, amp,
        )
    output = restored[0].permute(1, 2, 0).cpu().numpy()
    output = np.clip((output + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    result = Image.fromarray(output)
    if output_format == "png":
        result.save(buffer, format="PNG", compress_level=4)
        media_type = "image/png"
        extension = "png"
    else:
        result.save(buffer, format="JPEG", quality=95, subsampling=0)
        media_type = "image/jpeg"
        extension = "jpg"
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return Response(
        content=buffer.getvalue(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="restored.{extension}"',
            "Cache-Control": "no-store",
            "X-Checkpoint-Mode": str(checkpoint.get("mode", "unknown")),
            "X-Generator": str(checkpoint.get("config", {}).get("generator", "unknown")),
            "X-Inference-Ms": str(elapsed_ms),
            "X-Output-Width": str(width),
            "X-Output-Height": str(height),
        },
    )
