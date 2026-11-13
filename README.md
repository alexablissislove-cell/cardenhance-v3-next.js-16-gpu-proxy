# Chrome / RaGAN Studio

A Next.js 16 restoration workspace for Vercel, backed by a private FastAPI + PyTorch inference service. It supports paired RaGAN/RRDB checkpoints and unpaired CycleGAN checkpoints produced by the companion training pipeline.

The browser uploads an image to Next.js, the Next.js route validates and forwards it to the GPU service, and the restored PNG or JPEG streams back for before/after comparison and download. The application does not persist uploaded images.

## Architecture

```text
Browser
  │  JPEG / PNG / WebP, up to 4 MB by default
  ▼
Next.js on Vercel
  ├── /api/health   → authenticated GPU health check
  └── /api/restore  → validation + authenticated streaming proxy
                         ▼
                  FastAPI on a GPU host
                  checkpoint loading + tiled PyTorch inference
```

PyTorch training and model inference do **not** run in a Vercel function. Deploy `inference/` separately on infrastructure with a compatible GPU and keep its endpoint private with HTTPS and bearer authentication.

## Prerequisites

- Node.js 20 or newer
- A trained `.pth` checkpoint from the Chrome RaGAN pipeline
- Python 3.12 and PyTorch compatible with the inference host
- A CUDA GPU is recommended for practical restoration times; CPU inference is supported but can be slow

No pretrained restoration checkpoint is included. The interface intentionally disables restoration until a healthy inference service is configured.

## Run the inference service

From the project root:

```bash
cd inference
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export CHECKPOINT_PATH=/absolute/path/to/checkpoint.pth
export RESTORATION_API_KEY='replace-with-a-long-random-secret'
export DEVICE=auto               # auto, cuda, or cpu
export MAX_UPLOAD_BYTES=25000000
export MAX_IMAGE_PIXELS=40000000
uvicorn server:app --host 0.0.0.0 --port 8000
```

For production, use a process manager or container orchestrator, TLS termination, network restrictions, and a host-specific CUDA-enabled PyTorch installation. The service serializes inference jobs with an async lock to avoid concurrent GPU memory spikes. One worker per GPU is the intended starting point.

### Inference container

```bash
docker build -t chrome-ragan-inference ./inference
docker run --rm --gpus all \
  -p 8000:8000 \
  -v /absolute/path/to/checkpoints:/checkpoints:ro \
  -e CHECKPOINT_PATH=/checkpoints/model.pth \
  -e RESTORATION_API_KEY='replace-with-a-long-random-secret' \
  chrome-ragan-inference
```

The container host needs NVIDIA Container Toolkit and a PyTorch build compatible with its driver. Adjust the image's PyTorch installation for your CUDA version when necessary.

Health check:

```bash
curl -H "Authorization: Bearer $RESTORATION_API_KEY" http://localhost:8000/health
```

## Run the Next.js app locally

In a second terminal, from the project root:

```bash
cp .env.example .env.local
# Set RESTORATION_API_URL and the same RESTORATION_API_KEY in .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`.

```text
RESTORATION_API_URL=http://localhost:8000
RESTORATION_API_KEY=replace-with-a-long-random-secret
MAX_IMAGE_BYTES=4000000
```

All three variables are server-only. Never prefix the inference URL or key with `NEXT_PUBLIC_`.

## Deploy to Vercel

1. Deploy `inference/` to your GPU provider and verify `/health` returns `"status": "ready"` over HTTPS.
2. Import this project into Vercel. The framework is Next.js; use the default build command `npm run build` and output settings.
3. Add these environment variables in **Project → Settings → Environment Variables**:
   - `RESTORATION_API_URL=https://your-gpu-host.example` — no trailing `/restore` or `/health`
   - `RESTORATION_API_KEY=<the same long random secret configured on FastAPI>`
   - `MAX_IMAGE_BYTES=4000000`
4. Deploy or redeploy the Vercel project, then use the service-status control in the header to confirm the connection.

`src/app/api/restore/route.ts` declares `maxDuration = 300`; the actual maximum depends on the Vercel plan. Also configure the GPU host and any reverse proxy to allow the expected inference duration and response size.

## Upload and security limits

- The default 4,000,000-byte application cap leaves room for multipart overhead below Vercel's 4.5 MB function request-body limit. `MAX_IMAGE_BYTES` cannot be configured above 4,000,000 in this proxy.
- Accepted uploads are JPEG, PNG, and WebP. FastAPI also verifies decoding and enforces its own upload-byte and decoded-pixel limits.
- Only the required image and inference settings are forwarded. Browser cookies and authorization headers are not forwarded, and upstream cookies are not returned.
- `RESTORATION_API_KEY` is read only by server routes. Use HTTPS, rotate the key if exposed, and restrict inbound access to the inference service where possible.
- This app does not write images to a database, object storage, or `localStorage`. Review logging and retention settings on your GPU host and proxy independently.
- For scans larger than the Vercel request limit, implement a separate authenticated object-storage upload flow rather than raising this proxy's cap.

## Restoration settings

- **Full frame** processes the complete source at once.
- **256 / 512 / 1024 px tiles** bound GPU memory use for larger frames.
- **Tile overlap** supplies neighboring context and blends tile seams.
- **Mixed precision** is passed to the inference service and applies only when CUDA autocast is available.
- **PNG** preserves lossless output; **JPEG** produces a smaller download at quality 95.

Output retains the source width and height. Restoration quality and structure preservation depend on the supplied checkpoint and training data.

## Verification

```bash
npm run lint
npm test
npm run build
python -m compileall inference
```

The route tests cover configuration, image validation, upload limits, server-only authorization, forwarded form fields, streamed image responses, and upstream failures.

For a production smoke test, use a real trained checkpoint and a known validation image. A mock inference service is useful for interface testing only and must not be used to judge restoration quality.
