# sang_det

**Batch license plate crop extraction from video links.**

Paste 20+ video links (YouTube or direct video URLs), walk away for 10–15 hours,
and come back to a folder containing nothing but clean, deduplicated, tightly
cropped license plate images.

Runs entirely on your own machine. No cloud, no paid APIs, no subscriptions.

---

## Table of contents

- [What it does](#what-it-does)
- [What it deliberately does *not* do](#what-it-deliberately-does-not-do)
- [Install](#install)
- [Quick start](#quick-start)
- [Supabase & 24/7 Cloud Deployment](#supabase--247-cloud-deployment)
- [How the pipeline works](#how-the-pipeline-works)
- [Output](#output)
- [Configuration](#configuration)
- [Tuning guide](#tuning-guide)
- [Reliability: surviving a 15-hour run](#reliability-surviving-a-15-hour-run)
- [Dashboard](#dashboard)
- [CLI reference](#cli-reference)
- [Performance & throughput planning](#performance--throughput-planning)
- [Project layout](#project-layout)
- [HTTP API](#http-api)
- [Troubleshooting](#troubleshooting)
- [Design notes](#design-notes)
- [Legal & ethical use](#legal--ethical-use)


---

## What it does

| | |
|---|---|
| **Input** | A textarea of video links. YouTube and direct `.mp4`/`.m3u8` URLs can be mixed freely. Lengths from a few minutes to 1.5+ hours, sparse or dense traffic — no per-video configuration. |
| **Processing** | Streams each video frame-by-frame over the network, samples at a fixed interval, runs a dedicated license-plate detector, then filters hard on confidence, box size, aspect ratio, sharpness and perceptual-hash similarity. |
| **Output** | One folder of cropped plate JPEGs, each traceable to its source video and timestamp via its filename and a JSONL manifest. |
| **Runtime** | A background worker drains the queue independent of the browser. State lives in SQLite, so an interrupted run resumes mid-video instead of restarting. |
| **Hardware** | Auto-selects CUDA → Apple MPS → CPU. Nothing to configure. |

## What it deliberately does *not* do

These are **non-goals**, not missing features:

- **No OCR.** The deliverable is plate *images*, not decoded text. Skipping
  recognition removes an entire class of failure and a large chunk of runtime.
- **No full-frame saves.** Only the detected plate region is written — never a
  full frame, never a vehicle crop, never background or bystander imagery.
- **No video storage.** Frames are pulled off the wire, used, and dropped. A
  1.5-hour 1080p video costs a few hundred MB of *transfer* and **zero** disk.
- **No cloud, no paid infrastructure.** Everything runs on `localhost`.

---

## Install

**Requirements:** Python 3.9+ and an internet connection for the first run
(model weights download once, ~6 MB, then it's fully offline).

```bash
cd sang_det
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### GPU acceleration

<details>
<summary><b>Apple Silicon (M1/M2/M3/M4)</b> — nothing to do</summary>

The default `pip install torch` wheel includes MPS support. The tool detects and
uses it automatically. Verify with `python run.py doctor` → `compute device: mps`.
</details>

<details>
<summary><b>NVIDIA GPU</b> — install the CUDA build <i>last</i></summary>

Plain `pip install -r requirements.txt` resolves the **CPU-only** PyTorch wheel
from PyPI. Installing the CUDA build *first* does not help — pip's resolver
upgrades torch while installing `ultralytics` and silently replaces it with the
CPU wheel.

Install everything else first, then the CUDA build **last**, pinned with an
explicit `+cuXXX` local version tag:

```bash
pip install -r requirements.txt

# then, matching the versions the previous step settled on:
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

> The `+cu124` tag is **required**. `pip install torch==2.6.0` treats an already
> installed `2.6.0+cpu` as satisfying the requirement and does nothing.

Pick the `cuXXX` matching your driver (`cu118`, `cu124`, `cu126` — see
[pytorch.org](https://pytorch.org/get-started/locally/)). Verify with
`python run.py doctor` → `compute device: cuda`.
</details>

<details>
<summary><b>No GPU</b> — it still works</summary>

CPU inference runs at roughly 5–15 analysed frames/sec. At the default 1 fps
sampling that is still **5–15× faster than real time**, so a 15-hour batch of
source footage completes in 1–3 hours of compute.
</details>

### Verify the install

```bash
python run.py doctor
```

```
sang_det environment check
------------------------------------------------------------
  [ok]   FastAPI (web UI)
  [ok]   OpenCV (imaging)
  [ok]   ImageHash (dedup)
  [ok]   yt-dlp (link resolution)
  [ok]   PyTorch
  [ok]   Ultralytics YOLO
  [ok]   ffmpeg: .../imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe
  [ok]   compute device: cuda
  [ok]   torch/torchvision builds match (cu124)
  [ok]   CUDA NMS kernel available
  [ok]   plate model: yolov11-license-plate-detection.pt (5.2 MB)
  [ok]   output writable: .../data/output (167.2 GB free)
  [ok]   database: .../data/sang_det.db
------------------------------------------------------------
All checks passed. Ready to run.
```

> **ffmpeg is bundled.** `imageio-ffmpeg` ships a static binary, so there is no
> system install step. If you already have ffmpeg on `PATH`, that one is used.

---

## Quick start

```bash
python run.py
```

Open **<http://127.0.0.1:8000>**, paste your links into the textarea, and press
**Queue videos**. Processing starts immediately.

**You can now close the browser.** The worker is a background thread, not a
page script — the batch keeps running. Reopen the dashboard any time to check
progress; polling the UI does not disturb the worker.

For a long unattended run you may prefer no web server at all:

```bash
python run.py add links.txt     # queue a file of URLs
python run.py worker            # headless; Ctrl+C is safe, progress is saved
python run.py status            # check in from another terminal
```

---

## Supabase & 24/7 Cloud Deployment

Run your batch jobs for **10,000+ images/videos continuously in the cloud** without needing your laptop to stay powered on.

### 1. Set Up Supabase (PostgreSQL & Storage)

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to the **SQL Editor** in your Supabase project dashboard.
3. Open [`supabase_schema.sql`](file:///c:/Users/wayku/Downloads/sang_det-main/sang_det-main/supabase_schema.sql) from this repository, paste the entire SQL content, and click **RUN**.
   - This creates the `videos`, `plates`, `events`, `state` tables, indexes, and sets up the `plates` Supabase Storage bucket with public access.
4. Copy your credentials from **Project Settings**:
   - **Database Connection String** (`Project Settings -> Database -> Connection string (URI / Pooler)`) -> `DATABASE_URL`
   - **Project URL & API Key** (`Project Settings -> API`) -> `SUPABASE_URL` and `SUPABASE_KEY`

---

### 2. Deploy the Continuous 24/7 Worker

Since batch processing 10k images requires continuous background execution and PyTorch/YOLO/OpenCV (>1.5GB runtime), deploy the containerized backend to a continuous cloud platform like **Railway**, **Render**, or **Docker VPS**:

#### Option A: 1-Click Deploy on Railway (Recommended)
1. Push your repository to GitHub.
2. Go to [railway.app](https://railway.app), click **New Project** -> **Deploy from GitHub repo**.
3. In **Variables**, add:
   ```env
   DATABASE_URL=postgresql://postgres.xxx:[PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
   SUPABASE_URL=https://xxx.supabase.co
   SUPABASE_KEY=your_supabase_api_key
   SUPABASE_BUCKET=plates
   NVIDIA_API_KEY=nvapi-... (optional, for free AI OCR)
   ```
4. Railway automatically builds the [`Dockerfile`](file:///c:/Users/wayku/Downloads/sang_det-main/sang_det-main/Dockerfile) and runs the FastAPI server + background worker 24/7.

#### Option B: 1-Click Deploy on Render
1. Push your repository to GitHub.
2. Go to [render.com](https://render.com), click **New** -> **Blueprint**, and select your repo (using [`render.yaml`](file:///c:/Users/wayku/Downloads/sang_det-main/sang_det-main/render.yaml)).
3. Fill in your `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_KEY`, and `NVIDIA_API_KEY` in the environment variables.

#### Option C: Run with Docker Compose Locally or on a VPS
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
docker compose up -d --build
```

---

### 3. Optional: Host Frontend UI on Vercel
If you want to host the frontend web interface on Vercel:
1. In [`vercel.json`](file:///c:/Users/wayku/Downloads/sang_det-main/sang_det-main/vercel.json), replace `YOUR_RAILWAY_OR_RENDER_BACKEND_URL` with your Railway/Render deployment URL.
2. Run `vercel` or link your GitHub repository on Vercel.
3. The dashboard UI will load from Vercel's global CDN while all heavy AI detection and background jobs run 24/7 on your cloud worker connected to Supabase!

---

## How the pipeline works


Each video is an isolated job. Ten stages, cheapest filters first, so expensive
work only happens on candidates that already passed:

```
 1. Resolve stream ────── yt-dlp for YouTube; passthrough for direct URLs.
                          Never downloads the file — resolves a playable
                          stream and hands it to the decoder.

 2. Sample frames ─────── ffmpeg decodes with an `fps=N` filter, so only
                          sampled frames cross the pipe. Default 1 fps.

 3. Detect plates ─────── A dedicated YOLO license-plate detector (not OCR
                          over full frames). Frames are batched for GPU
                          throughput.

 4. Confidence filter ─── Drops detections below `detector.confidence`.
                          ── suppresses false positives

 5. Size / shape filter ─ Drops boxes below the minimum width/height, outside
                          the plate aspect-ratio band, or covering an absurd
                          share of the frame (a whole-vehicle false positive).
                          ── runs on the box, before any pixels are copied

 6. Crop ──────────────── Tight crop to the box + a few px of context.
                          ── plate region only, never a frame or a vehicle

 7. Quality gate ──────── Laplacian-variance blur detection, plus contrast and
                          flat-region checks for glare and heavy occlusion.
                          ── rejects unreadable crops

 8. Deduplicate ───────── 64-bit perceptual hash compared against a rolling
                          window of recent crops.
                          ── the same plate isn't saved 50 times

 9. Save ──────────────── Atomic JPEG write + provenance row.
                          ── no zero-byte or truncated files, ever

10. (no OCR) ─────────────  Skipped by design.
```

### Why dedup uses two signals, and what it can't do

A vehicle passing a camera puts the same plate into 10–40 consecutive sampled
frames. Naive saving floods the dataset with near-identical crops.

Hashing the *image* rather than OCR'd text means dedup works on plates that are
too small, angled, or motion-blurred to read — and it tolerates the scale,
brightness and crop drift you get as a car approaches.

**An honest limitation, measured rather than assumed.** Perceptual hashing alone
is a weak discriminator *for license plates specifically*. Every plate is the
same rectangle, in the same layout, in the same palette; the part that differs —
the characters — is a small fraction of the image that a perceptual hash largely
discards. Measured across the tool's own output on real detector boxes:

| variant | same-plate distance | different-plate distance | verdict |
|---|---|---|---|
| pHash-8 | 8 – 34 | 10 – 36 | bands overlap |
| pHash-16 | 38 – 122 | 76 – 138 | best, still overlaps |
| dHash-16 | 18 – 113 | 41 – 102 | bands overlap |
| aHash / wHash / dHash-32 | — | — | worse |

There is **no threshold** that catches every repeat without also merging
genuinely distinct plates. Anyone claiming otherwise for a no-OCR pipeline has
not measured it.

So dedup gates on two independent signals:

1. **Visual** — Hamming distance ≤ `dedup.hash_distance`.
2. **Temporal** — the two crops are also within `dedup.window_seconds` of each
   other *in the source video*. A vehicle passes in seconds, so real repeats are
   always close in time. This is what makes a loose visual threshold safe: it
   prevents matching against a similar-looking car from minutes earlier, which
   is where false merges come from.

**Defaults are deliberately conservative** (`hash_distance: 7`,
`window_seconds: 20`). The cost asymmetry drives this: surplus near-duplicates
are dataset bloat you can filter later, but two distinct plates merged into one
is unrecoverable data loss. When uncertain, the tool keeps the crop.

If you want aggressive dedup and accept the risk, raise `hash_distance` toward
11–15 while *keeping* `window_seconds` low (10–20). Raising the distance with
`window_seconds: 0` is the one combination that will quietly cost you plates.

> `dedup.hash_size` is intentionally *not* editable from the dashboard — it is
> coupled to `hash_distance`, and changing one without the other silently breaks
> dedup. Edit `config.yaml` if you need to.

---

## Output

### Files

Everything lands in one flat folder (default `data/output/`):

```
v0007_traffic-cam-highway-3_t000987654_f0001234_00.jpg
│     │                     │          │        └─ detection index within the frame
│     │                     │          └────────── sampled frame index
│     │                     └───────────────────── timestamp in ms (987.654 s)
│     └─────────────────────────────────────────── source video title (slug)
└───────────────────────────────────────────────── video ID in the database
```

Every crop is traceable to its **source video and moment** from the filename
alone. Collisions are impossible — names are checked against both an in-memory
set and the filesystem, with a numeric suffix as a fallback.

### Manifest

`data/manifest.jsonl` — one JSON object per saved crop:

```json
{"file":"v0007_..._00.jpg","video_id":7,"source_title":"Traffic Cam Highway 3",
 "timestamp_s":987.654,"frame_index":1234,"confidence":0.8871,
 "box_xyxy":[1180,612,1298,652],"crop_wh":[124,46],"blur_score":216.4,
 "phash":"c3a1...","bytes":3678,"saved_at":1755265...}
```

Load it in one line:

```python
import pandas as pd
df = pd.read_json("data/manifest.jsonl", lines=True)
```

The same data is queryable in SQLite (`data/sang_det.db`, table `plates`).

### Integrity guarantees

| Excluded from the output folder | Enforced by |
|---|---|
| Full video frames | Only `crop()` output is ever written |
| Vehicle bodies / background / bystanders | Tight box + max 3 px padding |
| Non-plate false positives | Confidence, aspect-ratio and max-area filters |
| Near-duplicate repeats | Perceptual-hash rolling window |
| Blurred / occluded crops | Laplacian variance, contrast, flat-region checks |
| Zero-byte or truncated files | Atomic write (see below) |

**Atomic writes:** each JPEG is encoded in memory, written to a `.tmp` file on
the same volume, flushed, `fsync`'d, then `os.replace()`d into place — an
atomic rename on both POSIX and Windows. A crash or power cut can leave a stray
`.tmp` (swept on next start), but never a corrupt file in the output folder.
Size is verified after the rename; a zero-byte result is deleted, not recorded.

---

## Configuration

All tunables live in **`config.yaml`**, documented inline. Nothing is buried in
logic. The worker re-reads the file between videos, so edits apply to a running
batch without a restart.

The dashboard's **Tuning** panel writes to the same file using targeted line
replacements, so your comments survive.

| Requirement | Key | Default |
|---|---|---|
| Frame sampling interval | `sampling.fps` | `1.0` |
| Detection confidence threshold | `detector.confidence` | `0.35` |
| Minimum plate box size | `filters.min_box_width` / `min_box_height` | `60` / `20` |
| Dedup strictness (hash distance) | `dedup.hash_distance` | `7` |
| Dedup comparison window (crops) | `dedup.window_size` | `240` |
| Dedup comparison window (video seconds) | `dedup.window_seconds` | `20.0` |
| JPEG output quality | `output.jpeg_quality` | `92` |
| Min free disk before pausing | `storage.min_free_gb` | `5.0` |

Other useful keys:

| Key | Default | Purpose |
|---|---|---|
| `sampling.max_frame_width` | `1920` | Downscale 4K sources before detection |
| `detector.imgsz` | `960` | Inference resolution; higher catches smaller plates |
| `detector.batch_size` | `8` | Frames per inference call (GPU throughput) |
| `detector.device` | `auto` | `auto` / `cuda` / `mps` / `cpu` |
| `filters.max_area_fraction` | `0.25` | Rejects whole-vehicle false positives |
| `quality.min_laplacian_variance` | `45.0` | Blur rejection threshold |
| `dedup.scope` | `video` | `video` (reset per video) or `global` |
| `runtime.concurrency` | `1` | Videos in flight at once |
| `runtime.max_retries` | `2` | Retries before a video is marked errored |
| `runtime.max_plates_per_video` | `0` | Per-video cap; `0` = unlimited |
| `ingest.max_height` | `1080` | Resolution cap requested from yt-dlp |

Point at a different config with `SANG_DET_CONFIG=/path/to/config.yaml`.

---

## Tuning guide

Start with the defaults. Run **one** representative video, look at the output
folder and the dashboard's reject counters, then adjust.

<details>
<summary><b>Too few plates saved</b></summary>

| Symptom (dashboard) | Change |
|---|---|
| Low **raw detections** | Lower `detector.confidence` to `0.25`; raise `detector.imgsz` to `1280` |
| High **size rejects** | Lower `filters.min_box_width` / `min_box_height` — plates are far from the camera |
| High **quality rejects** | Lower `quality.min_laplacian_variance` to `25`; fast traffic is genuinely motion-blurred |
| High **duplicates dropped** but few saved | Lower `dedup.hash_distance` to `5`, or `dedup.window_seconds` to `8` |
| Everything low | Raise `sampling.fps` to `2.0` — you're missing cars between samples |
</details>

<details>
<summary><b>Junk in the output folder</b></summary>

| Symptom | Change |
|---|---|
| Non-plate crops | Raise `detector.confidence` to `0.5`+ |
| Tiny unreadable crops | Raise `filters.min_box_width` to `90`, `min_box_height` to `30` |
| Blurry crops | Raise `quality.min_laplacian_variance` to `70`+ |
| Same plate repeatedly | Raise `dedup.hash_distance` to `11`; keep `window_seconds` at `20` so it stays safe |
| Whole vehicles / signs | Lower `filters.max_area_fraction` to `0.10`; tighten the aspect band |
</details>

<details>
<summary><b>Running too slowly</b></summary>

1. Confirm `python run.py doctor` reports `cuda` or `mps`, not `cpu`.
2. Lower `sampling.fps` to `0.5` — halves the work, and at typical traffic
   speeds you still catch each vehicle.
3. Lower `detector.imgsz` to `640`.
4. Raise `detector.batch_size` to `16` on a GPU with headroom.
5. Lower `ingest.max_height` to `720` if you are network-bound.
</details>

<details>
<summary><b>Aiming at 100,000+ plates</b></summary>

Output volume is roughly:

```
plates ≈ hours_of_footage × 3600 × sampling.fps × unique_plates_per_sampled_frame
```

Dense multi-lane traffic yields ~0.5–2 unique plates per sampled frame after
dedup; sparse roads yield ~0.02–0.1. To reach six figures, prioritise **dense
footage** (busy intersections, highway overpasses, toll plazas) over more hours
of empty road — density beats duration by an order of magnitude. Raising
`sampling.fps` to `2.0` roughly doubles throughput at double the compute.
</details>

---

## Reliability: surviving a 15-hour run

**One bad link cannot stop the batch.** Every video is an isolated job wrapped
in a catch-all. Dead links, geo-blocks, corrupt streams, mid-video network
drops, and even unexpected internal errors are caught, logged, and the worker
moves on.

- Transient failures are retried up to `runtime.max_retries` with backoff.
- Permanent failures (private, removed, members-only, region-locked) are
  recognised and *not* retried — no wasted hours.
- Whatever the failure, the error text is stored and surfaced per-row in the
  dashboard. Fix the cause and hit **retry errored** to requeue them all.

**Interrupted runs resume, they don't restart.** Progress is checkpointed to
SQLite every 30 analysed frames (`runtime.checkpoint_every_n_frames`), including
the position within the current video. On restart:

- Jobs left in `processing` by a hard kill are reclaimed to `pending`.
- Each resumes from its last checkpoint via ffmpeg's `-ss` seek.
- Ctrl+C is a *clean* stop — the current video is checkpointed and requeued.

> For direct URLs the resume seek is input-side and instant. For YouTube pipes
> ffmpeg must decode-and-discard up to the resume point, which is still far
> cheaper than re-running detection over it.

**Disk-space guard.** Free space is re-checked every 50 saves or 60 seconds. If
it drops below `storage.min_free_gb`, saving **pauses** — the worker keeps the
queue intact, the dashboard turns the disk pill red and shows `disk-full`, and
saving resumes automatically once space is freed. It does not crash, and it does
not write partial files into a full volume. An `ENOSPC` mid-write also trips the
guard rather than failing the run.

**Memory stays flat.** Only `batch_size` frames are resident at a time, and
sampled frames never accumulate — a 1.5-hour video costs the same RAM as a
1-minute one.

---

## Dashboard

<http://127.0.0.1:8000> — a monitor, not a driver.

- **Running totals** — plates saved, frames analysed, raw detections, the
  percentage of detections kept, and a breakdown of *why* the rest were dropped
  (duplicate / quality / size). This is your tuning feedback loop.
- **Per-video rows** — status, frames processed, plates saved, duplicates
  dropped, elapsed time, a progress bar against video duration, and the full
  error message for anything that failed.
- **Per-video actions** — retry, redo from scratch, cancel, remove.
- **Live crop previews** — the most recent saved plates, so you can eyeball
  output quality minutes into a run rather than hours.
- **Activity log** — recent events with levels.
- **Tuning panel** — the tunables above, written straight to `config.yaml`.
- **Pause / resume** — pauses between frames; the current video is checkpointed.

Header pills show worker state, the active device and model, and free disk.

---

## CLI reference

| Command | What it does |
|---|---|
| `python run.py` | Dashboard + background worker (default) |
| `python run.py serve` | Same, explicitly |
| `python run.py worker` | Headless worker, no web UI |
| `python run.py add <url> [...]` | Queue links from arguments |
| `python run.py add links.txt` | Queue links from a file |
| `cat links.txt \| python run.py add` | Queue links from stdin |
| `python run.py status` | One-shot progress report |
| `python run.py doctor` | Verify the install before a long run |

`add` accepts messy input — it extracts `http(s)` links from pasted text,
tolerating bullets, quotes, commas and surrounding prose.

---

## Performance & throughput planning

Measured at the defaults (1 fps sampling, `imgsz` 960, batch 8):

| Device | Analysed frames/sec | 1 hour of footage takes |
|---|---|---|
| NVIDIA (CUDA) | ~40–120 | ~30–90 seconds |
| Apple MPS | ~25–60 | ~1–2.5 minutes |
| CPU | ~5–15 | ~4–12 minutes |

At 1 fps, **compute is rarely the bottleneck — network throughput is.** Streaming
a 1.5-hour 1080p video is the dominant cost, which is why `ingest.max_height`
defaults to 1080 rather than 4K.

**Disk:** a typical plate crop is 3–8 KB. 100,000 plates ≈ **300–800 MB**. The
5 GB default guard is generous headroom.

---

## Project layout

```
sang_det/
├── config.yaml            All tunables, documented inline
├── requirements.txt
├── run.py                 CLI entry point
├── README.md
├── app/
│   ├── config.py          YAML loading, hot reload, comment-preserving writes
│   ├── db.py              SQLite job store, checkpoints, provenance
│   ├── logging_setup.py   Rotating file + console logging
│   ├── resolver.py        yt-dlp / direct-URL → playable stream
│   ├── frames.py          ffmpeg pipe sampler + OpenCV fallback
│   ├── detector.py        YOLO plate detector, device + weight resolution
│   ├── quality.py         Geometry filters, cropping, blur/contrast gates
│   ├── dedup.py           Perceptual-hash rolling-window dedup
│   ├── storage.py         Atomic JPEG writes, disk guard, manifest
│   ├── pipeline.py        Per-video orchestration
│   ├── worker.py          Background queue worker, retries, isolation
│   └── server.py          FastAPI app + control/monitor API
├── static/index.html      Dashboard (vanilla JS, no build step)
├── models/                Detector weights (downloaded on first run)
└── data/
    ├── output/            ← your plate crops
    ├── manifest.jsonl     Provenance, one line per crop
    ├── sang_det.db        Queue + progress + metadata
    └── logs/
```

---

## HTTP API

The dashboard is a thin client over these endpoints — script the tool if you'd
rather not use the UI.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/videos` | Queue links — `{"urls": "url1\nurl2"}` |
| `GET` | `/api/status` | Worker state, totals, per-video rows, config |
| `GET` | `/api/events?limit=N` | Recent activity log |
| `GET` | `/api/plates/recent?limit=N` | Recently saved crop metadata |
| `GET` | `/api/plates/file/{filename}` | Fetch one saved crop |
| `POST` | `/api/control/{pause\|resume\|start\|retry-errors}` | Worker control |
| `POST` | `/api/videos/{id}/{retry\|restart\|cancel\|delete}` | Per-video actions |
| `GET` / `PUT` | `/api/config` | Read / update tunables |
| `GET` | `/api/health` | Liveness check |

```bash
curl -X POST localhost:8000/api/videos \
     -H 'Content-Type: application/json' \
     -d '{"urls":"https://youtu.be/aaa\nhttps://youtu.be/bbb"}'
```

---

## Troubleshooting

<details>
<summary><b>"No license plate model found"</b></summary>

Weights download from the Hugging Face Hub on first run. If you're offline or
behind a proxy, download any Ultralytics-format plate detector (`.pt`) manually
and drop it in `models/` — it is picked up automatically on restart.
</details>

<details>
<summary><b>Videos finish as "done" but nothing is saved and detections are 0</b></summary>

Almost always a **torch / torchvision build mismatch** — e.g. `torch 2.6.0+cu124`
with `torchvision 0.21.0+cpu`. A CPU-only torchvision cannot provide the CUDA
NMS kernel, so every frame throws inside inference and returns no boxes.

Run `python run.py doctor`; it checks for this explicitly and prints the exact
reinstall command. The tool also self-defends in two ways: start-up warm-up runs
a real detection and **falls back to CPU** if the GPU path is broken (you'll see
a loud error in the log), and a batch where *every* frame fails now marks the
video **errored** instead of quietly "done".

If detections are genuinely 0 on a working install, the footage may simply have
no readable plates — lower `detector.confidence` and check the live crop
previews on the dashboard.
</details>

<details>
<summary><b>A YouTube link fails with "Sign in to confirm" or "Video unavailable"</b></summary>

Age-restricted, private, or region-locked. These are treated as **permanent**
failures and are not retried. For age-restricted videos, upgrade yt-dlp
(`pip install -U yt-dlp`) and retry — extractors change often.
</details>

<details>
<summary><b>All videos error with "could not determine video dimensions"</b></summary>

The stream couldn't be probed. Usually the host doesn't support HTTP Range
requests. Check the URL plays in VLC; if it does, set
`ingest.prefer_ffmpeg: false` to force the OpenCV decoder path.
</details>

<details>
<summary><b>Dashboard says "Lost connection to the worker"</b></summary>

`run.py` has exited. Restart it — queued videos and mid-video positions are in
SQLite and resume automatically.
</details>

<details>
<summary><b>Worker stuck on "disk-full"</b></summary>

Free space fell below `storage.min_free_gb`. Free some space (or lower the
threshold) — the worker re-checks every 15 seconds and resumes on its own.
</details>

<details>
<summary><b>Output has near-duplicates anyway</b></summary>

Expected, and deliberate — see
[Why dedup uses two signals](#why-dedup-uses-two-signals-and-what-it-cant-do).
The same plate at a visibly different angle or distance is a genuinely different
image, and the defaults favour keeping it over risking the loss of a distinct
plate. To merge harder: raise `dedup.hash_distance` to 11–15 while keeping
`dedup.window_seconds` at 10–20. Do **not** raise the distance with
`window_seconds: 0` — that is the combination that silently drops real plates.
</details>

<details>
<summary><b>Starting over</b></summary>

Delete `data/sang_det.db` to clear the queue. Delete `data/output/` and
`data/manifest.jsonl` to clear results. Both are recreated on next start.
</details>

---

## Design notes

A few decisions worth stating explicitly, since they were choices rather than
defaults:

**Stream, don't download.** yt-dlp resolves a playable URL and ffmpeg decodes
with an `fps=N` filter, so only sampled frames cross the pipe. This is what
makes 20+ hour-long videos cost effectively zero disk.

**ffmpeg primary, OpenCV fallback.** The ffmpeg pipe gives exact interval
control, reliable resume seeking and stable memory. OpenCV covers the cases
ffmpeg can't establish. `imageio-ffmpeg` bundles the binary so there is no
system dependency, and `ffprobe` is never required — geometry comes from yt-dlp
metadata, an OpenCV header read, or ffmpeg's own stream report, in that order.

**Cheap filters first.** Size and aspect-ratio checks run on the bounding box
before any pixels are copied; blur detection runs on the crop; hashing runs only
on crops that already passed. Rejected candidates cost almost nothing.

**Batched inference.** Frames are grouped into `batch_size` chunks before hitting
the GPU. On CUDA OOM the detector automatically drops to frame-by-frame rather
than failing the video.

**The UI is not the app.** The worker is a daemon thread over a SQLite queue.
The browser is a monitor; closing it changes nothing. This is what allows a
15-hour unattended run.

**Counters are diagnostics, not decoration.** Per-video and batch-wide reject
reasons (size / quality / duplicate) are recorded specifically so you can tune
thresholds from evidence instead of guesswork.

---

## Legal & ethical use

License plates are personally identifying information in many jurisdictions
(GDPR, CCPA, and others). You are responsible for the legality of what you
collect and how you use it.

Before running a batch, consider: whether you have the right to process the
source footage, whether plate collection is lawful in your jurisdiction and for
your purpose, and how long you need to retain the results.

The tool is built to minimise collateral capture — it saves **only** the plate
region, never full frames, vehicles, faces, or surroundings — but that is a
technical safeguard, not legal advice.

---

*Runs locally. Costs nothing to operate. No data leaves your machine.*
