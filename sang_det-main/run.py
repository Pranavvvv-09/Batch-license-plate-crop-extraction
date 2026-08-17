#!/usr/bin/env python3
"""sang_det entry point.

    python run.py                  # dashboard + background worker (default)
    python run.py serve            # same, explicitly
    python run.py worker           # headless worker only, no web UI
    python run.py add <urls...>    # queue links from the CLI or a file
    python run.py status           # one-shot progress report
    python run.py doctor           # check the install before a long run
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import db  # noqa: E402
from app.config import CONFIG_PATH, load as load_config  # noqa: E402
from app.logging_setup import setup  # noqa: E402


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def cmd_serve(_args: argparse.Namespace) -> int:
    from app.server import run

    cfg = load_config(force=True)
    print("=" * 66)
    print("  sang_det - license plate extraction")
    print(f"  Dashboard : http://{cfg.get('server.host')}:{cfg.get('server.port')}")
    print(f"  Output    : {cfg.path('output.dir')}")
    print(f"  Config    : {CONFIG_PATH}")
    print("  Ctrl+C to stop. Progress is checkpointed; restarting resumes.")
    print("=" * 66)
    run()
    return 0


def cmd_worker(_args: argparse.Namespace) -> int:
    from app.worker import get_worker

    setup()
    db.init()
    worker = get_worker()
    worker.start()

    stopping = {"flag": False}

    def handle(_signum, _frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print("\nStopping (progress is checkpointed)...")
        worker.stop(timeout=30.0)

    signal.signal(signal.SIGINT, handle)
    with_term = getattr(signal, "SIGTERM", None)
    if with_term is not None:
        signal.signal(with_term, handle)

    print("Headless worker running. Ctrl+C to stop.")
    try:
        while not stopping["flag"] and worker.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        handle(None, None)
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    from app.server import parse_urls

    setup()
    db.init()

    blob_parts: list[str] = []
    for item in args.urls:
        path = Path(item)
        if path.exists() and path.is_file():
            blob_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        else:
            blob_parts.append(item)
    if not blob_parts and not sys.stdin.isatty():
        blob_parts.append(sys.stdin.read())

    urls = parse_urls("\n".join(blob_parts))
    if not urls:
        print("No valid http(s) links found.")
        return 1

    added, skipped = db.add_videos(urls)
    print(f"Queued {added} link(s); skipped {skipped} already known.")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    db.init()
    cfg = load_config()
    totals = db.totals()
    videos = db.list_videos()

    print(f"\nOutput folder : {cfg.path('output.dir')}")
    print(
        f"Videos        : {totals['videos']} total | {totals['done']} done | "
        f"{totals['processing']} running | {totals['pending']} pending | "
        f"{totals['errored']} errored"
    )
    print(
        f"Plates saved  : {totals['plates']:,}  "
        f"({_human_bytes(totals['output_bytes'])} on disk)"
    )
    print(
        f"Frames        : {totals['frames']:,} analysed | "
        f"{totals['detections']:,} raw detections"
    )
    print(
        f"Rejected      : {totals['rejected_size']:,} size | "
        f"{totals['rejected_blur']:,} quality | {totals['rejected_dupe']:,} duplicate"
    )

    if videos:
        print(f"\n{'ID':>4}  {'STATUS':<11} {'FRAMES':>8} {'PLATES':>8}  TITLE / ERROR")
        print("-" * 78)
        for v in videos[:60]:
            label = (v.get("error") or v.get("title") or v["url"])[:44]
            print(
                f"{v['id']:>4}  {v['status']:<11} {v['frames_processed']:>8,} "
                f"{v['plates_saved']:>8,}  {label}"
            )
    print()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Verify the install before committing to a 15-hour unattended run."""
    setup()
    cfg = load_config(force=True)
    problems = 0

    print("\nsang_det environment check")
    print("-" * 60)

    for module, label in [
        ("fastapi", "FastAPI (web UI)"),
        ("uvicorn", "Uvicorn (server)"),
        ("cv2", "OpenCV (imaging)"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("imagehash", "ImageHash (dedup)"),
        ("yt_dlp", "yt-dlp (link resolution)"),
        ("gdown", "gdown (Google Drive resolution)"),
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics YOLO"),
    ]:
        try:
            __import__(module)
            print(f"  [ok]   {label}")
        except ImportError as exc:
            problems += 1
            print(f"  [FAIL] {label}: {exc}")

    from app.frames import ffmpeg_binary

    ffmpeg = ffmpeg_binary()
    if ffmpeg:
        print(f"  [ok]   ffmpeg: {ffmpeg}")
    else:
        problems += 1
        print("  [FAIL] ffmpeg not found (pip install imageio-ffmpeg)")

    try:
        from app.detector import select_device

        device = select_device(str(cfg.get("detector.device", "auto")))
        print(f"  [ok]   compute device: {device}")
        if device == "cpu":
            print("         note: CPU-only. Expect ~5-15 analysed frames/sec.")
    except Exception as exc:
        problems += 1
        device = "cpu"
        print(f"  [FAIL] device selection: {exc}")

    # torch and torchvision must come from the same build. A +cuXXX torch with
    # a +cpu torchvision leaves NMS unimplemented on CUDA, which shows up as
    # every frame returning zero detections rather than as an obvious error.
    try:
        import torch
        import torchvision

        t_ver, tv_ver = torch.__version__, torchvision.__version__
        t_local = t_ver.partition("+")[2] or "cpu"
        tv_local = tv_ver.partition("+")[2] or "cpu"
        if t_local != tv_local:
            problems += 1
            print(f"  [FAIL] torch ({t_ver}) and torchvision ({tv_ver}) builds differ")
            print("         Detection will silently return nothing on GPU. Reinstall both:")
            print(f"           pip install torch=={t_ver.split('+')[0]}+{t_local} "
                  f"torchvision=={tv_ver.split('+')[0]}+{t_local} \\")
            print(f"               --index-url https://download.pytorch.org/whl/{t_local}")
        else:
            print(f"  [ok]   torch/torchvision builds match ({t_local})")

        if device.startswith("cuda"):
            from torchvision.ops import nms

            boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
            scores = torch.tensor([0.9], device="cuda")
            nms(boxes, scores, 0.5)
            print("  [ok]   CUDA NMS kernel available")
    except Exception as exc:
        problems += 1
        print(f"  [FAIL] CUDA NMS unavailable: {str(exc)[:110]}")
        print("         The run would produce zero detections. Fix the install above.")

    try:
        from app.detector import resolve_model_path

        model = resolve_model_path(cfg)
        print(f"  [ok]   plate model: {model.name} ({_human_bytes(model.stat().st_size)})")
    except Exception as exc:
        problems += 1
        print(f"  [FAIL] plate model: {exc}")

    out_dir = cfg.path("output.dir")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".write_test"
        probe.write_bytes(b"ok")
        probe.unlink()
        import shutil

        free_gb = shutil.disk_usage(out_dir).free / (1024 ** 3)
        threshold = float(cfg.get("storage.min_free_gb", 5.0))
        marker = "ok" if free_gb >= threshold else "WARN"
        print(f"  [{marker}]   output writable: {out_dir} ({free_gb:.1f} GB free)")
        if free_gb < threshold:
            print(f"         below the {threshold} GB guard - saving would pause immediately")
    except Exception as exc:
        problems += 1
        print(f"  [FAIL] output folder: {exc}")

    try:
        db.init()
        print(f"  [ok]   database: {db.DB_PATH}")
    except Exception as exc:
        problems += 1
        print(f"  [FAIL] database: {exc}")

    from app.labeler import NimLabeler
    labeler_client = NimLabeler(cfg)
    api_key = labeler_client.get_api_key()
    if api_key:
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  [ok]   NVIDIA NIM API key: configured ({masked_key})")
        print(f"         model: {labeler_client.get_model()}")
    else:
        print("  [WARN] NVIDIA NIM API key: not set (optional for detection, required for AI OCR labeling)")

    print("-" * 60)
    print("All checks passed. Ready to run.\n" if problems == 0
          else f"{problems} problem(s) found - see above.\n")
    return 0 if problems == 0 else 1


def cmd_label(args: argparse.Namespace) -> int:
    """Run NVIDIA NIM OCR to extract plate text and sync CSV."""
    from app import labeler
    setup()
    db.init()
    cfg = load_config()

    client = labeler.NimLabeler(cfg)
    api_key = args.api_key or client.get_api_key()
    if not api_key:
        print("Error: NVIDIA NIM API key is missing. Specify --api-key <key> or set NVIDIA_API_KEY environment variable.")
        return 1

    limit = args.limit or 500
    if args.all:
        plates, total = db.list_plates(limit=limit, offset=0)
    else:
        plates = db.get_unlabeled_plates(limit=limit)

    print(f"Found {len(plates)} plate crop(s) to scan with model '{client.get_model()}'...")
    if not plates:
        print("No plates to scan.")
        _, csv_path = labeler.export_labels_to_csv(cfg=cfg)
        print(f"Labels CSV is up to date: {csv_path}")
        return 0

    succeeded = 0
    failed = 0
    for idx, p in enumerate(plates, 1):
        plate_id = p["id"]
        filename = p["filename"]
        res = client.scan_plate_record(plate_id, api_key=api_key)
        if res.ok:
            succeeded += 1
            print(f"[{idx}/{len(plates)}] #{plate_id} {filename} -> '{res.plate_text}' ({res.latency_s}s)")
        else:
            failed += 1
            print(f"[{idx}/{len(plates)}] #{plate_id} {filename} -> FAILED: {res.error}")

    csv_dest = Path(args.csv) if args.csv else None
    _, csv_path = labeler.export_labels_to_csv(output_path=csv_dest, cfg=cfg)
    print("\nScan summary:")
    print(f"  Processed : {len(plates)}")
    print(f"  Succeeded : {succeeded}")
    print(f"  Failed    : {failed}")
    print(f"  CSV file  : {csv_path}")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Batch license plate crop extraction from video links.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="run dashboard + worker (default)")
    sub.add_parser("worker", help="run the worker headless, no web UI")
    sub.add_parser("status", help="print a one-shot progress report")
    sub.add_parser("doctor", help="check the install before a long run")

    add = sub.add_parser("add", help="queue links from arguments, a file, or stdin")
    add.add_argument("urls", nargs="*", help="URLs, or a path to a file of URLs")

    lbl = sub.add_parser("label", help="scan saved plate crops to text using NVIDIA NIM OCR")
    lbl.add_argument("--all", action="store_true", help="re-scan all plates including already labeled ones")
    lbl.add_argument("--limit", type=int, default=500, help="max plates to scan (default: 500)")
    lbl.add_argument("--api-key", help="override NVIDIA NIM API key")
    lbl.add_argument("--csv", help="output CSV destination path (default: data/labels.csv)")

    args = parser.parse_args()
    handlers = {
        None: cmd_serve,
        "serve": cmd_serve,
        "worker": cmd_worker,
        "add": cmd_add,
        "status": cmd_status,
        "doctor": cmd_doctor,
        "label": cmd_label,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

