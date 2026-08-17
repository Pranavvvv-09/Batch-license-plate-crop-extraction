"""Local web app: submission form, monitoring dashboard, control API.

The UI is a monitor, not a driver. The worker runs in the background whether
or not a browser is open; closing the tab has no effect on the batch.
"""

from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as config_module
from . import db
from . import labeler
from . import storage
from .config import ROOT, load as load_config

from .logging_setup import get
from .worker import get_worker

log = get("server")

STATIC_DIR = ROOT / "static"

from .gdrive import is_gdrive_url, parse_gdrive_url, scan_gdrive_folder

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class SubmitPayload(BaseModel):
    urls: str = Field(default="", description="Newline/whitespace separated links")


class GDriveScanPayload(BaseModel):
    url: str = Field(..., description="Google Drive folder or file URL")


class GDriveQueuePayload(BaseModel):
    url: str = Field(default="", description="Google Drive folder or file URL")
    file_ids: list[str] = Field(default_factory=list, description="Optional specific file IDs to queue")
    only_images: bool = Field(default=True, description="Queue only images if True")


class ConfigPayload(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)


def parse_urls(blob: str, expand_gdrive: bool = True) -> list[str]:
    """Extract links from pasted text; expands Google Drive folders into file links."""
    found: list[str] = []
    seen: set[str] = set()
    for line in (blob or "").splitlines():
        for match in URL_RE.findall(line):
            url = match.strip().strip(",;\"'<>()[]")
            if not url or url in seen:
                continue

            if expand_gdrive and is_gdrive_url(url):
                kind, _ = parse_gdrive_url(url)
                if kind == "folder":
                    try:
                        folder_files = scan_gdrive_folder(url)
                        for f in folder_files:
                            f_url = f.get("url")
                            if f_url and f_url not in seen:
                                seen.add(f_url)
                                found.append(f_url)
                        continue
                    except Exception as exc:
                        log.warning("Could not auto-expand Google Drive folder: %s", exc)

            seen.add(url)
            found.append(url)
    return found


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    cfg = load_config(force=True)
    worker = get_worker()
    if bool(cfg.get("server.autostart_worker", True)):
        worker.start()
    try:
        yield
    finally:
        worker.stop(timeout=20.0)


app = FastAPI(title="sang_det", version="1.0.0", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ----------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>sang_det</h1><p>static/index.html is missing.</p>", 500)
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- API


@app.post("/api/videos")
async def submit_videos(payload: SubmitPayload) -> JSONResponse:
    urls = parse_urls(payload.urls)
    if not urls:
        raise HTTPException(400, "No valid http(s) links found in the input.")

    added, skipped = db.add_videos(urls)
    db.log_event("info", f"Queued {added} link(s), skipped {skipped} duplicate(s)")

    worker = get_worker()
    if not worker.running:
        worker.start()

    return JSONResponse({"added": added, "skipped": skipped, "total_submitted": len(urls)})


@app.post("/api/gdrive/scan")
async def gdrive_scan(payload: GDriveScanPayload) -> dict:
    url = payload.url.strip()
    if not url:
        raise HTTPException(400, "No Google Drive link provided.")
    if not is_gdrive_url(url):
        raise HTTPException(400, "Not a recognized Google Drive URL.")

    kind, target_id = parse_gdrive_url(url)
    if kind == "file":
        return {
            "ok": True,
            "kind": "file",
            "files": [{
                "id": target_id,
                "name": f"gdrive_{target_id}",
                "url": f"https://drive.google.com/file/d/{target_id}/view",
                "type": "image",
                "extension": "",
            }],
            "images_count": 1,
            "videos_count": 0,
            "total": 1,
        }

    try:
        files = scan_gdrive_folder(url)
        images = [f for f in files if f.get("type") == "image"]
        videos = [f for f in files if f.get("type") == "video"]
        return {
            "ok": True,
            "kind": "folder",
            "folder_id": target_id,
            "files": files,
            "images_count": len(images),
            "videos_count": len(videos),
            "total": len(files),
        }
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/gdrive/queue")
async def gdrive_queue(payload: GDriveQueuePayload) -> dict:
    urls: list[str] = []
    if payload.file_ids:
        urls = [f"https://drive.google.com/file/d/{fid.strip()}/view" for fid in payload.file_ids if fid.strip()]
    elif payload.url:
        kind, target_id = parse_gdrive_url(payload.url)
        if kind == "file":
            urls = [f"https://drive.google.com/file/d/{target_id}/view"]
        else:
            try:
                files = scan_gdrive_folder(payload.url)
                if payload.only_images:
                    files = [f for f in files if f.get("type") == "image"]
                else:
                    files = [f for f in files if f.get("type") in ("image", "video")]
                urls = [f["url"] for f in files if f.get("url")]
            except Exception as exc:
                raise HTTPException(400, f"Failed scanning Google Drive folder: {exc}") from exc

    if not urls:
        raise HTTPException(400, "No files found to queue.")

    added, skipped = db.add_videos(urls)
    db.log_event("info", f"Queued {added} Google Drive file(s), skipped {skipped} duplicate(s)")

    worker = get_worker()
    if not worker.running:
        worker.start()

    return {"ok": True, "added": added, "skipped": skipped, "total_submitted": len(urls)}


@app.get("/api/status")
async def status() -> dict:
    cfg = load_config()
    worker = get_worker()
    videos = db.list_videos()
    totals = db.totals()

    now = time.time()
    for video in videos:
        started = video.get("started_at")
        finished = video.get("finished_at")
        video["elapsed_s"] = round((finished or now) - started, 1) if started else None
        duration = video.get("duration_s") or 0
        position = video.get("position_s") or 0
        video["progress"] = round(min(1.0, position / duration), 4) if duration > 0 else None

    return {
        "worker": worker.status(),
        "totals": totals,
        "videos": videos,
        "config": cfg.as_dict(),
        "editable_keys": sorted(config_module.EDITABLE_KEYS),
        "server_time": now,
    }


@app.get("/api/events")
async def events(limit: int = 60) -> dict:
    return {"events": db.recent_events(max(1, min(500, limit)))}


@app.get("/api/plates/recent")
async def plates_recent(limit: int = 24) -> dict:
    return {"plates": db.recent_plates(max(1, min(200, limit)))}


@app.get("/api/plates/file/{filename}")
async def plate_file(filename: str):
    """Serve one saved crop for the dashboard preview strip (from disk or Supabase)."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    cfg = load_config()
    output_dir = cfg.path("output.dir").resolve()
    path = (output_dir / filename).resolve()

    if path.exists() and str(path).startswith(str(output_dir)) and path.is_file():
        try:
            data = path.read_bytes()
            ext = path.suffix.lower()
            mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
            return Response(content=data, media_type=mime)
        except OSError:
            pass

    # Check database or Supabase Storage
    plate = db.get_plate_by_filename(filename)
    storage_url = plate.get("storage_url") if plate else None
    image_bytes = storage.fetch_image_bytes(filename, storage_url=storage_url, cfg=cfg)
    if image_bytes:
        ext = Path(filename).suffix.lower()
        mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
        return Response(content=image_bytes, media_type=mime)

    raise HTTPException(404, "Plate image not found")



@app.post("/api/control/{action}")
async def control(action: str) -> dict:
    worker = get_worker()
    action = action.lower()

    if action == "pause":
        worker.pause()
    elif action == "resume":
        if not worker.running:
            worker.start()
        worker.resume()
    elif action == "start":
        worker.start()
    elif action == "retry-errors":
        count = db.requeue_all_errored()
        if not worker.running:
            worker.start()
        return {"ok": True, "requeued": count}
    else:
        raise HTTPException(400, f"Unknown action: {action}")

    return {"ok": True, "state": worker.status()["state"]}


@app.post("/api/videos/{video_id}/{action}")
async def video_action(video_id: int, action: str) -> dict:
    video = db.get_video(video_id)
    if video is None:
        raise HTTPException(404, "Video not found")

    action = action.lower()
    if action == "retry":
        db.requeue_video(video_id, reset_progress=False)
    elif action == "restart":
        db.requeue_video(video_id, reset_progress=True)
    elif action == "cancel":
        db.cancel_video(video_id)
    elif action == "delete":
        db.delete_video(video_id)
    else:
        raise HTTPException(400, f"Unknown action: {action}")

    worker = get_worker()
    if action in ("retry", "restart") and not worker.running:
        worker.start()
    return {"ok": True}


@app.get("/api/config")
async def get_config() -> dict:
    return {
        "config": load_config().as_dict(),
        "editable_keys": sorted(config_module.EDITABLE_KEYS),
        "path": str(config_module.CONFIG_PATH),
    }


@app.put("/api/config")
async def put_config(payload: ConfigPayload) -> dict:
    if not payload.changes:
        raise HTTPException(400, "No changes supplied")
    updated = config_module.update(payload.changes)
    db.log_event("info", f"Config updated: {sorted(payload.changes)}")
    return {"ok": True, "config": updated.as_dict()}


@app.get("/api/health")
async def health() -> dict:
    worker = get_worker()
    return {
        "ok": True,
        "worker_running": worker.running,
        "version": app.version,
        "output_dir": str(load_config().path("output.dir")),
    }


# ----------------------------------------------------------- AI Labeler (NVIDIA NIM)


class LabelScanPayload(BaseModel):
    api_key: str | None = None
    model: str | None = None
    prompt: str | None = None


class BatchScanPayload(BaseModel):
    limit: int = Field(default=10000, description="Max plates to process in this run")
    force_all: bool = Field(default=False, description="Re-scan already labeled plates if True")
    retry_errors: bool = Field(default=True, description="Retry previously errored plates if True")
    api_key: str | None = None



class PlateUpdatePayload(BaseModel):
    plate_text: str = Field(default="", description="Cleaned license plate text")


class CSVImportPayload(BaseModel):
    csv_text: str = Field(default="", description="CSV string to import")


class FolderScanPayload(BaseModel):
    folder_path: str = Field(..., description="Local folder directory path on disk")
    recursive: bool = Field(default=True, description="Scan subfolders recursively if True")


class FolderImportPayload(BaseModel):
    folder_path: str = Field(..., description="Local folder directory path on disk")
    recursive: bool = Field(default=True, description="Scan subfolders recursively if True")
    auto_scan: bool = Field(default=False, description="Auto-scan with NVIDIA NIM on import if True")
    api_key: str | None = Field(default=None, description="Optional NVIDIA NIM API Key")


class GDriveLabelerImportPayload(BaseModel):
    url: str = Field(..., description="Google Drive folder URL or ID")
    auto_scan: bool = Field(default=False, description="Auto-scan with NVIDIA NIM on import if True")
    api_key: str | None = Field(default=None, description="Optional NVIDIA NIM API Key")


class ResetLabelerPayload(BaseModel):
    mode: str = Field(default="clear_all", description="'clear_all' or 'reset_labels'")
    delete_files: bool = Field(default=True, description="Delete crop image files from disk if True")


@app.get("/api/labeler/status")
async def labeler_status() -> dict:
    batch = labeler.get_batch_labeler()
    return {"ok": True, **batch.status()}


@app.get("/api/labeler/plates")
async def labeler_list_plates(
    limit: int = 50,
    offset: int = 0,
    status: str = "all",
    search: str = "",
) -> dict:
    items, total = db.list_plates(
        limit=limit,
        offset=offset,
        status=None if status == "all" else status,
        search=search if search else None,
    )
    return {
        "ok": True,
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "stats": db.get_label_stats(),
    }


@app.post("/api/labeler/scan-plate/{plate_id}")
async def labeler_scan_single_plate(
    plate_id: int,
    payload: LabelScanPayload | None = None,
) -> dict:
    cfg = load_config()
    client = labeler.NimLabeler(cfg)
    api_key = payload.api_key if payload else None
    res = client.scan_plate_record(plate_id, api_key=api_key)
    plate = db.get_plate(plate_id)
    return {
        "ok": res.ok,
        "plate_text": res.plate_text,
        "error": res.error,
        "latency_s": res.latency_s,
        "plate": plate,
    }


@app.post("/api/labeler/scan-all")
async def labeler_scan_all(payload: BatchScanPayload | None = None) -> dict:
    limit = payload.limit if payload else 10000
    force_all = payload.force_all if payload else False
    retry_errors = payload.retry_errors if payload else True
    api_key = payload.api_key if payload else None

    batch = labeler.get_batch_labeler()
    started = batch.start_batch(limit=limit, force_all=force_all, retry_errors=retry_errors, api_key=api_key)
    if not started:
        raise HTTPException(400, "A batch labeling job is already in progress.")
    return {"ok": True, "message": "Batch labeling started.", "status": batch.status()}



@app.post("/api/labeler/cancel")
async def labeler_cancel() -> dict:
    batch = labeler.get_batch_labeler()
    batch.cancel()
    return {"ok": True, "message": "Cancellation requested."}


@app.put("/api/labeler/plates/{plate_id}")
async def labeler_update_plate(plate_id: int, payload: PlateUpdatePayload) -> dict:
    plate = db.get_plate(plate_id)
    if plate is None:
        raise HTTPException(404, "Plate not found")

    text = labeler.clean_plate_text(payload.plate_text)
    db.update_plate_label(plate_id, text, ocr_status="labeled")
    labeler.sync_labels_csv()
    return {"ok": True, "plate": db.get_plate(plate_id)}


@app.delete("/api/labeler/plates/{plate_id}")
async def labeler_delete_plate(plate_id: int) -> dict:
    plate = db.get_plate(plate_id)
    if plate is None:
        raise HTTPException(404, "Plate not found")

    # Delete db record
    db.delete_plate(plate_id)

    # Optionally unlink file if desired
    output_dir = load_config().path("output.dir")
    file_path = output_dir / plate["filename"]
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass

    labeler.sync_labels_csv()
    return {"ok": True}


@app.post("/api/labeler/upload")
async def labeler_upload_image(
    file: UploadFile = File(...),
    auto_scan: bool = Form(True),
    api_key: str = Form(""),
) -> dict:
    """Directly upload an image, save it to output folder, create a record, and auto-scan."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    output_dir = load_config().path("output.dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    if len(contents) < 128:
        raise HTTPException(400, "Uploaded file is too small or invalid")

    # Generate safe unique filename
    slug = labeler.clean_plate_text(Path(file.filename).stem)[:24].replace(" ", "_") or "upload"
    safe_name = f"upload_{slug}_{int(time.time()*1000)}_{file.filename}"
    safe_path = output_dir / safe_name
    safe_path.write_bytes(contents)

    # Image dimensions
    width, height = 0, 0
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            height, width = img.shape[:2]
    except Exception:
        pass

    cfg = load_config()
    storage_url = None
    if storage.is_supabase_storage_enabled(cfg):
        storage_url = storage.upload_to_supabase_storage(safe_name, contents, "image/jpeg", cfg)

    plate_id = db.record_custom_plate(
        filename=safe_name,
        width=width,
        height=height,
        nbytes=len(contents),
        ocr_status="unlabeled",
        storage_url=storage_url,
    )


    scan_result: dict[str, Any] = {"ok": False, "plate_text": ""}
    if auto_scan:
        cfg = load_config()
        client = labeler.NimLabeler(cfg)
        res = client.scan_plate_record(plate_id, api_key=api_key or None)
        scan_result = {
            "ok": res.ok,
            "plate_text": res.plate_text,
            "error": res.error,
            "latency_s": res.latency_s,
        }

    labeler.sync_labels_csv()

    return {
        "ok": True,
        "plate_id": plate_id,
        "filename": safe_name,
        "scan": scan_result,
        "plate": db.get_plate(plate_id),
    }


@app.get("/api/labeler/export-csv")
async def labeler_export_csv() -> Response:
    cfg = load_config()
    csv_text, _ = labeler.export_labels_to_csv(cfg=cfg)
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sang_det_labels.csv"'},
    )


@app.post("/api/labeler/import-csv")
async def labeler_import_csv(
    file: UploadFile | None = File(None),
    payload: CSVImportPayload | None = None,
) -> dict:
    if file is not None:
        contents = await file.read()
        text = contents.decode("utf-8", errors="replace")
    elif payload and payload.csv_text:
        text = payload.csv_text
    else:
        raise HTTPException(400, "No CSV file or text provided")

    result = labeler.import_labels_from_csv(text)
    return result


@app.post("/api/labeler/scan-folder")
async def labeler_scan_folder(payload: FolderScanPayload) -> dict:
    folder_path = payload.folder_path.strip().strip('"\'')
    if not folder_path:
        raise HTTPException(400, "No folder path provided.")

    try:
        files = labeler.scan_local_folder(folder_path, recursive=payload.recursive)
        preview = [f.name for f in files[:25]]
        return {
            "ok": True,
            "path": str(Path(folder_path).expanduser().resolve()),
            "total": len(files),
            "files_preview": preview,
        }
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Failed scanning folder: {exc}") from exc


@app.post("/api/labeler/import-folder")
async def labeler_import_folder(payload: FolderImportPayload) -> dict:
    folder_path = payload.folder_path.strip().strip('"\'')
    if not folder_path:
        raise HTTPException(400, "No folder path provided.")

    try:
        res = labeler.import_local_image_folder(
            folder_path=folder_path,
            recursive=payload.recursive,
            auto_scan=payload.auto_scan,
            api_key=payload.api_key,
        )
        return res
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Failed importing folder: {exc}") from exc


@app.post("/api/labeler/import-gdrive")
async def labeler_import_gdrive(payload: GDriveLabelerImportPayload) -> dict:
    url = payload.url.strip()
    if not url:
        raise HTTPException(400, "No Google Drive folder URL provided.")

    try:
        res = labeler.import_gdrive_folder_images(
            url_or_id=url,
            auto_scan=payload.auto_scan,
            api_key=payload.api_key,
        )
        return res
    except Exception as exc:
        raise HTTPException(400, f"Failed importing Google Drive folder: {exc}") from exc


@app.post("/api/labeler/reset")
async def labeler_reset(payload: ResetLabelerPayload | None = None) -> dict:
    mode = payload.mode if payload else "clear_all"
    delete_files = payload.delete_files if payload else True
    return labeler.reset_labeler_data(mode=mode, delete_files=delete_files)


def run() -> None:
    """Entry point used by run.py serve."""
    import uvicorn
    import os

    cfg = load_config(force=True)
    host = os.environ.get("HOST") or str(cfg.get("server.host", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    port = int(os.environ.get("PORT") or cfg.get("server.port", 8000))
    log.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)



__all__ = ["app", "run", "parse_urls", "Path"]
