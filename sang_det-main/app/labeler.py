"""NVIDIA NIM API integration for AI license plate OCR and labeling.

Uses multimodal models (such as nvidia/nemotron-3-nano-omni-30b-a3b-reasoning)
via the OpenAI-compatible NVIDIA NIM API to scan plate crop images into text,
store the results in SQLite, and synchronize with CSV manifests.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db
from . import storage
from .config import Config, load as load_config
from .logging_setup import get


log = get("labeler")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"```(?:[a-zA-Z]*\n)?(.*?)```", re.DOTALL)


def clean_plate_text(raw_text: str) -> str:
    """Clean model response, stripping thinking tokens, markdown, and noise."""
    if not raw_text:
        return ""

    text = raw_text.strip()
    # Remove <think>...</think> reasoning blocks
    text = _THINK_RE.sub("", text).strip()

    # Extract code blocks if model wrapped output in markdown backticks
    code_match = _MARKDOWN_RE.search(text)
    if code_match:
        text = code_match.group(1).strip()

    # Take the first non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    # Clean leading/trailing quotes, colons, brackets
    text = text.strip("\"'`()[]{}")

    # Common prefixes model might generate
    for prefix in ("plate:", "license plate:", "plate number:", "text:", "number:", "extracted plate:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # Normalize whitespace and convert to uppercase
    text = re.sub(r"\s+", " ", text).strip().upper()

    if text in ("UNREADABLE", "NOT_FOUND", "NONE", "UNKNOWN", "N/A", "NO PLATE"):
        return "UNREADABLE"

    return text


@dataclass
class ScanResult:
    ok: bool
    plate_text: str = ""
    raw_response: str = ""
    error: str = ""
    latency_s: float = 0.0


def _optimize_image_for_nim(image_bytes: bytes) -> tuple[bytes | None, str]:
    """Ensure plate crop is compact and optimized (< 50KB) to minimize NIM inference time and token limits."""
    if not image_bytes or len(image_bytes) < 64:
        return None, ""

    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            # Try PIL fallback
            from PIL import Image
            import io

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB to BGR

        if img is not None and img.size > 0:
            h, w = img.shape[:2]
            max_dim = 480
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                new_w = max(32, int(round(w * scale)))
                new_h = max(32, int(round(h * scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ok, buffer = cv2.imencode(
                ".jpg", img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
            )
            if ok and buffer is not None and len(buffer) > 64:
                return buffer.tobytes(), "image/jpeg"
    except Exception as exc:
        log.debug("Image optimization failed: %s", exc)

    return None, ""


class NimLabeler:
    """Client for NVIDIA NIM Vision/OCR API."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()

    def get_api_key(self) -> str:
        key = str(self.cfg.get("labeler.api_key", "")).strip().strip('"\'')
        if not key:
            key = os.environ.get("NVIDIA_API_KEY", "").strip() or os.environ.get("NVIDIA_NIM_API_KEY", "").strip()

        # Auto-sanitize if multiple nvapi keys were accidentally pasted together
        if "nvapi-" in key:
            parts = [("nvapi-" + p.strip()) for p in key.split("nvapi-") if p.strip()]
            if parts:
                key = parts[-1].strip()  # use the latest valid key token
        return key

    def get_base_url(self) -> str:
        url = str(self.cfg.get("labeler.base_url", "https://integrate.api.nvidia.com/v1")).strip()
        return url.rstrip("/")

    def get_model(self) -> str:
        model = str(self.cfg.get("labeler.model", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")).strip()
        return model or "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

    def get_fallback_model(self) -> str:
        fallback = str(self.cfg.get("labeler.fallback_model", "meta/llama-3.2-11b-vision-instruct")).strip()
        return fallback

    def get_prompt(self) -> str:
        prompt = str(self.cfg.get("labeler.prompt", "")).strip()
        if not prompt:
            prompt = (
                "Extract and output only the license plate alphanumeric characters from this image. "
                "Output uppercase text with standard spacing. If unreadable, output UNREADABLE. "
                "Do not include any explanations, reasoning, or extra text."
            )
        return prompt

    def scan_image_bytes(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        api_key: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        max_retries: int = 3,
        allow_fallback: bool = True,
    ) -> ScanResult:
        """Send image bytes to NVIDIA NIM API for OCR plate text extraction with auto-retry and fallback."""
        key = (api_key or self.get_api_key()).strip()
        if not key:
            return ScanResult(
                ok=False,
                error="NVIDIA NIM API key is missing. Configure it in Settings or set NVIDIA_API_KEY.",
            )

        # Optimize image size before base64 encoding and validate readability
        opt_bytes, opt_mime = _optimize_image_for_nim(image_bytes)
        if opt_bytes is None:
            return ScanResult(
                ok=False,
                error="Image file is corrupt, empty, or cannot be decoded as an image.",
            )

        target_model = (model or self.get_model()).strip()
        target_prompt = (prompt or self.get_prompt()).strip()
        base_url = self.get_base_url()
        endpoint = f"{base_url}/chat/completions"

        b64_img = base64.b64encode(opt_bytes).decode("utf-8")
        data_url = f"data:{opt_mime};base64,{b64_img}"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": target_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            "temperature": float(self.cfg.get("labeler.temperature", 0.1)),
            "max_tokens": int(self.cfg.get("labeler.max_tokens", 128)),
        }

        # Add reasoning_budget for Nemotron-3 reasoning models to avoid runaway thinking loops
        if "reasoning" in target_model.lower() or "nemotron" in target_model.lower():
            budget = int(self.cfg.get("labeler.reasoning_budget", 128))
            payload["reasoning_budget"] = budget

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sang_det-nim-labeler/1.0",
        }

        req_data = json.dumps(payload).encode("utf-8")
        timeout_s = float(self.cfg.get("labeler.timeout", 60.0))

        last_error = ""
        total_latency = 0.0

        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(endpoint, data=req_data, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    status_code = resp.status
                    body = resp.read().decode("utf-8", errors="replace")
                    latency = round(time.time() - t0, 3)
                    total_latency += latency

                    if status_code != 200:
                        return ScanResult(
                            ok=False,
                            error=f"NVIDIA NIM API error (HTTP {status_code}): {body[:300]}",
                            latency_s=total_latency,
                        )

                    data = json.loads(body)
                    choices = data.get("choices", [])
                    if not choices:
                        return ScanResult(
                            ok=False,
                            error="No completions returned from NVIDIA NIM API.",
                            raw_response=body,
                            latency_s=total_latency,
                        )

                    raw_text = choices[0].get("message", {}).get("content", "")
                    clean_text = clean_plate_text(raw_text)
                    return ScanResult(
                        ok=True,
                        plate_text=clean_text,
                        raw_response=raw_text,
                        latency_s=total_latency,
                    )

            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                latency = round(time.time() - t0, 3)
                total_latency += latency
                log.warning("NVIDIA NIM HTTP %d (attempt %d/%d): %s", exc.code, attempt, max_retries, err_body[:180])

                if exc.code in (401, 403):
                    return ScanResult(
                        ok=False,
                        error="Authentication/Authorization failed (HTTP 403/401): invalid or unauthorized NVIDIA NIM API key.",
                        latency_s=total_latency,
                    )
                elif exc.code == 503 and allow_fallback and self.get_fallback_model() and self.get_fallback_model() != target_model:
                    log.info("Primary model '%s' queue is exhausted (503); immediately switching to fallback model '%s'...", target_model, self.get_fallback_model())
                    last_error = f"Primary model busy ({exc.code}); switched to fallback."
                    break
                elif exc.code in (503, 429, 502, 504) and attempt < max_retries:
                    # ResourceExhausted or RateLimit -> exponential backoff with jitter
                    backoff = min(8.0, (1.5 ** attempt) + (attempt * 0.5))
                    log.info("NIM busy (%d); backing off for %.1fs before retry...", exc.code, backoff)
                    time.sleep(backoff)
                    last_error = f"NVIDIA NIM busy ({exc.code} ResourceExhausted). Server queue reached limit."
                    continue
                else:
                    msg = f"NVIDIA NIM API error ({exc.code}): {err_body[:200]}"
                    last_error = msg
                    break

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                latency = round(time.time() - t0, 3)
                total_latency += latency
                log.warning("NVIDIA NIM connection error (attempt %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    backoff = min(15.0, (2.0 ** attempt) + 1.0)
                    log.info("NIM connection timeout/error; retrying in %.1fs...", backoff)
                    time.sleep(backoff)
                    last_error = f"NVIDIA NIM connection timed out after {timeout_s}s."
                    continue
                else:
                    last_error = f"NVIDIA NIM request timed out or network error ({exc})"
                    break

            except Exception as exc:
                latency = round(time.time() - t0, 3)
                total_latency += latency
                log.exception("NVIDIA NIM unexpected exception: %s", exc)
                return ScanResult(ok=False, error=f"Scan error: {exc}", latency_s=total_latency)

        # If primary model is exhausted (503) or timed out, try fallback model automatically
        fallback = self.get_fallback_model()
        if allow_fallback and fallback and fallback != target_model:
            log.info("Primary model '%s' queue is exhausted; automatically trying fallback '%s'...", target_model, fallback)
            fallback_res = self.scan_image_bytes(
                image_bytes=image_bytes,
                mime_type=mime_type,
                api_key=api_key,
                model=fallback,
                prompt=prompt,
                max_retries=2,
                allow_fallback=False,
            )
            if fallback_res.ok:
                return fallback_res

        return ScanResult(ok=False, error=last_error or "NVIDIA NIM request failed after retries.", latency_s=total_latency)

    def scan_image_file(
        self,
        image_path: Path,
        api_key: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
    ) -> ScanResult:
        if not image_path.exists() or not image_path.is_file():
            return ScanResult(ok=False, error=f"Image file not found: {image_path}")

        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            return ScanResult(ok=False, error=f"Could not read image file: {exc}")

        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        return self.scan_image_bytes(
            image_bytes,
            mime_type=mime,
            api_key=api_key,
            model=model,
            prompt=prompt,
        )

    def scan_plate_record(self, plate_id: int, api_key: str | None = None) -> ScanResult:
        """Scan a recorded plate crop from the database and output directory."""
        plate = db.get_plate(plate_id)
        if not plate:
            return ScanResult(ok=False, error=f"Plate record #{plate_id} not found in database.")

        filename = plate["filename"]
        storage_url = plate.get("storage_url")
        output_dir = self.cfg.path("output.dir")
        image_path = output_dir / filename

        if image_path.exists():
            res = self.scan_image_file(image_path, api_key=api_key)
        else:
            image_bytes = storage.fetch_image_bytes(filename, storage_url=storage_url, cfg=self.cfg)
            if image_bytes:
                mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                res = self.scan_image_bytes(image_bytes, mime_type=mime, api_key=api_key)
            else:
                res = ScanResult(ok=False, error=f"Plate image '{filename}' not found on disk or Supabase Storage.")

        if res.ok:
            db.update_plate_label(plate_id, res.plate_text, ocr_status="labeled")
            db.log_event("info", f"NIM OCR #{plate_id} ({filename}) -> '{res.plate_text}'", video_id=plate.get("video_id"))
            # Keep CSV file in sync
            sync_labels_csv(self.cfg)
        else:
            db.update_plate_label(plate_id, None, ocr_status="error")
            db.log_event("warning", f"NIM OCR #{plate_id} ({filename}) failed: {res.error}", video_id=plate.get("video_id"))

        return res



class BatchLabeler:
    """Background worker for batch scanning plate images."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "running": False,
            "total": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "current_filename": "",
            "last_error": "",
            "started_at": 0.0,
            "elapsed_s": 0.0,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            st = dict(self._status)
            if st["running"] and st["started_at"] > 0:
                st["elapsed_s"] = round(time.time() - st["started_at"], 1)
            # Add overall DB labeling stats
            st["stats"] = db.get_label_stats()
            return st

    def start_batch(self, limit: int = 500, force_all: bool = False, api_key: str | None = None) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False  # already running

            self._cancel_flag.clear()
            self._status = {
                "running": True,
                "total": 0,
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "current_filename": "",
                "last_error": "",
                "started_at": time.time(),
                "elapsed_s": 0.0,
            }

            self._thread = threading.Thread(
                target=self._run_batch,
                args=(limit, force_all, api_key),
                name="nim-batch-labeler",
                daemon=True,
            )
            self._thread.start()
            return True

    def cancel(self) -> None:
        self._cancel_flag.set()

    def _run_batch(self, limit: int, force_all: bool, api_key: str | None) -> None:
        cfg = load_config()
        client = NimLabeler(cfg)
        log.info("Starting NVIDIA NIM batch OCR labeling run...")

        if force_all:
            plates, _ = db.list_plates(limit=limit, offset=0)
        else:
            plates = db.get_unlabeled_plates(limit=limit)

        with self._lock:
            self._status["total"] = len(plates)

        if not plates:
            with self._lock:
                self._status["running"] = False
            log.info("No plates to label.")
            return

        for p in plates:
            if self._cancel_flag.is_set():
                log.info("NIM batch labeling cancelled by user.")
                break

            plate_id = p["id"]
            filename = p["filename"]
            with self._lock:
                self._status["current_filename"] = filename

            res = client.scan_plate_record(plate_id, api_key=api_key)

            with self._lock:
                self._status["processed"] += 1
                if res.ok:
                    self._status["succeeded"] += 1
                else:
                    self._status["failed"] += 1
                    self._status["last_error"] = res.error

            # Dynamic pacing: fast if OK, pause if server under pressure
            if res.ok:
                time.sleep(0.4)
            else:
                time.sleep(2.5)

        # Final CSV sync
        sync_labels_csv(cfg)

        with self._lock:
            self._status["running"] = False
            self._status["current_filename"] = ""
            log.info(
                "NIM batch OCR completed: %d/%d processed (%d succeeded, %d failed)",
                self._status["processed"],
                self._status["total"],
                self._status["succeeded"],
                self._status["failed"],
            )


# Global singleton
_batch_labeler: BatchLabeler | None = None


def get_batch_labeler() -> BatchLabeler:
    global _batch_labeler
    if _batch_labeler is None:
        _batch_labeler = BatchLabeler()
    return _batch_labeler


# ---------------------------------------------------------------- CSV handling


CSV_HEADER = [
    "id",
    "plate_text",
]


def export_labels_to_csv(output_path: Path | None = None, cfg: Config | None = None) -> tuple[str, Path]:
    """Export all plate records and labels to a clean CSV file (id, plate_text)."""
    cfg = cfg or load_config()
    dest = output_path or cfg.path("labeler.csv_path")
    dest.parent.mkdir(parents=True, exist_ok=True)

    plates = db.get_all_plates_for_export()
    output_io = io.StringIO()
    writer = csv.DictWriter(output_io, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()

    for p in plates:
        writer.writerow({
            "id": p.get("id"),
            "plate_text": p.get("plate_text") or "",
        })

    csv_text = output_io.getvalue()
    try:
        dest.write_text(csv_text, encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write CSV to %s: %s", dest, exc)

    return csv_text, dest


def sync_labels_csv(cfg: Config | None = None) -> None:
    """Synchronize the current DB state to the configured CSV path."""
    try:
        export_labels_to_csv(cfg=cfg)
    except Exception as exc:
        log.warning("Failed syncing labels to CSV: %s", exc)


def import_labels_from_csv(csv_content_or_path: str | Path) -> dict[str, Any]:
    """Parse CSV text or file and update matching plates in the database."""
    if isinstance(csv_content_or_path, Path):
        text = csv_content_or_path.read_text(encoding="utf-8")
    else:
        text = csv_content_or_path

    reader = csv.DictReader(io.StringIO(text))
    records: list[dict[str, Any]] = []

    for row in reader:
        # Standardize keys (strip whitespace, lowercase)
        norm_row = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}
        plate_text = (
            norm_row.get("plate_text")
            or norm_row.get("text")
            or norm_row.get("plate")
            or norm_row.get("label")
            or norm_row.get("number")
        )
        if not plate_text:
            continue

        item: dict[str, Any] = {"plate_text": clean_plate_text(plate_text)}
        if "id" in norm_row and norm_row["id"].isdigit():
            item["id"] = int(norm_row["id"])
        if "filename" in norm_row and norm_row["filename"]:
            item["filename"] = norm_row["filename"]

        if "id" in item or "filename" in item:
            records.append(item)

    if not records:
        return {"ok": True, "updated": 0, "total_rows": 0, "message": "No valid plate records found in CSV."}

    updated = db.batch_update_labels(records)
    sync_labels_csv()
    db.log_event("info", f"Imported labels CSV: updated {updated} plate record(s)")

    return {
        "ok": True,
        "updated": updated,
        "total_parsed": len(records),
    }


# ----------------------------------------------------------- Folder Operations


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".jfif", ".heic"
}


def scan_local_folder(folder_path: str | Path, recursive: bool = True) -> list[Path]:
    """Scan a local directory for image files."""
    p = Path(folder_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    files: list[Path] = []
    if recursive:
        for root, _, filenames in os.walk(p):
            for fname in sorted(filenames):
                if Path(fname).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    files.append(Path(root) / fname)
    else:
        for item in sorted(p.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                files.append(item)
    return files


def import_local_image_folder(
    folder_path: str | Path,
    recursive: bool = True,
    auto_scan: bool = False,
    api_key: str | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Import all images from a local filesystem folder into the labeler."""
    cfg = cfg or load_config()
    files = scan_local_folder(folder_path, recursive=recursive)
    output_dir = cfg.path("output.dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    scanned = 0
    errors: list[str] = []
    client = NimLabeler(cfg) if auto_scan else None

    for img_path in files:
        try:
            contents = img_path.read_bytes()
            if len(contents) < 128:
                continue

            slug = clean_plate_text(img_path.stem)[:24].replace(" ", "_") or "crop"
            safe_name = f"folder_{slug}_{int(time.time()*1000)}_{img_path.name}"
            dest_path = output_dir / safe_name
            dest_path.write_bytes(contents)

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
            imported += 1

            if auto_scan and client:
                res = client.scan_plate_record(plate_id, api_key=api_key)
                if res.ok:
                    scanned += 1
                else:
                    errors.append(f"{img_path.name}: {res.error}")

        except Exception as exc:
            log.warning("Could not import %s: %s", img_path, exc)
            errors.append(f"{img_path.name}: {exc}")

    sync_labels_csv(cfg)
    return {
        "ok": True,
        "total_found": len(files),
        "imported": imported,
        "scanned": scanned,
        "errors": errors[:50],
    }


def import_gdrive_folder_images(
    url_or_id: str,
    auto_scan: bool = False,
    api_key: str | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Import images from a Google Drive folder directly into the labeler."""
    from . import gdrive

    cfg = cfg or load_config()
    output_dir = cfg.path("output.dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = gdrive.scan_gdrive_folder(url_or_id)
    image_items = [item for item in items if item.get("type") == "image"]

    imported = 0
    scanned = 0
    errors: list[str] = []
    client = NimLabeler(cfg) if auto_scan else None

    for item in image_items:
        try:
            import cv2

            img_arr, title = gdrive.fetch_gdrive_image(item["id"])
            if img_arr is None or img_arr.size == 0:
                continue

            h, w = img_arr.shape[:2]
            slug = clean_plate_text(Path(title).stem)[:24].replace(" ", "_") or "gdrive"
            safe_name = f"gdrive_{slug}_{int(time.time()*1000)}_{title}"
            if not any(safe_name.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
                safe_name += ".jpg"

            dest_path = output_dir / safe_name
            cv2.imwrite(str(dest_path), img_arr)
            nbytes = dest_path.stat().st_size if dest_path.exists() else 0

            storage_url = None
            if storage.is_supabase_storage_enabled(cfg) and dest_path.exists():
                try:
                    storage_url = storage.upload_to_supabase_storage(safe_name, dest_path.read_bytes(), "image/jpeg", cfg)
                except Exception:
                    pass

            plate_id = db.record_custom_plate(
                filename=safe_name,
                width=w,
                height=h,
                nbytes=nbytes,
                ocr_status="unlabeled",
                storage_url=storage_url,
            )
            imported += 1


            if auto_scan and client:
                res = client.scan_plate_record(plate_id, api_key=api_key)
                if res.ok:
                    scanned += 1
                else:
                    errors.append(f"{title}: {res.error}")

        except Exception as exc:
            log.warning("Could not import gdrive image %s: %s", item.get("name"), exc)
            errors.append(f"{item.get('name', 'image')}: {exc}")

    sync_labels_csv(cfg)
    return {
        "ok": True,
        "total_found": len(image_items),
        "imported": imported,
        "scanned": scanned,
        "errors": errors[:50],
    }


def reset_labeler_data(
    mode: str = "clear_all",
    delete_files: bool = True,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Reset labeling session data.

    mode:
      - 'clear_all': Delete all plate records, remove crop files, and clear CSV.
      - 'reset_labels': Keep crop records/images, reset all labels to 'unlabeled'.
    """
    cfg = cfg or load_config()

    if mode == "reset_labels":
        updated = db.reset_all_plate_labels()
        sync_labels_csv(cfg)
        return {
            "ok": True,
            "mode": "reset_labels",
            "message": f"Reset labels for {updated} plate(s) to unlabeled. Ready to re-label.",
            "updated": updated,
            "stats": db.get_label_stats(),
        }

    # mode == "clear_all"
    plates = db.get_all_plates_for_export()
    output_dir = cfg.path("output.dir")
    deleted_files = 0

    if delete_files and output_dir.exists():
        for p in plates:
            fname = p.get("filename")
            if fname:
                fpath = output_dir / fname
                if fpath.exists():
                    try:
                        fpath.unlink()
                        deleted_files += 1
                    except OSError:
                        pass

    count = db.clear_all_plates()
    sync_labels_csv(cfg)

    return {
        "ok": True,
        "mode": "clear_all",
        "message": f"Cleared {count} plate record(s) and removed {deleted_files} crop image(s). Ready for new labeling.",
        "cleared_records": count,
        "deleted_files": deleted_files,
        "stats": db.get_label_stats(),
    }
