"""Output writing, disk guard, and the provenance manifest.

Writes are atomic: encode in memory, write to a temp file on the same volume,
flush + fsync, then os.replace() into place. A crash or power cut can leave a
stray .tmp file but never a truncated or zero-byte JPEG in the output folder.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import httpx
from .config import Config, load as load_config
from .logging_setup import get

log = get("storage")

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SaveResult:
    ok: bool
    path: Path | None = None
    nbytes: int = 0
    reason: str = ""
    storage_url: str | None = None


# ------------------------------------------------------------- Supabase Storage


def is_supabase_storage_enabled(cfg: Config | None = None) -> bool:
    c = cfg or load_config()
    url = str(c.get("supabase.url") or "").strip()
    key = str(c.get("supabase.key") or "").strip()
    return bool(url and key)


def get_supabase_public_url(filename: str, cfg: Config | None = None) -> str:
    c = cfg or load_config()
    sb_url = str(c.get("supabase.url") or "").rstrip("/")
    bucket = str(c.get("supabase.bucket") or "plates").strip("/")
    return f"{sb_url}/storage/v1/object/public/{bucket}/{filename}"


def upload_to_supabase_storage(
    filename: str,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
    cfg: Config | None = None,
) -> str | None:
    """Upload image bytes to Supabase Storage bucket and return the public URL."""
    c = cfg or load_config()
    sb_url = str(c.get("supabase.url") or "").rstrip("/")
    sb_key = str(c.get("supabase.key") or "").strip()
    bucket = str(c.get("supabase.bucket") or "plates").strip("/")

    if not sb_url or not sb_key:
        return None

    upload_endpoint = f"{sb_url}/storage/v1/object/{bucket}/{filename}"
    headers = {
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(upload_endpoint, content=image_bytes, headers=headers)
            if resp.status_code in (200, 201):
                public_url = get_supabase_public_url(filename, c)
                return public_url
            else:
                log.warning(
                    "Supabase storage upload failed for %s (status %d): %s",
                    filename, resp.status_code, resp.text[:200],
                )
                return None
    except Exception as exc:
        log.warning("Supabase storage upload error for %s: %s", filename, exc)
        return None


def fetch_image_bytes(
    filename: str,
    storage_url: str | None = None,
    cfg: Config | None = None,
) -> bytes | None:
    """Fetch image bytes from local disk or Supabase Storage."""
    c = cfg or load_config()
    output_dir = c.path("output.dir")
    local_path = output_dir / filename
    if local_path.exists():
        try:
            return local_path.read_bytes()
        except OSError:
            pass

    # Try Supabase Storage URL or construct public URL
    url = storage_url or (get_supabase_public_url(filename, c) if is_supabase_storage_enabled(c) else None)
    if url:
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    return resp.content
        except Exception as exc:
            log.warning("Failed fetching image from Supabase Storage (%s): %s", url, exc)

    return None



def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe token used in output filenames."""
    cleaned = _SLUG_RE.sub("-", (text or "video").strip()).strip("-._")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return (cleaned[:max_len] or "video").lower()


class DiskGuard:
    """Pauses saving - never crashes - when the output volume fills up."""

    def __init__(self, cfg: Config, output_dir: Path):
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._saves_since_check = 0
        self._last_check = 0.0
        self._free_gb = float("inf")
        self._paused = False
        self.refresh(cfg, force=True)

    def _free_bytes(self) -> int:
        probe = self.output_dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free
        except OSError as exc:
            log.warning("Could not read free space (%s); assuming OK", exc)
            return 1 << 60

    def refresh(self, cfg: Config, force: bool = False) -> bool:
        """Re-check free space when due. Returns True if saving is allowed."""
        every_n = int(cfg.get("storage.check_every_n_saves", 50))
        every_s = float(cfg.get("storage.check_every_n_seconds", 60))
        now = time.time()

        with self._lock:
            due = (
                force
                or self._saves_since_check >= every_n
                or (now - self._last_check) >= every_s
            )
            if not due:
                return not self._paused

            self._saves_since_check = 0
            self._last_check = now
            self._free_gb = self._free_bytes() / (1024 ** 3)
            threshold = float(cfg.get("storage.min_free_gb", 5.0))
            was_paused = self._paused
            self._paused = self._free_gb < threshold

            if self._paused and not was_paused:
                log.error(
                    "DISK GUARD: %.2f GB free < %.2f GB threshold - saving paused",
                    self._free_gb, threshold,
                )
            elif was_paused and not self._paused:
                log.info("DISK GUARD: %.2f GB free - saving resumed", self._free_gb)
            return not self._paused

    def note_save(self) -> None:
        with self._lock:
            self._saves_since_check += 1

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def free_gb(self) -> float:
        return self._free_gb


class PlateWriter:
    """Atomic JPEG writer plus the JSONL provenance manifest."""

    def __init__(self, cfg: Config):
        self.output_dir = cfg.path("output.dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = cfg.path("output.manifest")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.guard = DiskGuard(cfg, self.output_dir)
        self._manifest_lock = threading.Lock()
        self._name_lock = threading.Lock()
        self._used_names: set[str] = set()

    # ------------------------------------------------------------- naming

    def build_name(
        self, video_id: int, title: str, frame_index: int, timestamp_s: float, seq: int
    ) -> str:
        """`v{id}_{slug}_t{ms}_f{frame}_{n}.jpg` - traceable, collision-free.

        Provenance lives in the filename and the manifest, never in decoded
        plate text; no OCR runs anywhere in this tool.
        """
        base = (
            f"v{video_id:04d}_{slugify(title)}"
            f"_t{int(round(timestamp_s * 1000)):09d}"
            f"_f{frame_index:07d}_{seq:02d}"
        )
        with self._name_lock:
            name = f"{base}.jpg"
            suffix = 1
            while name in self._used_names or (self.output_dir / name).exists():
                name = f"{base}-{suffix}.jpg"
                suffix += 1
            self._used_names.add(name)
        return name

    # -------------------------------------------------------------- write

    def save(self, crop_bgr: np.ndarray, filename: str, cfg: Config) -> SaveResult:
        """Encode and atomically write one crop."""
        if self.guard.paused:
            return SaveResult(False, reason="disk_paused")

        quality = int(cfg.get("output.jpeg_quality", 92))
        min_width = int(cfg.get("output.min_save_width", 0) or 0)

        try:
            import cv2

            image = crop_bgr
            if min_width and image.shape[1] < min_width:
                scale = min_width / float(image.shape[1])
                image = cv2.resize(
                    image,
                    (min_width, max(1, int(round(image.shape[0] * scale)))),
                    interpolation=cv2.INTER_CUBIC,
                )

            ok, buffer = cv2.imencode(
                ".jpg", image,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, quality)),
                 int(cv2.IMWRITE_JPEG_OPTIMIZE), 1],
            )
            if not ok or buffer is None or buffer.size == 0:
                return SaveResult(False, reason="encode_failed")
            payload = buffer.tobytes()
        except Exception as exc:
            return SaveResult(False, reason=f"encode_error:{exc}")

        if len(payload) < 128:  # a valid JPEG is never this small
            return SaveResult(False, reason="encode_too_small")

        final_path = self.output_dir / filename
        tmp_path = final_path.with_name(final_path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
        except OSError as exc:
            self._cleanup(tmp_path)
            # Out of space mid-write: flip the guard rather than fail the run.
            if getattr(exc, "errno", None) in (28, 122):
                self.guard.refresh(cfg, force=True)
            return SaveResult(False, reason=f"write_error:{exc}")

        try:
            size = final_path.stat().st_size
        except OSError:
            return SaveResult(False, reason="stat_failed")
        if size <= 0:
            self._cleanup(final_path)
            return SaveResult(False, reason="zero_byte")

        self.guard.note_save()

        storage_url = None
        if is_supabase_storage_enabled(cfg):
            storage_url = upload_to_supabase_storage(filename, payload, "image/jpeg", cfg)

        return SaveResult(True, path=final_path, nbytes=size, storage_url=storage_url)


    @staticmethod
    def _cleanup(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass

    # ----------------------------------------------------------- manifest

    def append_manifest(self, record: dict) -> None:
        """One JSON object per saved crop: file -> source video + timestamp."""
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return
        with self._manifest_lock:
            try:
                with open(self.manifest_path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                log.warning("Manifest append failed: %s", exc)

    def sweep_temp_files(self) -> int:
        """Remove .tmp leftovers from a previous hard kill."""
        removed = 0
        try:
            for stale in self.output_dir.glob("*.jpg.tmp"):
                try:
                    stale.unlink()
                    removed += 1
                except OSError:
                    pass
        except OSError:
            pass
        if removed:
            log.info("Cleared %d stale temp file(s)", removed)
        return removed

    def count_files(self) -> int:
        try:
            return sum(1 for _ in self.output_dir.glob("*.jpg"))
        except OSError:
            return 0
