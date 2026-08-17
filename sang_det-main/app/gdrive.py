"""Google Drive integration for sang_det.

Provides:
  * URL parsing (detecting file vs folder links)
  * Public/shared folder scanning to list image and video files
  * In-memory image fetching (zero disk footprint)
  * Direct stream resolution for Google Drive videos
"""

from __future__ import annotations

import io
import os
import re
import urllib.parse
from typing import Any

import numpy as np
import requests

from .config import Config
from .logging_setup import get

log = get("gdrive")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".jfif", ".heic", ".gif"
}
VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".mpg", ".mpeg", ".ts"
}

GDRIVE_HOSTS = (
    "drive.google.com",
    "docs.google.com",
    "drive.usercontent.google.com",
    "googleusercontent.com",
)


def is_gdrive_url(url: str) -> bool:
    """Return True if the URL points to Google Drive."""
    lowered = (url or "").strip().lower()
    return any(host in lowered for host in GDRIVE_HOSTS)


def parse_gdrive_url(url: str) -> tuple[str, str | None]:
    """Parse a Google Drive URL into (kind, id).

    Returns:
      ('folder', folder_id), ('file', file_id), or ('unknown', id_or_none)
    """
    url = (url or "").strip()
    if not url:
        return "unknown", None

    # Folder match patterns
    folder_match = re.search(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]+)", url)
    if folder_match:
        return "folder", folder_match.group(1)

    # File match patterns (/file/d/<id>)
    file_match = re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if file_match:
        return "file", file_match.group(1)

    # Usercontent / thumbnail match patterns (/d/<id>)
    lh3_match = re.search(r"googleusercontent\.com/d/([a-zA-Z0-9_-]+)", url)
    if lh3_match:
        return "file", lh3_match.group(1)

    # Query param id=...
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        item_id = qs["id"][0]
        # Check path or query hints
        if "folder" in parsed.path.lower() or "folder" in url.lower():
            return "folder", item_id
        return "file", item_id

    # Fallback ID extraction for raw Google Drive IDs
    if re.match(r"^[a-zA-Z0-9_-]{20,}$", url):
        return "file", url

    return "unknown", None


def get_gdrive_download_url(file_id: str) -> str:
    """Return standard direct download URL for a file ID."""
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def scan_gdrive_folder(folder_url_or_id: str) -> list[dict[str, Any]]:
    """Scan a public/shared Google Drive folder and list all contained files.

    Returns a list of dicts with keys:
      - id: str
      - name: str
      - url: str (shareable file URL)
      - type: 'image' | 'video' | 'other'
      - extension: str
    """
    import gdown

    kind, target_id = parse_gdrive_url(folder_url_or_id)
    folder_id = target_id or folder_url_or_id
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    log.info("Scanning Google Drive folder: %s", folder_id)

    try:
        # Use gdown's skip_download=True to inspect folder metadata without downloading
        raw_files = gdown.download_folder(url=folder_url, skip_download=True, quiet=True)
    except Exception as exc:
        log.warning("gdown folder scan failed for %s: %s", folder_id, exc)
        raise RuntimeError(f"Could not scan Google Drive folder: {exc}") from exc

    results: list[dict[str, Any]] = []
    for item in raw_files or []:
        file_id = getattr(item, "id", None)
        path = getattr(item, "path", None) or getattr(item, "local_path", None) or ""
        if not file_id:
            continue

        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            file_type = "image"
        elif ext in VIDEO_EXTENSIONS:
            file_type = "video"
        else:
            file_type = "other"

        results.append({
            "id": file_id,
            "name": filename or f"gdrive_{file_id}",
            "url": f"https://drive.google.com/file/d/{file_id}/view",
            "type": file_type,
            "extension": ext,
        })

    log.info("Google Drive scan found %d file(s) in folder %s", len(results), folder_id)
    return results


def fetch_gdrive_image(file_id_or_url: str, timeout: float = 30.0) -> tuple[np.ndarray, str]:
    """Fetch an image from Google Drive directly into a BGR numpy array.

    Supports JPEG, PNG, WEBP, HEIC, and other image formats.
    Returns:
      (image_bgr_array, title_or_filename)
    """
    import cv2
    from PIL import Image
    import gdown

    kind, file_id = parse_gdrive_url(file_id_or_url)
    file_id = file_id or file_id_or_url

    # Candidate direct download endpoints (full-res Google image renderer first)
    endpoints = [
        f"https://lh3.googleusercontent.com/d/{file_id}=s0",
        f"https://lh3.googleusercontent.com/d/{file_id}",
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?id={file_id}&export=download",
    ]

    raw_bytes: bytes | None = None
    title = f"gdrive_{file_id}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    # Fast direct fetch
    for ep in endpoints:
        try:
            resp = requests.get(ep, headers=headers, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 100:
                # Check for Content-Disposition header filename
                cd = resp.headers.get("Content-Disposition", "")
                if "filename=" in cd:
                    match = re.search(r'filename=["\']?([^"\';]+)["\']?', cd)
                    if match:
                        title = match.group(1).strip()
                # Ensure it's not an HTML error / confirmation page
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    # Test if bytes can be decoded as image
                    test_arr = np.frombuffer(resp.content, dtype=np.uint8)
                    test_img = cv2.imdecode(test_arr, cv2.IMREAD_COLOR)
                    if test_img is not None:
                        return test_img, title
                    try:
                        pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        return bgr_img, title
                    except Exception:
                        pass
                    raw_bytes = resp.content
        except Exception as exc:
            log.debug("Direct fetch endpoint %s failed: %s", ep, exc)

    # Fallback to gdown memory download if direct endpoints didn't return image data
    if raw_bytes is None:
        log.info("Fetching Google Drive file %s via gdown memory stream...", file_id)
        buf = io.BytesIO()
        try:
            gdown.download(id=file_id, output=buf, quiet=True)
            raw_bytes = buf.getvalue()
        except Exception as exc:
            log.warning("gdown download failed for %s: %s", file_id, exc)
            raise RuntimeError(
                f"Cannot access Google Drive file ({file_id}). Ensure link is valid and sharing is set to 'Anyone with the link'."
            ) from exc

    if not raw_bytes:
        raise RuntimeError(f"Empty response received for Google Drive file {file_id}")

    # Decode image to BGR numpy array
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image, title

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return image, title
    except Exception as exc:
        raise RuntimeError(f"Could not decode image from Google Drive file {file_id}: {exc}") from exc
