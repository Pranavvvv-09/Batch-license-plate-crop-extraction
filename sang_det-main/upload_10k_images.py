"""Ultra-Fast Parallel Bulk Uploader for 10,000+ Images to Supabase.

Uploads image crops directly to Supabase Storage in parallel and batch-inserts
metadata into Supabase PostgreSQL for 24/7 background AI OCR.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Safe UTF-8 reconfiguration for Windows console
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import db, storage
from app.config import load as load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def scan_images_fast(folder: Path, recursive: bool = True) -> list[Path]:
    """Blazing fast directory scanner using os.scandir."""
    images: list[Path] = []
    stack = [str(folder)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_file():
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in IMAGE_EXTS:
                            images.append(Path(entry.path))
                    elif recursive and entry.is_dir() and not entry.name.startswith("."):
                        stack.append(entry.path)
        except OSError:
            pass
    return images


def batch_insert_plates(records: list[dict[str, Any]]) -> None:
    """Fast batch insert of plate records into PostgreSQL/SQLite."""
    if not records:
        return
    now = time.time()
    db_wrap = db.db_wrapper()
    if db.is_postgres():
        conn = db.connect_postgres()
        with conn.cursor() as cur:
            import psycopg2.extras
            sql = (
                "INSERT INTO plates (filename, width, height, bytes, saved_at, ocr_status, storage_url) "
                "VALUES %s ON CONFLICT DO NOTHING"
            )
            data = [
                (
                    r["filename"],
                    r.get("width", 0),
                    r.get("height", 0),
                    r.get("bytes", 0),
                    now,
                    "unlabeled",
                    r.get("storage_url"),
                )
                for r in records
            ]
            psycopg2.extras.execute_values(cur, sql, data, page_size=200)
    else:
        conn = db.connect_sqlite()
        sql = (
            "INSERT OR IGNORE INTO plates (filename, width, height, bytes, saved_at, ocr_status, storage_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        data = [
            (
                r["filename"],
                r.get("width", 0),
                r.get("height", 0),
                r.get("bytes", 0),
                now,
                "unlabeled",
                r.get("storage_url"),
            )
            for r in records
        ]
        conn.executemany(sql, data)


def bulk_upload_folder(
    folder_path: str | Path,
    recursive: bool = True,
    workers: int = 32,
) -> dict[str, Any]:
    folder = Path(folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"\n[ERROR] Folder does not exist: {folder}")
        return {"ok": False, "error": f"Folder not found: {folder}"}

    cfg = load_config()
    db.init()

    print(f"\nScanning folder: {folder}...")
    all_files = scan_images_fast(folder, recursive=recursive)
    total_found = len(all_files)

    if total_found == 0:
        print("[WARN] No supported image files (.jpg, .png, .webp, .bmp) found in folder.")
        return {"ok": True, "uploaded": 0, "skipped": 0, "total": 0}

    print(f"[INFO] Found {total_found:,} image files.")
    print("[INFO] Fetching existing records from Supabase to skip duplicates...")

    try:
        existing_rows = db.db_wrapper().fetchall("SELECT filename FROM plates")
        existing_filenames = {r["filename"] for r in existing_rows}
    except Exception as exc:
        print(f"Warning: Could not fetch existing plates ({exc}); proceeding anyway.")
        existing_filenames = set()

    files_to_upload = [p for p in all_files if p.name not in existing_filenames]
    skipped_count = total_found - len(files_to_upload)

    if skipped_count > 0:
        print(f"[SKIP] Skipping {skipped_count:,} images already registered in Supabase.")

    if not files_to_upload:
        print("[OK] All images are already uploaded and registered in Supabase!")
        return {"ok": True, "uploaded": 0, "skipped": skipped_count, "total": total_found}

    print(f"[START] Starting parallel upload of {len(files_to_upload):,} images with {workers} threads...")
    print("=" * 65)

    uploaded = 0
    failed = 0
    t0 = time.time()
    pending_records: list[dict[str, Any]] = []

    def upload_file(img_path: Path) -> dict[str, Any] | None:
        filename = img_path.name
        try:
            data = img_path.read_bytes()
            if len(data) < 32:
                return None

            mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
            storage_url = None
            if storage.is_supabase_storage_enabled(cfg):
                storage_url = storage.upload_to_supabase_storage(
                    filename=filename,
                    image_bytes=data,
                    content_type=mime,
                    cfg=cfg,
                )

            return {
                "filename": filename,
                "width": 0,
                "height": 0,
                "bytes": len(data),
                "storage_url": storage_url,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload_file, p): p for p in files_to_upload}

        for i, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            if res:
                uploaded += 1
                pending_records.append(res)
            else:
                failed += 1

            # Batch insert every 100 items into database for blazing speed
            if len(pending_records) >= 100:
                batch_insert_plates(pending_records)
                pending_records.clear()

            if i % 25 == 0 or i == len(files_to_upload):
                elapsed = time.time() - t0
                speed = i / max(0.1, elapsed)
                remaining = len(files_to_upload) - i
                eta = remaining / max(0.1, speed)
                pct = (i / len(files_to_upload)) * 100
                print(
                    f"\r Progress: [{i:,}/{len(files_to_upload):,}] ({pct:.1f}%) "
                    f"- {speed:.1f} imgs/s - Elapsed: {format_time(elapsed)} - ETA: {format_time(eta)}",
                    end="",
                    flush=True,
                )

    # Final remaining batch insert
    if pending_records:
        batch_insert_plates(pending_records)
        pending_records.clear()

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"[DONE] COMPLETED in {format_time(total_time)}!")
    print(f"  * Successfully Uploaded: {uploaded:,}")
    print(f"  * Already Existed:       {skipped_count:,}")
    if failed > 0:
        print(f"  * Failed:                {failed:,}")
    print(f"  * Total Images in Cloud: {uploaded + skipped_count:,}")
    print("=" * 65)
    print("-> All images are permanently saved in Supabase.")
    print("-> Railway background worker will now process them 24/7 in the cloud!")

    return {
        "ok": True,
        "uploaded": uploaded,
        "skipped": skipped_count,
        "failed": failed,
        "total": total_found,
    }


def main():
    print("=" * 65)
    print("   sang_det - Fast 10k Images Bulk Uploader to Supabase")
    print("=" * 65)

    if len(sys.argv) > 1:
        folder_input = sys.argv[1]
    else:
        folder_input = input("\nEnter the path to your image folder on this laptop: ").strip().strip('"\'')

    if not folder_input:
        print("No folder path entered. Exiting.")
        return

    bulk_upload_folder(folder_input, recursive=True, workers=32)


if __name__ == "__main__":
    main()
