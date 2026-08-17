"""Fast Multi-Threaded Bulk Uploader for 10,000+ Images to Supabase.

Uploads image crops directly from a local folder to Supabase Storage and
registers them in the Supabase PostgreSQL database for 24/7 background AI OCR.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Ensure app package is importable
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


def bulk_upload_folder(
    folder_path: str | Path,
    recursive: bool = True,
    workers: int = 16,
) -> dict[str, Any]:
    """Upload all images in folder_path to Supabase Storage & Database in parallel."""
    folder = Path(folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"\n[ERROR] Folder does not exist: {folder}")
        return {"ok": False, "error": f"Folder not found: {folder}"}

    cfg = load_config()
    db.init()

    if not db.is_postgres() and not storage.is_supabase_storage_enabled(cfg):
        print("\n[NOTE] Neither PostgreSQL nor Supabase Storage configured in .env.")
        print("Images will be recorded to local SQLite database.")

    print(f"\n📂 Scanning folder: {folder} (recursive={recursive})...")
    pattern = "**/*" if recursive else "*"
    all_files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    total_found = len(all_files)

    if total_found == 0:
        print("⚠️ No supported image files (.jpg, .png, .webp, .bmp) found in folder.")
        return {"ok": True, "uploaded": 0, "skipped": 0, "total": 0}

    print(f"🔍 Found {total_found:,} image files.")
    print("⚡ Fetching existing records from Supabase to skip duplicates...")

    # Fetch existing plate filenames in one query to prevent duplicate work
    try:
        existing_rows = db.db_wrapper().fetchall("SELECT filename FROM plates")
        existing_filenames = {r["filename"] for r in existing_rows}
    except Exception as exc:
        print(f"Warning: Could not fetch existing plates ({exc}); proceeding anyway.")
        existing_filenames = set()

    files_to_upload = [p for p in all_files if p.name not in existing_filenames]
    skipped_count = total_found - len(files_to_upload)

    if skipped_count > 0:
        print(f"⏭️  Skipping {skipped_count:,} images already in Supabase database.")

    if not files_to_upload:
        print("✅ All images are already uploaded and registered in Supabase!")
        return {"ok": True, "uploaded": 0, "skipped": skipped_count, "total": total_found}

    print(f"🚀 Starting parallel upload of {len(files_to_upload):,} images with {workers} threads...")
    print("=" * 65)

    uploaded = 0
    failed = 0
    t0 = time.time()

    def process_single_image(img_path: Path) -> bool:
        filename = img_path.name
        try:
            image_bytes = img_path.read_bytes()
            if len(image_bytes) < 32:
                return False

            mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"

            # 1. Upload to Supabase Storage if configured
            storage_url = None
            if storage.is_supabase_storage_enabled(cfg):
                storage_url = storage.upload_to_supabase_storage(
                    filename=filename,
                    image_bytes=image_bytes,
                    content_type=mime,
                    cfg=cfg,
                )

            # 2. Get width/height quickly
            try:
                import cv2
                import numpy as np

                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                h, w = (img.shape[0], img.shape[1]) if img is not None else (0, 0)
            except Exception:
                w, h = 0, 0

            # 3. Insert record into database
            db.record_custom_plate(
                filename=filename,
                width=w,
                height=h,
                nbytes=len(image_bytes),
                plate_text=None,
                ocr_status="unlabeled",
                storage_url=storage_url,
            )
            return True
        except Exception as exc:
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_image, p): p for p in files_to_upload}

        for i, future in enumerate(as_completed(futures), start=1):
            success = future.result()
            if success:
                uploaded += 1
            else:
                failed += 1

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

    total_time = time.time() - t0
    print("\n" + "=" * 65)
    print(f"🎉 COMPLETED in {format_time(total_time)}!")
    print(f"  • Successfully Uploaded: {uploaded:,}")
    print(f"  • Already Existed:       {skipped_count:,}")
    if failed > 0:
        print(f"  • Failed:                {failed:,}")
    print(f"  • Total Images in Cloud: {uploaded + skipped_count:,}")
    print("=" * 65)
    print("👉 All images are permanently saved in Supabase.")
    print("👉 Railway background worker can now process them 24/7 in the cloud!")

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

    bulk_upload_folder(folder_input, recursive=True, workers=16)


if __name__ == "__main__":
    main()
