"""Tests for dual-engine database (SQLite & Supabase PostgreSQL), storage, and configuration."""

import os
import unittest
from pathlib import Path

from app import config, db, storage
from app.config import Config


class TestDatabaseAndStorage(unittest.TestCase):

    def setUp(self):
        # Ensure fresh local test database
        if hasattr(db._local, "sqlite_conn") and db._local.sqlite_conn:
            try:
                db._local.sqlite_conn.close()
            except Exception:
                pass
        test_db_path = config.ROOT / "data" / "test_sang_det.db"
        if test_db_path.exists():
            try:
                test_db_path.unlink()
            except OSError:
                pass
        db.DB_PATH = test_db_path
        db._local = type("Local", (), {})()
        db.init()
        db.clear_all_plates()
        db.db_wrapper().execute("DELETE FROM videos")

    def tearDown(self):
        if hasattr(db._local, "sqlite_conn") and db._local.sqlite_conn:
            try:
                db._local.sqlite_conn.close()
            except Exception:
                pass
        test_db_path = config.ROOT / "data" / "test_sang_det.db"
        if test_db_path.exists():
            try:
                test_db_path.unlink()
            except OSError:
                pass


    def test_db_initialization_and_tables(self):
        """Verify SQLite tables and indexes initialize cleanly."""
        state_test = {"active": True, "count": 42}
        db.set_state("test_key", state_test)
        self.assertEqual(db.get_state("test_key"), state_test)
        self.assertIsNone(db.get_state("non_existent_key"))

    def test_videos_queue_operations(self):
        """Test queuing, claiming, bumping, and listing videos."""
        added, skipped = db.add_videos(["https://example.com/video1.mp4", "https://example.com/video2.mp4"])
        self.assertEqual(added, 2)
        self.assertEqual(skipped, 0)

        # Duplicate addition check
        added2, skipped2 = db.add_videos(["https://example.com/video1.mp4"])
        self.assertEqual(added2, 0)
        self.assertEqual(skipped2, 1)

        # Claim video
        claimed = db.claim_next_video()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["status"], db.STATUS_PROCESSING)

        # Bump counters
        db.bump_video(claimed["id"], frames_processed=10, plates_saved=2)
        v = db.get_video(claimed["id"])
        self.assertEqual(v["frames_processed"], 10)
        self.assertEqual(v["plates_saved"], 2)

    def test_plates_records_and_totals(self):
        """Test recording plates with storage_url and verifying totals calculation."""
        db.add_videos(["https://example.com/stream.mp4"])
        v = db.claim_next_video()

        plate_id = db.record_plate(
            video_id=v["id"],
            filename="plate_001.jpg",
            frame_index=15,
            timestamp_s=1.5,
            confidence=0.95,
            box=(10, 20, 100, 50),
            width=90,
            height=30,
            blur_score=55.2,
            phash="a1b2c3d4e5f60718",
            nbytes=4096,
            storage_url="https://xyz.supabase.co/storage/v1/object/public/plates/plate_001.jpg",
        )
        self.assertGreater(plate_id, 0)

        # Retrieve plate
        p = db.get_plate(plate_id)
        self.assertIsNotNone(p)
        self.assertEqual(p["filename"], "plate_001.jpg")
        self.assertEqual(p["storage_url"], "https://xyz.supabase.co/storage/v1/object/public/plates/plate_001.jpg")
        self.assertEqual(p["ocr_status"], "unlabeled")

        # Update plate label
        db.update_plate_label(plate_id, "MH12AB1234", ocr_status="labeled")
        p_updated = db.get_plate(plate_id)
        self.assertEqual(p_updated["plate_text"], "MH12AB1234")
        self.assertEqual(p_updated["ocr_status"], "labeled")

        db.bump_video(v["id"], plates_saved=1)

        # Check totals query (ANSI SQL compatible)
        stats = db.totals()
        self.assertIn("videos", stats)
        self.assertIn("plates", stats)
        self.assertEqual(stats["plates"], 1)

        label_stats = db.get_label_stats()
        self.assertEqual(label_stats["total_plates"], 1)
        self.assertEqual(label_stats["labeled"], 1)


    def test_custom_plate_upload_record(self):
        """Test recording manual / batch uploaded images."""
        custom_id = db.record_custom_plate(
            filename="upload_custom_123.jpg",
            width=200,
            height=60,
            nbytes=8192,
            storage_url="https://xyz.supabase.co/storage/v1/object/public/plates/upload_custom_123.jpg",
        )
        self.assertGreater(custom_id, 0)
        p = db.get_plate(custom_id)
        self.assertEqual(p["filename"], "upload_custom_123.jpg")
        self.assertEqual(p["storage_url"], "https://xyz.supabase.co/storage/v1/object/public/plates/upload_custom_123.jpg")

    def test_supabase_storage_helpers(self):
        """Verify Supabase Storage URL formatting and helper functions."""
        test_cfg = Config({
            "supabase": {
                "url": "https://testproject.supabase.co",
                "key": "testkey123",
                "bucket": "plates",
            }
        })
        self.assertTrue(storage.is_supabase_storage_enabled(test_cfg))
        pub_url = storage.get_supabase_public_url("my_crop.jpg", test_cfg)
        self.assertEqual(
            pub_url,
            "https://testproject.supabase.co/storage/v1/object/public/plates/my_crop.jpg",
        )

    def test_config_env_overrides(self):
        """Test that environment variables override configuration values."""
        os.environ["SUPABASE_URL"] = "https://envproject.supabase.co"
        os.environ["SUPABASE_KEY"] = "secret_env_key"
        os.environ["SUPABASE_BUCKET"] = "custom-plates"
        os.environ["PORT"] = "9000"

        try:
            cfg = config.load(force=True)
            self.assertEqual(cfg.get("supabase.url"), "https://envproject.supabase.co")
            self.assertEqual(cfg.get("supabase.key"), "secret_env_key")
            self.assertEqual(cfg.get("supabase.bucket"), "custom-plates")
            self.assertEqual(cfg.get("server.port"), 9000)
        finally:
            os.environ.pop("SUPABASE_URL", None)
            os.environ.pop("SUPABASE_KEY", None)
            os.environ.pop("SUPABASE_BUCKET", None)
            os.environ.pop("PORT", None)


if __name__ == "__main__":
    unittest.main()
