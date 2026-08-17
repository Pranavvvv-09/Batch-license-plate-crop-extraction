import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db, labeler
from app.config import load as load_config


def make_valid_test_jpeg() -> bytes:
    img = np.zeros((40, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


class TestLabeler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        import threading
        from app import config
        cls._orig_db_url = os.environ.pop("DATABASE_URL", None)
        cls._orig_supabase_url = os.environ.pop("SUPABASE_URL", None)
        cls._orig_db_engine = getattr(db, "_db_engine", None)
        db._db_engine = "sqlite"
        config._instance = None
        db._local = threading.local()
        db.init()

    @classmethod
    def tearDownClass(cls):
        import os
        import threading
        from app import config
        if hasattr(db._local, "sqlite_conn") and db._local.sqlite_conn:
            try:
                db._local.sqlite_conn.close()
            except Exception:
                pass
        if cls._orig_db_url:
            os.environ["DATABASE_URL"] = cls._orig_db_url
        if cls._orig_supabase_url:
            os.environ["SUPABASE_URL"] = cls._orig_supabase_url
        db._db_engine = cls._orig_db_engine
        config._instance = None
        db._local = threading.local()

    def setUp(self):
        db._db_engine = "sqlite"
        db.init()

    def test_clean_plate_text(self):
        # Basic alphanumeric
        self.assertEqual(labeler.clean_plate_text("ABC 1234"), "ABC 1234")
        self.assertEqual(labeler.clean_plate_text("mh12de1433"), "MH12DE1433")

        # Reasoning tags <think>...</think>
        reasoning_text = "<think>I see a white license plate with black characters DL 4C AB 1234.</think>DL 4C AB 1234"
        self.assertEqual(labeler.clean_plate_text(reasoning_text), "DL 4C AB 1234")

        # Markdown code block
        md_text = "```\nKA 01 AB 9999\n```"
        self.assertEqual(labeler.clean_plate_text(md_text), "KA 01 AB 9999")

        # Prefixes & quotes
        self.assertEqual(labeler.clean_plate_text('"Plate: NY-7890"'), "NY-7890")
        self.assertEqual(labeler.clean_plate_text("License Plate: CA 8XYZ123"), "CA 8XYZ123")

        # Unreadable
        self.assertEqual(labeler.clean_plate_text("unreadable"), "UNREADABLE")
        self.assertEqual(labeler.clean_plate_text("none"), "UNREADABLE")

    def test_csv_export_and_import(self):
        cfg = load_config()
        # Record a test plate
        plate_id = db.record_custom_plate(
            filename="test_csv_plate_01.jpg",
            width=200,
            height=60,
            nbytes=1024,
            plate_text="TEST 123",
            ocr_status="labeled",
        )
        self.assertGreater(plate_id, 0)
        try:
            # Export CSV
            csv_text, _ = labeler.export_labels_to_csv(cfg=cfg)
            self.assertIn("id,plate_text", csv_text)
            self.assertIn("TEST 123", csv_text)

            # Modify label via CSV import
            import_csv_data = (
                "id,plate_text\n"
                f"{plate_id},UPDATED 999\n"
            )
            res = labeler.import_labels_from_csv(import_csv_data)
            self.assertTrue(res["ok"])
            self.assertEqual(res["updated"], 1)

            # Verify DB updated
            plate = db.get_plate(plate_id)
            self.assertIsNotNone(plate)
            self.assertEqual(plate["plate_text"], "UPDATED 999")
            self.assertEqual(plate["ocr_status"], "labeled")
        finally:
            db.delete_plate(plate_id)

    @patch("urllib.request.urlopen")
    def test_nim_scan_mock(self, mock_urlopen):
        # Mock successful NIM API response
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"choices":[{"message":{"content":"<think>Plate read</think>KA 05 MN 4321"}}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = labeler.NimLabeler()
        valid_jpeg = make_valid_test_jpeg()
        res = client.scan_image_bytes(valid_jpeg, api_key="nvapi-mock-test-key")
        self.assertTrue(res.ok)
        self.assertEqual(res.plate_text, "KA 05 MN 4321")

    def test_api_labeler_endpoints(self):
        from starlette.testclient import TestClient
        from app.server import app

        with TestClient(app) as client:
            # Check labeler status
            res = client.get("/api/labeler/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["ok"])
            self.assertIn("stats", data)

            # Check plates listing
            res_plates = client.get("/api/labeler/plates?limit=10")
            self.assertEqual(res_plates.status_code, 200)
            data_plates = res_plates.json()
            self.assertTrue(data_plates["ok"])
            self.assertIsInstance(data_plates["items"], list)

            # Check export CSV endpoint
            res_csv = client.get("/api/labeler/export-csv")
            self.assertEqual(res_csv.status_code, 200)
            self.assertIn("text/csv", res_csv.headers.get("content-type", ""))

    @patch("app.storage.upload_to_supabase_storage", return_value="https://test.supabase.co/plates/test.jpg")
    def test_local_folder_scan_and_import(self, mock_upload):
        import tempfile
        valid_jpeg = make_valid_test_jpeg()

        plates_before, _ = db.list_plates(limit=1000)
        existing_ids = {p["id"] for p in plates_before}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create root images
            (temp_path / "img1.jpg").write_bytes(valid_jpeg)
            (temp_path / "img2.png").write_bytes(valid_jpeg)
            (temp_path / "ignored.txt").write_text("not an image")

            # Create subfolder with images
            sub = temp_path / "subfolder"
            sub.mkdir()
            (sub / "img3.webp").write_bytes(valid_jpeg)

            # Test non-recursive scan
            non_rec = labeler.scan_local_folder(temp_path, recursive=False)
            self.assertEqual(len(non_rec), 2)

            # Test recursive scan
            rec = labeler.scan_local_folder(temp_path, recursive=True)
            self.assertEqual(len(rec), 3)

            try:
                # Test folder import
                res = labeler.import_local_image_folder(temp_path, recursive=True, auto_scan=False)
                self.assertTrue(res["ok"])
                self.assertEqual(res["imported"], 3)
                self.assertEqual(res["total_found"], 3)
            finally:
                plates_after, _ = db.list_plates(limit=1000)
                out_dir = load_config().path("output.dir")
                for p in plates_after:
                    if p["id"] not in existing_ids:
                        db.delete_plate(p["id"])
                        f = out_dir / p["filename"]
                        if f.exists():
                            try:
                                f.unlink()
                            except OSError:
                                pass
                labeler.sync_labels_csv()

    @patch("app.storage.upload_to_supabase_storage", return_value="https://test.supabase.co/plates/test.jpg")
    def test_folder_api_endpoints(self, mock_upload):
        from starlette.testclient import TestClient
        from app.server import app
        import tempfile

        valid_jpeg = make_valid_test_jpeg()
        plates_before, _ = db.list_plates(limit=1000)
        existing_ids = {p["id"] for p in plates_before}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "crop1.jpg").write_bytes(valid_jpeg)
            (temp_path / "crop2.jpg").write_bytes(valid_jpeg)

            try:
                with TestClient(app) as client:
                    # Scan folder endpoint
                    res_scan = client.post(
                        "/api/labeler/scan-folder",
                        json={"folder_path": str(temp_path), "recursive": True},
                    )
                    self.assertEqual(res_scan.status_code, 200)
                    data_scan = res_scan.json()
                    self.assertTrue(data_scan["ok"])
                    self.assertEqual(data_scan["total"], 2)

                    # Import folder endpoint
                    res_import = client.post(
                        "/api/labeler/import-folder",
                        json={"folder_path": str(temp_path), "recursive": True, "auto_scan": False},
                    )
                    self.assertEqual(res_import.status_code, 200)
                    data_import = res_import.json()
                    self.assertTrue(data_import["ok"])
                    self.assertEqual(data_import["imported"], 2)
            finally:
                plates_after, _ = db.list_plates(limit=1000)
                out_dir = load_config().path("output.dir")
                for p in plates_after:
                    if p["id"] not in existing_ids:
                        db.delete_plate(p["id"])
                        f = out_dir / p["filename"]
                        if f.exists():
                            try:
                                f.unlink()
                            except OSError:
                                pass
                labeler.sync_labels_csv()

    @patch("app.labeler.reset_labeler_data", return_value={"ok": True, "mode": "reset_labels", "count": 1})
    def test_reset_labeler_endpoint(self, mock_reset):
        from starlette.testclient import TestClient
        from app.server import app

        with TestClient(app) as client:
            # Test reset labels mode
            res = client.post("/api/labeler/reset", json={"mode": "reset_labels", "delete_files": False})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["ok"])
            mock_reset.assert_called_once()

    def test_rate_limiter(self):
        """Test rate limiter pacing and backoff penalty mechanism."""
        limiter = labeler.RateLimiter(rps=100.0, min_delay_s=0.01)
        limiter.wait()
        self.assertEqual(limiter.rate_limit_hits, 0)

        # Trigger rate limit penalty with explicit duration
        limiter.report_rate_limit(0.05)
        self.assertEqual(limiter.rate_limit_hits, 1)
        t0 = labeler.time.time()
        limiter.wait()
        elapsed = labeler.time.time() - t0
        self.assertGreaterEqual(elapsed, 0.04)

        # Trigger rate limit penalty with default duration
        limiter.report_rate_limit()
        self.assertEqual(limiter.rate_limit_hits, 2)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_nim_scan_503_rate_limit(self, mock_urlopen, mock_sleep):
        import urllib.error
        import io

        # 503 error on first call, success on second
        mock_503 = urllib.error.HTTPError(
            url="http://mock",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b"Service Unavailable"),
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"choices":[{"message":{"content":"DL 1A AA 1111"}}]}'

        mock_urlopen.side_effect = [mock_503, mock_resp]
        mock_resp.__enter__.return_value = mock_resp

        limiter = labeler.RateLimiter(rps=100.0, min_delay_s=0.01)
        client = labeler.NimLabeler(rate_limiter=limiter)
        valid_jpeg = make_valid_test_jpeg()

        res = client.scan_image_bytes(valid_jpeg, api_key="nvapi-mock-key")
        self.assertTrue(res.ok)
        self.assertEqual(res.plate_text, "DL 1A AA 1111")
        self.assertEqual(limiter.rate_limit_hits, 1)

    @patch("app.db.get_label_stats")
    @patch("app.db.get_unlabeled_plates")
    @patch.object(labeler.NimLabeler, "scan_plate_record")
    def test_batch_labeler_queue_and_resumption(self, mock_scan, mock_unlabeled, mock_stats):
        """Test queue-based batch processing, memory safety, and resuming."""
        mock_stats.return_value = {"total_plates": 3, "labeled": 1, "unlabeled": 2, "error": 0}
        test_plates = [
            {"id": 101, "filename": "batch_p1.jpg", "ocr_status": "unlabeled"},
            {"id": 103, "filename": "batch_p3.jpg", "ocr_status": "unlabeled"},
        ]
        # Return test plates on first call, empty on second
        mock_unlabeled.side_effect = [test_plates, []]
        mock_scan.return_value = labeler.ScanResult(ok=True, plate_text="RESUME 100")

        batch = labeler.BatchLabeler()
        batch._run_batch(limit=10, force_all=False, retry_errors=True, api_key="nvapi-mock-test")

        st = batch.status()
        self.assertEqual(st["processed"], 2)
        self.assertEqual(st["succeeded"], 2)
        self.assertEqual(mock_scan.call_count, 2)

    def test_key_pool(self):
        pool = labeler.KeyPool("nvapi-key11111111111111111111, nvapi-key22222222222222222222; nvapi-key33333333333333333333")
        self.assertEqual(pool.total_count, 3)
        self.assertEqual(pool.active_count, 3)

        # Test round robin
        k1 = pool.get_next_key()
        k2 = pool.get_next_key()
        k3 = pool.get_next_key()
        k4 = pool.get_next_key()
        self.assertEqual(k1, "nvapi-key11111111111111111111")
        self.assertEqual(k2, "nvapi-key22222222222222222222")
        self.assertEqual(k3, "nvapi-key33333333333333333333")
        self.assertEqual(k4, "nvapi-key11111111111111111111")

        # Test mark exhausted
        pool.mark_exhausted("nvapi-key22222222222222222222")
        self.assertEqual(pool.active_count, 2)
        healthy = {pool.get_next_key(), pool.get_next_key()}
        self.assertNotIn("nvapi-key22222222222222222222", healthy)


if __name__ == "__main__":
    unittest.main()

