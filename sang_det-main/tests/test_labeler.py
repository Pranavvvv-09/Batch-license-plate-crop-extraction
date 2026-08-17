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
    def setUp(self):
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

    def test_local_folder_scan_and_import(self):
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

    def test_folder_api_endpoints(self):
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

    def test_reset_labeler_endpoint(self):
        from starlette.testclient import TestClient
        from app.server import app

        # Create a test plate with a label
        plate_id = db.record_custom_plate(
            filename="test_reset_plate_99.jpg",
            width=200,
            height=50,
            nbytes=512,
            plate_text="RESET 123",
            ocr_status="labeled",
        )
        self.assertGreater(plate_id, 0)

        with TestClient(app) as client:
            # Test reset labels mode
            res = client.post("/api/labeler/reset", json={"mode": "reset_labels", "delete_files": False})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["ok"])

            # Verify plate became unlabeled
            plate = db.get_plate(plate_id)
            self.assertIsNotNone(plate)
            self.assertEqual(plate["ocr_status"], "unlabeled")
            self.assertIsNone(plate["plate_text"])

            # Test clear all mode
            res_clear = client.post("/api/labeler/reset", json={"mode": "clear_all", "delete_files": False})
            self.assertEqual(res_clear.status_code, 200)
            data_clear = res_clear.json()
            self.assertTrue(data_clear["ok"])

            # Verify plates table is now 0
            stats = db.get_label_stats()
            self.assertEqual(stats["total_plates"], 0)

    def test_rate_limiter(self):
        """Test rate limiter pacing and backoff penalty mechanism."""
        limiter = labeler.RateLimiter(rps=100.0, min_delay_s=0.01)
        limiter.wait()
        self.assertEqual(limiter.rate_limit_hits, 0)

        # Trigger rate limit penalty
        limiter.report_rate_limit(0.05)
        self.assertEqual(limiter.rate_limit_hits, 1)
        t0 = labeler.time.time()
        limiter.wait()
        elapsed = labeler.time.time() - t0
        self.assertGreaterEqual(elapsed, 0.04)

    @patch.object(labeler.NimLabeler, "scan_plate_record")
    def test_batch_labeler_queue_and_resumption(self, mock_scan):
        """Test queue-based batch processing, memory safety, and resuming."""
        def fake_scan(plate_id, api_key=None):
            db.update_plate_label(plate_id, "RESUME 100", ocr_status="labeled")
            return labeler.ScanResult(ok=True, plate_text="RESUME 100")
        mock_scan.side_effect = fake_scan


        # Create 3 test plates (2 unlabeled, 1 already labeled)
        p1 = db.record_custom_plate(filename="batch_p1.jpg", plate_text=None, ocr_status="unlabeled")
        p2 = db.record_custom_plate(filename="batch_p2.jpg", plate_text="ALREADY LABELED", ocr_status="labeled")
        p3 = db.record_custom_plate(filename="batch_p3.jpg", plate_text=None, ocr_status="unlabeled")

        try:
            batch = labeler.BatchLabeler()
            batch._run_batch(limit=10, force_all=False, retry_errors=True, api_key="nvapi-mock-test")

            st = batch.status()
            self.assertEqual(st["processed"], 2)  # Only processed the 2 unlabeled ones!
            self.assertEqual(st["succeeded"], 2)

            # Check that p1 and p3 were updated
            plate1 = db.get_plate(p1)
            plate3 = db.get_plate(p3)
            self.assertEqual(plate1["plate_text"], "RESUME 100")
            self.assertEqual(plate3["plate_text"], "RESUME 100")
        finally:
            db.delete_plate(p1)
            db.delete_plate(p2)
            db.delete_plate(p3)


if __name__ == "__main__":
    unittest.main()

