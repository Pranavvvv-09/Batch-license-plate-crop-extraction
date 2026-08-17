import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unittest
import numpy as np
import cv2
import tempfile

from app.gdrive import is_gdrive_url, parse_gdrive_url, get_gdrive_download_url
from app.resolver import _looks_image, _looks_direct, StreamSource
from app.frames import iter_frames
from app.config import load as load_config


class TestGDriveIntegration(unittest.TestCase):
    def test_gdrive_url_detection(self):
        self.assertTrue(is_gdrive_url("https://drive.google.com/file/d/12345/view"))
        self.assertTrue(is_gdrive_url("https://drive.google.com/drive/folders/abcdef"))
        self.assertTrue(is_gdrive_url("https://lh3.googleusercontent.com/d/12345"))
        self.assertTrue(is_gdrive_url("https://docs.google.com/uc?id=xyz"))
        self.assertFalse(is_gdrive_url("https://youtube.com/watch?v=123"))
        self.assertFalse(is_gdrive_url("https://example.com/video.mp4"))

    def test_gdrive_url_parsing(self):
        self.assertEqual(parse_gdrive_url("https://drive.google.com/file/d/abc-123_xyz/view"), ("file", "abc-123_xyz"))
        self.assertEqual(parse_gdrive_url("https://drive.google.com/drive/folders/folder_id_123"), ("folder", "folder_id_123"))
        self.assertEqual(parse_gdrive_url("https://drive.google.com/drive/u/0/folders/folder_id_456"), ("folder", "folder_id_456"))
        self.assertEqual(parse_gdrive_url("https://drive.google.com/open?id=folder_id_789&folders=1"), ("folder", "folder_id_789"))
        self.assertEqual(parse_gdrive_url("https://drive.google.com/open?id=file_id_999"), ("file", "file_id_999"))

    def test_download_url_generation(self):
        url = get_gdrive_download_url("test_file_id")
        self.assertIn("test_file_id", url)
        self.assertIn("drive.usercontent.google.com", url)

    def test_looks_image(self):
        self.assertTrue(_looks_image("https://example.com/test.jpg"))
        self.assertTrue(_looks_image("https://example.com/path/photo.PNG?w=100"))
        self.assertTrue(_looks_image("https://example.com/image.webp"))
        self.assertFalse(_looks_image("https://example.com/video.mp4"))
        self.assertFalse(_looks_image("https://drive.google.com/file/d/123/view"))

    def test_iter_image(self):
        cfg = load_config()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
            cv2.imwrite(f.name, dummy_img)
            tmp_path = f.name

        try:
            source = StreamSource(url=tmp_path, title="test_dummy", kind="image")
            frames = list(iter_frames(source, cfg))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].index, 0)
            self.assertEqual(frames[0].timestamp_s, 0.0)
            self.assertEqual(frames[0].image.shape[:2], (100, 200))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_api_endpoints(self):
        from starlette.testclient import TestClient
        from app.server import app

        with TestClient(app) as client:
            # Test scan with single file
            res = client.post("/api/gdrive/scan", json={"url": "https://drive.google.com/file/d/test_file_id/view"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["kind"], "file")
            self.assertEqual(len(data["files"]), 1)

            # Test invalid URL
            res_bad = client.post("/api/gdrive/scan", json={"url": "https://invalid-non-gdrive.com/123"})
            self.assertEqual(res_bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
