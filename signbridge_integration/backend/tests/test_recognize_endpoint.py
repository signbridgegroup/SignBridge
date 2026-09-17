"""Full HTTP round-trip for POST /api/recognize, plus a temp-file cleanup check."""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
SAMPLE_VIDEO = Path(r"C:\SignBridge_Project\data\raw\sign_videos\Stack\Stack_07_1.mp4")


class RecognizeEndpointTest(unittest.TestCase):
    def test_recognize_http_roundtrip_and_temp_cleanup(self) -> None:
        before = set(Path(tempfile.gettempdir()).glob("signbridge_upload_*"))

        # Forced deterministically regardless of whether this machine's real
        # backend/.env has a real key configured - never touches the real
        # key/file, and prevents an accidental real Groq network call from
        # this otherwise-offline recognition test.
        from app.config import settings as real_settings

        fake_settings = dataclasses.replace(real_settings, groq_api_key=None)
        with patch("app.rag.service.settings", fake_settings):
            with SAMPLE_VIDEO.open("rb") as handle:
                response = client.post(
                    "/api/recognize",
                    files={"video": ("Stack_07_1.mp4", handle, "video/mp4")},
                    data={"mode": "it", "top_k": "5"},
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["recognition"]["predicted_label"], "Stack")
        self.assertFalse(body["answer"]["educational_answer_available"])

        after = set(Path(tempfile.gettempdir()).glob("signbridge_upload_*"))
        self.assertEqual(before, after, "Temporary upload directory was not cleaned up")

    def test_recognize_rejects_unsupported_extension(self) -> None:
        response = client.post(
            "/api/recognize",
            files={"video": ("clip.txt", b"not a video", "text/plain")},
            data={"mode": "auto"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
