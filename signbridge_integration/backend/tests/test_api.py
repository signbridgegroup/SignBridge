"""Lightweight backend smoke tests (unittest + FastAPI TestClient, no
extra test-runner dependency beyond what's already installed).

Run with:
  C:\\SignBridge_Project\\.venv\\Scripts\\python.exe -m unittest backend.tests.test_api -v
(from signbridge_integration/, with backend on PYTHONPATH — see docs/TESTING.md)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import dataclasses
from unittest.mock import patch

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def _force_no_groq_key():
    """Deterministically forces the "no GROQ_API_KEY configured" state for
    one test, regardless of whether this machine's real backend/.env has a
    real key. Settings is a frozen dataclass, so the binding used by each
    consuming module is replaced (never mutated, never read/logged), and
    nothing here ever touches the real .env file or its contents.
    """
    from app.config import settings as real_settings

    fake_settings = dataclasses.replace(real_settings, groq_api_key=None)
    return (
        patch("app.main.settings", fake_settings),
        patch("app.rag.service.settings", fake_settings),
    )


class HealthAndConfigTests(unittest.TestCase):
    def test_health_endpoint_works_regardless_of_groq_state(self) -> None:
        # Per the API contract, /api/health must work whether or not a key
        # is configured - this only checks the contract (field present,
        # boolean-typed, endpoint doesn't crash), not a specific value,
        # since that value legitimately differs by machine/deployment.
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("checkpoint", body)
        self.assertIn("groq_configured", body)
        self.assertIsInstance(body["groq_configured"]["ok"], bool)

    def test_health_reports_groq_not_configured_when_key_missing(self) -> None:
        patch_main, _ = _force_no_groq_key()
        with patch_main:
            response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["groq_configured"]["ok"])

    def test_config_endpoint(self) -> None:
        response = client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["supported_subjects"], ["Stack", "Queue", "Pointers", "Database"])
        self.assertIn("auto", body["recognition_modes"])


class MotionsTests(unittest.TestCase):
    def test_list_motions(self) -> None:
        response = client.get("/api/motions")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreater(body["total_manifest_entries"], 0)

    def test_status_endpoint(self) -> None:
        response = client.get("/api/motions/status")
        self.assertEqual(response.status_code, 200)

    def test_ready_motion_resolves(self) -> None:
        response = client.get("/api/motions/jordanian_it/STACK")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["format"], "signbridge-motion-v1")

    def test_missing_motion_returns_404_not_500(self) -> None:
        response = client.get("/api/motions/jordanian_it/NOT_A_REAL_TOKEN")
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "motion_not_found")

    def test_path_traversal_is_rejected(self) -> None:
        response = client.get("/api/motions/jordanian_it/..%2F..%2F..%2Fetc")
        self.assertIn(response.status_code, (400, 404))

    def test_path_traversal_dotdot_in_dataset_rejected(self) -> None:
        response = client.get("/api/motions/..%2F..%2Fwindows/STACK")
        self.assertIn(response.status_code, (400, 404))


class AskWithoutGroqTests(unittest.TestCase):
    def test_ask_returns_clear_configuration_error(self) -> None:
        # Forced deterministically (see _force_no_groq_key) so this test's
        # result does not depend on whether this particular machine has a
        # real backend/.env - it never touches the real key or file.
        _, patch_rag_service = _force_no_groq_key()
        with patch_rag_service:
            response = client.post("/api/ask", json={"question": "ما هو الـ Stack؟"})
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["code"], "configuration_error")
        self.assertIn("GROQ_API_KEY", body["detail"])


if __name__ == "__main__":
    unittest.main()
