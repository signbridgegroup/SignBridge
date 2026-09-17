"""Confirms RAG failure (no GROQ_API_KEY) never crashes recognition output."""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.pipeline import recognize_and_answer  # noqa: E402

SAMPLE_VIDEO = Path(r"C:\SignBridge_Project\data\raw\sign_videos\Stack\Stack_07_1.mp4")


class PipelineDegradationTest(unittest.TestCase):
    def test_recognition_survives_missing_groq_key(self) -> None:
        # Forced deterministically regardless of whether this machine's real
        # backend/.env has a real key configured - never touches the real
        # key/file, and this specifically prevents an accidental real Groq
        # network call from an otherwise-offline test.
        from app.config import settings as real_settings
        from app.rag import service as rag_service

        fake_settings = dataclasses.replace(real_settings, groq_api_key=None)
        with patch("app.rag.service.settings", fake_settings):
            self.assertIs(rag_service.settings, fake_settings)
            result = recognize_and_answer(SAMPLE_VIDEO, mode="it")
        self.assertEqual(result.recognition.predicted_label, "Stack")
        self.assertFalse(result.answer.educational_answer_available)
        self.assertIsNotNone(result.answer.unavailable_reason)
        print(f"\n[smoke] recognized_gloss={result.recognized_gloss!r} "
              f"question={result.question!r} "
              f"unavailable_reason={result.answer.unavailable_reason!r}")


if __name__ == "__main__":
    unittest.main()
