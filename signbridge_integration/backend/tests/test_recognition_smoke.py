"""One-sample recognition smoke test against the real final checkpoint.

This is the "not yet revalidated inside the new integrated backend" check
called for in the brief — it loads the checkpoint once and runs a single
known-good Jordanian IT clip (Stack, signer 07) through the full traced
inference path exactly as the backend would.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.recognition.service import recognition_service  # noqa: E402

SAMPLE_VIDEO = Path(r"C:\SignBridge_Project\data\raw\sign_videos\Stack\Stack_07_1.mp4")


class RecognitionSmokeTest(unittest.TestCase):
    def test_stack_video_recognized(self) -> None:
        self.assertTrue(SAMPLE_VIDEO.is_file(), f"Sample video missing: {SAMPLE_VIDEO}")

        started = time.perf_counter()
        result = recognition_service.recognize(SAMPLE_VIDEO, mode="it", top_k=5)
        elapsed = time.perf_counter() - started

        print(f"\n[smoke] resolved_mode={result['resolved_mode']} "
              f"predicted={result['predicted_label']} "
              f"confidence={result['confidence']:.3f} "
              f"device={result['device']} elapsed={elapsed:.2f}s")
        print(f"[smoke] top_k={result['top_k']}")

        self.assertEqual(result["resolved_mode"], "it")
        self.assertIsNotNone(result["predicted_label"])
        self.assertGreater(result["frame_count"], 0)


if __name__ == "__main__":
    unittest.main()
