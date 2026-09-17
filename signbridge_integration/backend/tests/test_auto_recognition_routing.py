"""Tests for the genuinely automatic (content-based) recognition router.

Background: the previous "auto" mode decided isolated-vs-continuous by
checking whether "karsl" appeared in the video's file path, or whether the
file was found in a saved training-time active-length report keyed by
label/filename. A real camera recording or uploaded file is always saved
under a random temporary filename (e.g. ``upload_<uuid>.mp4``), so neither
check could ever match, and every "auto" request silently fell through to
"continuous" regardless of what was actually signed -- this is the
confirmed cause of a Stack sign being reported as the unrelated word
"جرس". These tests exercise the new content-based router
(``backend/legacy_runtime/auto_recognition_router.py`` plus
``RecognitionService._recognize_auto``) that replaces that filename check.

Real trained-checkpoint tests below use real project video assets and the
real MediaPipe/Torch pipeline -- no mocking of the recognition path itself.
Where a required real asset does not exist in this repository (a playable
continuous/Isharah example video), that is stated explicitly rather than
faked; see ContinuousPathTests below.
"""

from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BACKEND_DIR / "legacy_runtime" / "rag"
LEGACY_DIR = BACKEND_DIR / "legacy_runtime"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(LEGACY_DIR))

from auto_recognition_router import (  # noqa: E402
    ContentDiagnostics,
    analyze_video_content,
    decide_automatic_mode,
)

STACK_VIDEO = Path(r"C:\SignBridge_Project\data\raw\sign_videos\Stack\Stack_07_1.mp4")
KARSL_VIDEO = Path(
    r"C:\SignBridge_Project\data\external\karsl\videos\03\train\0001"
    r"\01_03_0001_(06_12_16_17_20_54)_c.mp4"
)
REQUIRED_ASSETS_AVAILABLE = STACK_VIDEO.is_file()


def _make_temp_upload_copy(source: Path) -> Path:
    """Copies a real sample video to a temp-upload-style path -- a random
    uuid filename inside a signbridge_upload_-style temp directory, with no
    "karsl"/label substring anywhere in the path -- exactly mimicking what
    ``backend/app/main.py``'s real upload handler produces."""
    tmp_dir = Path.home() / "AppData" / "Local" / "Temp" / f"signbridge_upload_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    destination = tmp_dir / f"upload_{uuid.uuid4().hex}{source.suffix}"
    shutil.copyfile(source, destination)
    return destination


def _synthetic_diagnostics(**overrides) -> ContentDiagnostics:
    base = dict(
        frame_count=100,
        active_start_frame=5,
        active_end_frame=90,
        active_frame_count=86,
        active_to_total_ratio=0.86,
        segment_count=1,
        left_hand_observed_rate=0.9,
        right_hand_observed_rate=0.85,
        pose_observed_rate=1.0,
        hand_observed_rate=0.9,
        enough_motion=True,
    )
    base.update(overrides)
    return ContentDiagnostics(**base)


class ContentAnalysisUnitTests(unittest.TestCase):
    """Stage A (analyze_video_content) is a pure function of landmark
    content -- no model, no video path, no filename involved."""

    def _synthetic_raw(self, frame_count: int, hand_active_slice: slice | None) -> np.ndarray:
        raw = np.zeros((frame_count, 1629), dtype=np.float32)
        if hand_active_slice is not None:
            # POSE_END:LEFT_HAND_END is the left-hand block; set a nonzero
            # value there for the active frames so hand_presence() is True.
            from scripts.prepare_model_sequences import LEFT_HAND_END, POSE_END

            raw[hand_active_slice, POSE_END:LEFT_HAND_END] = 0.5
        return raw

    def test_no_hand_ever_observed_has_no_active_window(self) -> None:
        raw = self._synthetic_raw(60, None)
        diagnostics = analyze_video_content(raw)
        self.assertFalse(diagnostics.enough_motion)
        self.assertEqual(diagnostics.active_frame_count, 0)

    def test_single_contiguous_motion_is_one_segment(self) -> None:
        raw = self._synthetic_raw(60, slice(10, 50))
        diagnostics = analyze_video_content(raw)
        self.assertTrue(diagnostics.enough_motion)
        self.assertEqual(diagnostics.segment_count, 1)

    def test_two_separated_motion_bursts_are_two_segments(self) -> None:
        raw = self._synthetic_raw(80, None)
        from scripts.prepare_model_sequences import LEFT_HAND_END, POSE_END

        raw[5:20, POSE_END:LEFT_HAND_END] = 0.5
        raw[50:70, POSE_END:LEFT_HAND_END] = 0.5  # big gap (>GAP_MERGE_FRAMES) between bursts
        diagnostics = analyze_video_content(raw)
        self.assertTrue(diagnostics.enough_motion)
        self.assertEqual(diagnostics.segment_count, 2)

    def test_content_analysis_never_reads_a_filename(self) -> None:
        # analyze_video_content only accepts a numpy array -- there is no
        # path/filename parameter it could branch on even by mistake.
        import inspect

        signature = inspect.signature(analyze_video_content)
        self.assertEqual(list(signature.parameters), ["raw"])


class AutomaticRoutingDecisionUnitTests(unittest.TestCase):
    """Stage B/C (decide_automatic_mode) is a pure function of already-
    computed candidate evidence -- directly testable without a real video
    or model, including the one candidate path (continuous, multi-sign)
    for which no real playable sample video exists in this repository
    (see ContinuousPathTests below for why)."""

    def test_insufficient_motion_is_rejected_before_scoring_candidates(self) -> None:
        diagnostics = _synthetic_diagnostics(enough_motion=False, active_frame_count=0)
        decision = decide_automatic_mode([], [], [], [], 0.0, diagnostics)
        self.assertEqual(decision["resolved_mode"], "rejected")
        self.assertFalse(decision["accepted"])
        self.assertIn("motion", decision["routing_reason"].lower())

    def test_all_low_confidence_candidates_are_rejected(self) -> None:
        diagnostics = _synthetic_diagnostics()
        decision = decide_automatic_mode(
            it_top_k=[("Row", 0.05), ("Column", 0.04)],
            karsl_top_k=[("0012", 0.06), ("0013", 0.05)],
            decoded_gloss_sequence=[],
            decoded_gloss_confidences=[],
            ctc_non_blank_fraction_value=0.1,
            diagnostics=diagnostics,
        )
        self.assertEqual(decision["resolved_mode"], "rejected")
        self.assertFalse(decision["accepted"])

    def test_near_tied_candidates_are_rejected_as_ambiguous(self) -> None:
        # IT and KArSL are given identical evidence shape (same top-1/top-2
        # confidences), so their composite scores tie exactly, while the
        # continuous candidate has no decoded sequence at all (score 0) and
        # cannot become the winner - isolating the "top two candidates are
        # too close together" rejection path specifically.
        diagnostics = _synthetic_diagnostics(segment_count=1)
        decision = decide_automatic_mode(
            it_top_k=[("Row", 0.50), ("Column", 0.45)],
            karsl_top_k=[("0012", 0.50), ("0013", 0.45)],
            decoded_gloss_sequence=[],
            decoded_gloss_confidences=[],
            ctc_non_blank_fraction_value=0.1,
            diagnostics=diagnostics,
        )
        self.assertEqual(decision["resolved_mode"], "rejected")
        self.assertIn("ambiguous", decision["routing_reason"].lower())

    def test_strong_isolated_it_evidence_is_accepted(self) -> None:
        diagnostics = _synthetic_diagnostics(segment_count=1)
        decision = decide_automatic_mode(
            it_top_k=[("Stack", 0.9), ("Queue", 0.02)],
            karsl_top_k=[("0012", 0.05), ("0013", 0.04)],
            decoded_gloss_sequence=["Stack"],
            decoded_gloss_confidences=[0.9],
            ctc_non_blank_fraction_value=0.3,
            diagnostics=diagnostics,
        )
        self.assertEqual(decision["resolved_mode"], "it")
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["candidate_scores"]["it"]["label"], "Stack")

    def test_strong_continuous_evidence_with_multiple_segments_is_accepted(self) -> None:
        # Synthetic stand-in for a real multi-sign continuous recording
        # (no playable Isharah video asset exists in this repo - see
        # ContinuousPathTests): several motion segments, a multi-token
        # decoded sequence with reasonable confidence, and low isolated-head
        # confidence, should out-score both isolated candidates.
        diagnostics = _synthetic_diagnostics(segment_count=4, active_to_total_ratio=0.95)
        decision = decide_automatic_mode(
            it_top_k=[("Row", 0.10), ("Column", 0.08)],
            karsl_top_k=[("0012", 0.07), ("0013", 0.05)],
            decoded_gloss_sequence=["Row", "Column", "Key"],
            decoded_gloss_confidences=[0.7, 0.65, 0.68],
            ctc_non_blank_fraction_value=0.8,
            diagnostics=diagnostics,
        )
        self.assertEqual(decision["resolved_mode"], "continuous")
        self.assertTrue(decision["accepted"])
        self.assertEqual(
            decision["candidate_scores"]["continuous"]["decoded_gloss_sequence"],
            ["Row", "Column", "Key"],
        )

    def test_never_picks_the_largest_raw_confidence_naively(self) -> None:
        # KArSL has the highest RAW confidence (0.95) but a razor-thin
        # margin over its runner-up and comes from a low-quality clip;
        # IT has a lower raw confidence but a much larger margin and better
        # quality. The composite score - not the raw confidence - must
        # decide, and it must not simply be "biggest raw softmax wins".
        diagnostics = _synthetic_diagnostics(
            left_hand_observed_rate=0.95, right_hand_observed_rate=0.9, pose_observed_rate=1.0
        )
        decision = decide_automatic_mode(
            it_top_k=[("Stack", 0.6), ("Queue", 0.05)],
            karsl_top_k=[("0012", 0.95), ("0013", 0.94)],
            decoded_gloss_sequence=["Stack"],
            decoded_gloss_confidences=[0.6],
            ctc_non_blank_fraction_value=0.2,
            diagnostics=diagnostics,
        )
        # KArSL's razor-thin margin (0.01) heavily penalizes its composite
        # score versus IT's margin (0.55), so IT should win despite the
        # lower raw top-1 number - proving the router is not naive-argmax.
        self.assertEqual(decision["resolved_mode"], "it")


@unittest.skipUnless(REQUIRED_ASSETS_AVAILABLE, f"Sample video not found: {STACK_VIDEO}")
class RealCheckpointAutoRoutingTests(unittest.TestCase):
    """End-to-end tests against the real trained checkpoint and real
    MediaPipe extraction -- the same path a live /api/recognize request
    takes. Each test copies a real sample video to a temp-upload-style
    path first, so these are also the direct regression tests for the
    reported bug (a temp filename must not force "continuous")."""

    def setUp(self) -> None:
        from app.recognition.service import recognition_service

        self.recognition_service = recognition_service
        self._temp_paths: list[Path] = []

    def tearDown(self) -> None:
        for path in self._temp_paths:
            shutil.rmtree(path.parent, ignore_errors=True)

    def _temp_copy(self, source: Path) -> Path:
        destination = _make_temp_upload_copy(source)
        self._temp_paths.append(destination)
        return destination

    def test_temp_camera_style_filename_does_not_force_continuous(self) -> None:
        temp_video = self._temp_copy(STACK_VIDEO)
        self.assertNotIn("stack", str(temp_video).casefold())

        result = self.recognition_service.recognize(temp_video, mode="auto", top_k=5)

        self.assertNotEqual(
            result["resolved_mode"],
            "continuous",
            "A temp-upload-style filename with no dataset hints must not force "
            "'continuous' mode - this was the exact reported bug.",
        )

    def test_temp_upload_filename_does_not_determine_dataset(self) -> None:
        temp_video = self._temp_copy(STACK_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="auto", top_k=5)
        # The result must be driven by content, not by anything derivable
        # from the (randomized) filename.
        self.assertIn(result["resolved_mode"], {"it", "karsl", "continuous"})
        self.assertIsNotNone(result.get("candidate_scores"))

    def test_known_stack_video_selects_it_path_and_returns_stack(self) -> None:
        temp_video = self._temp_copy(STACK_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="auto", top_k=5)

        self.assertEqual(result["resolved_mode"], "it")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["predicted_label"], "Stack")
        self.assertIn("it", result["candidate_scores"])
        self.assertIn("karsl", result["candidate_scores"])
        self.assertIn("continuous", result["candidate_scores"])

    @unittest.skipUnless(KARSL_VIDEO.is_file(), f"KArSL sample video not found: {KARSL_VIDEO}")
    def test_isolated_karsl_example_selects_karsl_path(self) -> None:
        temp_video = self._temp_copy(KARSL_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="auto", top_k=5)

        # Routing correctness only - not label correctness (no accuracy
        # claim is made about which specific KArSL sign this checkpoint
        # predicts).
        self.assertEqual(result["resolved_mode"], "karsl")
        self.assertTrue(result["accepted"])

    def test_explicit_it_mode_still_selects_it_path(self) -> None:
        temp_video = self._temp_copy(STACK_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="it", top_k=5)
        self.assertEqual(result["resolved_mode"], "it")
        self.assertEqual(result["requested_mode"], "it")

    @unittest.skipUnless(KARSL_VIDEO.is_file(), f"KArSL sample video not found: {KARSL_VIDEO}")
    def test_explicit_karsl_mode_still_selects_karsl_path(self) -> None:
        temp_video = self._temp_copy(KARSL_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="karsl", top_k=5)
        self.assertEqual(result["resolved_mode"], "karsl")
        self.assertEqual(result["requested_mode"], "karsl")

    def test_explicit_continuous_mode_still_selects_continuous_path(self) -> None:
        # Explicit modes must still short-circuit exactly as before, even
        # on a video that auto mode would route elsewhere.
        temp_video = self._temp_copy(STACK_VIDEO)
        result = self.recognition_service.recognize(temp_video, mode="continuous", top_k=5)
        self.assertEqual(result["resolved_mode"], "continuous")
        self.assertEqual(result["requested_mode"], "continuous")


class ContinuousPathTests(unittest.TestCase):
    """A real playable continuous/Isharah example video does NOT exist
    anywhere in this repository or on this machine (confirmed by a
    dedicated asset search: the Isharah dataset here only ships extracted
    frame JPEGs, a 2.37GB frame-image zip, and a 7.74GB precomputed pose
    pickle - no `.mp4`/`.mov`/`.avi`/`.webm` file). This is reported
    explicitly, per the requirement not to mark an unavailable-asset test
    as passed, rather than fabricated or silently skipped without
    explanation. AutomaticRoutingDecisionUnitTests.
    test_strong_continuous_evidence_with_multiple_segments_is_accepted
    above covers the *decision logic* for a continuous-shaped candidate
    with synthetic (clearly labeled as synthetic) evidence instead.
    """

    def test_no_real_continuous_video_asset_is_available(self) -> None:
        self.skipTest(
            "No real playable continuous/Isharah video file exists in this "
            "repository (data/external/isharah only has frame JPEGs, a "
            "frame-image zip, and a pose pickle - no video). End-to-end "
            "continuous-path routing was not run against real checkpoint "
            "inference; see the synthetic decision-logic test instead."
        )


class RejectedRecognitionSafetyTests(unittest.TestCase):
    """A rejected recognition must never reach avatar motion playback or
    produce a misleading course answer - it must look, to every downstream
    consumer, exactly like 'no usable recognition happened'."""

    def test_rejected_result_resolves_to_no_motion(self) -> None:
        from app.services.pipeline import _resolve_path_a_motions

        rejected_raw_result = {
            "resolved_mode": "rejected",
            "motion_dataset_namespace": None,
            "decoded_gloss_sequence": [],
            "predicted_token": None,
        }
        self.assertEqual(_resolve_path_a_motions(rejected_raw_result), [])

    def test_rejected_result_produces_no_gloss_text(self) -> None:
        from app.services.pipeline import _recognition_to_gloss_text

        rejected_raw_result = {
            "resolved_mode": "rejected",
            "predicted_label": None,
            "decoded_gloss_sequence": [],
        }
        self.assertIsNone(_recognition_to_gloss_text(rejected_raw_result))

    @unittest.skipUnless(REQUIRED_ASSETS_AVAILABLE, f"Sample video not found: {STACK_VIDEO}")
    def test_rejected_recognition_never_reaches_motion_playback_end_to_end(self) -> None:
        from unittest.mock import patch

        from app.services.pipeline import recognize_and_answer

        fake_rejected_result = {
            "requested_mode": "auto",
            "resolved_mode": "rejected",
            "accepted": False,
            "routing_reason": "synthetic rejection for this test",
            "candidate_scores": {},
            "quality": {},
            "predicted_label": None,
            "predicted_token": None,
            "decoded_gloss_sequence": [],
            "decoded_gloss_confidences": [],
            "top_k": [],
            "temporal_token_evidence": [],
            "confidence": None,
            "frame_count": 10,
            "active_start_frame": 0,
            "active_end_frame": 9,
            "active_source": "content-based motion detection",
            "left_hand_observed_rate": 0.0,
            "right_hand_observed_rate": 0.0,
            "dataset_source": "rejected",
            "motion_dataset_namespace": None,
            "device": "cpu",
            "warnings": ["synthetic rejection"],
        }

        with patch(
            "app.services.pipeline.recognition_service.recognize",
            return_value=fake_rejected_result,
        ):
            response = recognize_and_answer(STACK_VIDEO, mode="auto")

        self.assertEqual(response.recognized_motion_sequence, [])
        self.assertEqual(response.recognized_gloss, "")
        self.assertFalse(response.recognition.accepted)
        self.assertEqual(response.recognition.resolved_mode, "rejected")


class CameraAndUploadShareTheSameContractTests(unittest.TestCase):
    """Camera recordings (always .webm, per camera-recorder.js) and file
    uploads (any of .mp4/.mov/.avi/.mkv/.webm) must resolve through the
    identical automatic-routing contract - confirmed structurally, since
    both reach the exact same code path in backend/app/main.py before any
    real webm sample asset would be needed."""

    def test_recognize_endpoint_uses_one_shared_code_path_for_every_extension(self) -> None:
        import inspect

        from app.main import recognize

        source = inspect.getsource(recognize)
        # Exactly one call to recognize_and_answer, parameterized only by
        # the extension-validated, already-decoded path and the caller's
        # requested mode - no extension-specific branch exists that could
        # route .webm (camera) differently from .mp4/.mov/.avi/.mkv (upload).
        self.assertEqual(source.count("recognize_and_answer("), 1)
        self.assertIn("mode=mode", source)


if __name__ == "__main__":
    unittest.main()
