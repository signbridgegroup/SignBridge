"""Reusable recognition service built on the traced inference import chain.

This module refactors the logic that previously only existed as
``predict_final_signbridge_video.py``'s ``main()`` into a class that loads
the checkpoint once and answers many requests. It deliberately imports the
exact functions/classes that file itself imports (model class, feature
prep, decoding helpers) so behaviour and checkpoint compatibility are
identical to the standalone script. Nothing here retrains, fine-tunes, or
recomputes the checkpoint's saved vocabulary/label mappings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config import settings
from app.errors import RecognitionError
from app.motions.manifest import RECOGNITION_MODE_TO_DATASET

# These imports only work because app.config already inserted
# backend/legacy_runtime onto sys.path.
from auto_recognition_router import (  # noqa: E402
    IT_TARGET_FRAMES,
    ContentDiagnostics,
    analyze_video_content,
    ctc_non_blank_fraction,
    decide_automatic_mode,
)
from predict_final_signbridge_video import (  # noqa: E402
    classification_top_k,
    detect_mode,
    display_karsl_label,
    greedy_decode_with_confidence,
    load_karsl_labels,
    resolve_bounds,
    temporal_token_evidence,
)
from predict_it_video import HOLISTIC_MODEL, active_bounds, extract_video  # noqa: E402
from prepare_unified_sign_data import jordanian_572_to_shared_198  # noqa: E402
from scripts.prepare_model_sequences import prepare_sequence  # noqa: E402
from train_unified_karsl_multitask import UnifiedMultiTaskModel  # noqa: E402
from train_unified_sign_ctc import make_model_features  # noqa: E402

_MODE_LABELS = {
    "auto": "تلقائي",
    "it": "الإشارات الأردنية التقنية",
    "karsl": "KArSL",
    "continuous": "Isharah / إشارات متصلة",
}


class RecognitionService:
    """Loads the final checkpoint once and reuses it for every request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: UnifiedMultiTaskModel | None = None
        self._vocabulary: list[str] = []
        self._karsl_class_to_index: dict[str, int] = {}
        self._it_label_to_class: dict[str, int] = {}
        self._index_to_karsl: dict[int, str] = {}
        self._index_to_it: dict[int, str] = {}
        self._stride: int = 2
        self._device: torch.device = torch.device("cpu")
        self._karsl_labels: dict[str, str] = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def checkpoint_available(self) -> bool:
        return settings.checkpoint.is_file()

    def mediapipe_model_available(self) -> bool:
        return HOLISTIC_MODEL.is_file() and settings.mediapipe_model.is_file()

    def karsl_labels_available(self) -> bool:
        return settings.karsl_labels.is_file()

    def ensure_loaded(self, force_cpu: bool = False) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not settings.checkpoint.is_file():
                raise RecognitionError(
                    f"Checkpoint not found at {settings.checkpoint}."
                )
            checkpoint = torch.load(
                settings.checkpoint, map_location="cpu", weights_only=False
            )
            self._vocabulary = list(checkpoint["vocabulary"])
            self._karsl_class_to_index = {
                str(key).zfill(4): int(value)
                for key, value in checkpoint["karsl_class_to_index"].items()
            }
            self._it_label_to_class = {
                str(key): int(value)
                for key, value in checkpoint["it_label_to_class"].items()
            }
            self._index_to_karsl = {
                value: key for key, value in self._karsl_class_to_index.items()
            }
            self._index_to_it = {
                value: key for key, value in self._it_label_to_class.items()
            }
            self._stride = int(checkpoint.get("stride", 2))

            self._device = torch.device(
                "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
            )
            model = UnifiedMultiTaskModel(
                vocabulary_size=len(self._vocabulary),
                karsl_classes=len(self._karsl_class_to_index),
                it_classes=len(self._it_label_to_class),
            ).to(self._device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self._model = model
            self._checkpoint_input_features = int(
                checkpoint.get("input_features", 0)
            )
            self._karsl_labels = load_karsl_labels(settings.karsl_labels)

    def recognize(
        self,
        video_path: Path,
        mode: str = "auto",
        top_k: int = 5,
        start_frame: int | None = None,
        end_frame: int | None = None,
        cpu: bool = False,
    ) -> dict[str, Any]:
        self.ensure_loaded(force_cpu=cpu)
        assert self._model is not None

        warnings: list[str] = []
        video = video_path.resolve()
        if not video.is_file():
            raise RecognitionError(f"Uploaded video not found: {video}")
        if not HOLISTIC_MODEL.is_file():
            raise RecognitionError(f"MediaPipe model not found: {HOLISTIC_MODEL}")

        raw = extract_video(video)
        if len(raw) == 0:
            raise RecognitionError("No frames were extracted from the video.")

        # Genuine automatic routing: decide isolated-vs-continuous and which
        # trained head to trust from the video's own content (Stage A/B/C),
        # never from its filename or temp-upload path. A manual start/end
        # frame override is a diagnostic escape hatch, not part of the
        # normal user flow, so it still uses the simpler explicit-bounds
        # path below.
        if mode == "auto" and start_frame is None and end_frame is None:
            return self._recognize_auto(video, raw, top_k)

        _, _, found_in_it_report = active_bounds(video, len(raw))
        resolved_mode = detect_mode(video, found_in_it_report, mode)
        start, end, bounds_source = resolve_bounds(
            video, len(raw), resolved_mode, start_frame, end_frame
        )

        active_frames = end - start + 1
        target_frames = 200 if resolved_mode == "it" else active_frames
        prepared_572, stats = prepare_sequence(raw, start, end, target_frames, 5)
        shared_198 = jordanian_572_to_shared_198(prepared_572)

        features = make_model_features(shared_198, self._stride)
        if (
            self._checkpoint_input_features
            and features.shape[1] != self._checkpoint_input_features
        ):
            raise RecognitionError(
                "Feature mismatch: checkpoint expects "
                f"{self._checkpoint_input_features}, got {features.shape[1]}."
            )

        tensor = features.unsqueeze(0).to(self._device)
        input_lengths = torch.tensor([len(features)], dtype=torch.long)

        with self._lock, torch.inference_mode():
            logits, output_lengths, encoded = self._model(tensor, input_lengths)

        output_length = int(output_lengths[0])
        token_ids, token_confidences = greedy_decode_with_confidence(
            logits, output_length
        )
        predicted_tokens = [self._vocabulary[index] for index in token_ids]

        result: dict[str, Any] = {
            "requested_mode": mode,
            "resolved_mode": resolved_mode,
            "resolved_mode_label": _MODE_LABELS.get(resolved_mode, resolved_mode),
            "predicted_label": None,
            # The exact, unformatted token for direct motion-manifest
            "predicted_token": None,
            "decoded_gloss_sequence": predicted_tokens,
            "decoded_gloss_confidences": [float(c) for c in token_confidences],
            "top_k": [],
            "temporal_token_evidence": [],
            "confidence": None,
            "frame_count": int(len(raw)),
            "active_start_frame": int(start),
            "active_end_frame": int(end),
            "active_source": bounds_source,
            "left_hand_observed_rate": float(stats["resampled_left_observed_rate"]),
            "right_hand_observed_rate": float(stats["resampled_right_observed_rate"]),
            # This is the recognizer's own mode name ("it"/"karsl"/
            # "continuous"), NOT the motion-manifest dataset namespace
            # ("jordanian_it"/"karsl"/"isharah") -- see
            # RECOGNITION_MODE_TO_DATASET for that bridge, applied by the
            # pipeline layer, never guessed downstream.
            "dataset_source": resolved_mode,
            "motion_dataset_namespace": RECOGNITION_MODE_TO_DATASET.get(resolved_mode),
            "device": str(self._device),
            "warnings": warnings,
        }

        if resolved_mode == "it":
            with self._lock, torch.inference_mode():
                class_logits, _ = self._model.isolated_logits(
                    encoded, output_lengths, "it"
                )
            top = classification_top_k(class_logits, self._index_to_it, top_k)
            result["top_k"] = [{"label": label, "confidence": float(score)} for label, score in top]
            if top:
                result["predicted_label"] = top[0][0]
                result["predicted_token"] = top[0][0]  # raw IT label, unformatted
                result["confidence"] = float(top[0][1])
        elif resolved_mode == "karsl":
            with self._lock, torch.inference_mode():
                class_logits, _ = self._model.isolated_logits(
                    encoded, output_lengths, "karsl"
                )
            top = classification_top_k(class_logits, self._index_to_karsl, top_k)
            result["top_k"] = [
                {
                    "label": display_karsl_label(sign_id, self._karsl_labels),
                    "confidence": float(score),
                }
                for sign_id, score in top
            ]
            if top:
                result["predicted_label"] = display_karsl_label(top[0][0], self._karsl_labels)
                result["predicted_token"] = top[0][0]  # raw sign_id, e.g. "0001"
                result["confidence"] = float(top[0][1])
        else:
            if predicted_tokens:
                result["predicted_label"] = " ".join(predicted_tokens)
                # No single "predicted_token" for continuous mode: it's a
                # sequence. Path A resolves decoded_gloss_sequence directly,
                # in order, one manifest lookup per gloss token.
                result["confidence"] = (
                    float(np.mean(token_confidences)) if token_confidences else None
                )
            else:
                warnings.append("The continuous decoder returned an empty gloss sequence.")

        if resolved_mode in {"it", "karsl"}:
            evidence = temporal_token_evidence(
                logits, output_length, self._vocabulary, top_k
            )
            result["temporal_token_evidence"] = [
                {
                    "label": (
                        display_karsl_label(token, self._karsl_labels)
                        if token.startswith("KARSL_")
                        else token
                    ),
                    "confidence": float(score),
                }
                for token, score in evidence
            ]

        return result

    def _prepare_forward_pass(
        self, raw: np.ndarray, start: int, end: int, target_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one full encoder forward pass for a given active window/target
        frame count. Returns (ctc_logits, output_lengths, encoded) exactly
        like ``self._model(tensor, lengths)`` -- factored out so the auto
        router can evaluate the IT-style (200-frame) and native-length
        (KArSL/continuous) contracts independently, as their preprocessing
        genuinely differs (see module docstring in auto_recognition_router.py).
        """
        prepared_572, _stats = prepare_sequence(raw, start, end, target_frames, 5)
        shared_198 = jordanian_572_to_shared_198(prepared_572)
        features = make_model_features(shared_198, self._stride)
        tensor = features.unsqueeze(0).to(self._device)
        lengths = torch.tensor([len(features)], dtype=torch.long)
        with self._lock, torch.inference_mode():
            assert self._model is not None
            return self._model(tensor, lengths)

    def _build_rejected_result(
        self,
        requested_mode: str,
        diagnostics: ContentDiagnostics,
        decision: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        warnings = list(warnings) + [
            "Recording rejected: please record the sign again with your upper "
            "body and both hands visible, a stable camera, adequate lighting, "
            "and the complete sign inside the recording."
        ]
        return {
            "requested_mode": requested_mode,
            "resolved_mode": "rejected",
            "resolved_mode_label": "تعذر التعرف بثقة كافية",
            "predicted_label": None,
            "predicted_token": None,
            "decoded_gloss_sequence": [],
            "decoded_gloss_confidences": [],
            "top_k": [],
            "temporal_token_evidence": [],
            "confidence": None,
            "frame_count": int(diagnostics.frame_count),
            "active_start_frame": int(diagnostics.active_start_frame),
            "active_end_frame": int(diagnostics.active_end_frame),
            "active_source": "content-based motion detection",
            "left_hand_observed_rate": float(diagnostics.left_hand_observed_rate * 100),
            "right_hand_observed_rate": float(diagnostics.right_hand_observed_rate * 100),
            "dataset_source": "rejected",
            "motion_dataset_namespace": None,
            "device": str(self._device),
            "warnings": warnings,
            "accepted": False,
            "routing_reason": decision["routing_reason"],
            "candidate_scores": decision["candidate_scores"],
            "quality": decision["quality"],
        }

    def _recognize_auto(self, video: Path, raw: np.ndarray, top_k: int) -> dict[str, Any]:
        """Genuinely automatic routing (Stage A/B/C): decide isolated vs.
        continuous, and which trained head to trust, from the video's own
        content -- never from ``video``'s filename or temp-upload path.
        """
        warnings: list[str] = []
        diagnostics = analyze_video_content(raw)

        if not diagnostics.enough_motion:
            decision = decide_automatic_mode([], [], [], [], 0.0, diagnostics)
            return self._build_rejected_result("auto", diagnostics, decision, warnings)

        # IT-style evaluation: resampled to exactly 200 frames, matching the
        # contract Jordanian IT was trained with.
        ctc_logits_it, output_lengths_it, encoded_it = self._prepare_forward_pass(
            raw, diagnostics.active_start_frame, diagnostics.active_end_frame, IT_TARGET_FRAMES
        )
        with self._lock, torch.inference_mode():
            it_class_logits, _ = self._model.isolated_logits(encoded_it, output_lengths_it, "it")
        it_top_k = classification_top_k(it_class_logits, self._index_to_it, top_k)

        # Native-length evaluation: shared by the KArSL head and the
        # continuous CTC decoder, exactly like the existing explicit-mode
        # contract (target_frames = active_frames for "karsl"/"continuous").
        native_frames = diagnostics.active_frame_count
        ctc_logits_native, output_lengths_native, encoded_native = self._prepare_forward_pass(
            raw, diagnostics.active_start_frame, diagnostics.active_end_frame, native_frames
        )
        with self._lock, torch.inference_mode():
            karsl_class_logits, _ = self._model.isolated_logits(encoded_native, output_lengths_native, "karsl")
        karsl_top_k_raw = classification_top_k(karsl_class_logits, self._index_to_karsl, top_k)
        karsl_top_k_display = [
            (display_karsl_label(sign_id, self._karsl_labels), score) for sign_id, score in karsl_top_k_raw
        ]

        native_output_length = int(output_lengths_native[0])
        token_ids, token_confidences = greedy_decode_with_confidence(ctc_logits_native, native_output_length)
        decoded_gloss_sequence = [self._vocabulary[index] for index in token_ids]
        non_blank_fraction = ctc_non_blank_fraction(ctc_logits_native, native_output_length)

        decision = decide_automatic_mode(
            it_top_k,
            karsl_top_k_raw,
            decoded_gloss_sequence,
            token_confidences,
            non_blank_fraction,
            diagnostics,
        )

        resolved_mode = decision["resolved_mode"]
        if resolved_mode == "rejected":
            return self._build_rejected_result("auto", diagnostics, decision, warnings)

        result: dict[str, Any] = {
            "requested_mode": "auto",
            "resolved_mode": resolved_mode,
            "resolved_mode_label": _MODE_LABELS.get(resolved_mode, resolved_mode),
            "predicted_label": None,
            "predicted_token": None,
            "decoded_gloss_sequence": decoded_gloss_sequence,
            "decoded_gloss_confidences": [float(c) for c in token_confidences],
            "top_k": [],
            "temporal_token_evidence": [],
            "confidence": None,
            "frame_count": int(diagnostics.frame_count),
            "active_start_frame": int(diagnostics.active_start_frame),
            "active_end_frame": int(diagnostics.active_end_frame),
            "active_source": "content-based motion detection",
            "left_hand_observed_rate": float(diagnostics.left_hand_observed_rate * 100),
            "right_hand_observed_rate": float(diagnostics.right_hand_observed_rate * 100),
            "dataset_source": resolved_mode,
            "motion_dataset_namespace": RECOGNITION_MODE_TO_DATASET.get(resolved_mode),
            "device": str(self._device),
            "warnings": warnings,
            "accepted": True,
            "routing_reason": decision["routing_reason"],
            "candidate_scores": decision["candidate_scores"],
            "quality": decision["quality"],
        }

        if resolved_mode == "it":
            result["top_k"] = [{"label": label, "confidence": float(score)} for label, score in it_top_k]
            if it_top_k:
                result["predicted_label"] = it_top_k[0][0]
                result["predicted_token"] = it_top_k[0][0]
                result["confidence"] = float(it_top_k[0][1])
            evidence = temporal_token_evidence(
                ctc_logits_it, int(output_lengths_it[0]), self._vocabulary, top_k
            )
            result["temporal_token_evidence"] = [
                {
                    "label": display_karsl_label(token, self._karsl_labels) if token.startswith("KARSL_") else token,
                    "confidence": float(score),
                }
                for token, score in evidence
            ]
        elif resolved_mode == "karsl":
            result["top_k"] = [{"label": label, "confidence": float(score)} for label, score in karsl_top_k_display]
            if karsl_top_k_raw:
                result["predicted_label"] = karsl_top_k_display[0][0]
                result["predicted_token"] = karsl_top_k_raw[0][0]
                result["confidence"] = float(karsl_top_k_raw[0][1])
            evidence = temporal_token_evidence(
                ctc_logits_native, native_output_length, self._vocabulary, top_k
            )
            result["temporal_token_evidence"] = [
                {
                    "label": display_karsl_label(token, self._karsl_labels) if token.startswith("KARSL_") else token,
                    "confidence": float(score),
                }
                for token, score in evidence
            ]
        else:  # continuous
            if decoded_gloss_sequence:
                result["predicted_label"] = " ".join(decoded_gloss_sequence)
                result["confidence"] = float(np.mean(token_confidences)) if token_confidences else None
            else:
                warnings.append("The continuous decoder returned an empty gloss sequence.")

        return result


recognition_service = RecognitionService()
