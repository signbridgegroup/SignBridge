"""Content-based automatic sign-recognition routing.

Replaces filename/path-based "auto" mode detection (the previous
``detect_mode`` in ``predict_final_signbridge_video.py`` decided isolated
vs. continuous by checking whether "karsl" appeared in the video's path, or
whether the file was found in a saved training-time active-length report --
both of which are always false for a real camera recording or uploaded
file, since those are saved under a temporary, randomly-named path). This
module decides using the actual extracted landmarks and the already-trained
model's own outputs instead.

Nothing here retrains, fine-tunes, or recomputes the checkpoint's saved
vocabulary/label mappings -- it only reads MediaPipe landmarks (Stage A) and
the existing model's classification/CTC outputs (Stage B/C) to build an
explicit, documented heuristic composite routing score. This score is a
heuristic, NOT a calibrated probability -- no calibration data (e.g. saved
validation temperature-scaling or per-head accuracy statistics) exists
anywhere in this project's checkpoints, so none is claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from scripts.prepare_model_sequences import FACE_END, LEFT_HAND_END, POSE_END, RIGHT_HAND_END, hand_presence

# --------------------------------------------------------------------------- #
# Tunable thresholds -- documented here, in one place, rather than scattered
# magic numbers. These are conservative heuristic choices, not fitted/
# calibrated constants (no labeled auto-routing dataset exists to fit them
# against).
# --------------------------------------------------------------------------- #
IT_TARGET_FRAMES = 200
MIN_ACTIVE_FRAMES = 5
MIN_HAND_OBSERVED_RATE = 0.25
MIN_ACCEPT_SCORE = 0.35
MIN_WINNER_MARGIN = 0.05
GAP_MERGE_FRAMES = 6
BOUNDARY_PADDING_FRAMES = 3


@dataclass
class ContentDiagnostics:
    """Stage A output: what the video's own content shows, independent of
    its filename, folder, or upload path."""

    frame_count: int
    active_start_frame: int
    active_end_frame: int
    active_frame_count: int
    active_to_total_ratio: float
    segment_count: int
    left_hand_observed_rate: float
    right_hand_observed_rate: float
    pose_observed_rate: float
    hand_observed_rate: float
    enough_motion: bool


def _presence(points: np.ndarray) -> np.ndarray:
    return np.any(np.abs(points) > 1e-8, axis=(1, 2))


def analyze_video_content(raw: np.ndarray) -> ContentDiagnostics:
    """Stage A -- shared video analysis.

    Computes quality/temporal diagnostics directly from the MediaPipe
    landmarks already extracted for this request (the same ``raw`` array
    every recognition mode uses) -- total frames, the active signing
    window, how many separate motion segments it contains, and hand/pose
    observation rates. No part of this reads the video's path or filename.
    """
    frame_count = raw.shape[0]
    pose = raw[:, FACE_END:POSE_END].reshape(frame_count, 33, 3)
    left_hand = raw[:, POSE_END:LEFT_HAND_END].reshape(frame_count, 21, 3)
    right_hand = raw[:, LEFT_HAND_END:RIGHT_HAND_END].reshape(frame_count, 21, 3)

    left_observed = hand_presence(left_hand)
    right_observed = hand_presence(right_hand)
    pose_observed = _presence(pose)
    hand_observed = left_observed | right_observed

    active_indices = np.flatnonzero(hand_observed)
    if active_indices.size == 0:
        return ContentDiagnostics(
            frame_count=frame_count,
            active_start_frame=0,
            active_end_frame=frame_count - 1,
            active_frame_count=0,
            active_to_total_ratio=0.0,
            segment_count=0,
            left_hand_observed_rate=0.0,
            right_hand_observed_rate=0.0,
            pose_observed_rate=round(float(pose_observed.mean()), 4),
            hand_observed_rate=0.0,
            enough_motion=False,
        )

    start = max(0, int(active_indices[0]) - BOUNDARY_PADDING_FRAMES)
    end = min(frame_count - 1, int(active_indices[-1]) + BOUNDARY_PADDING_FRAMES)
    window = slice(start, end + 1)

    # Count contiguous "a hand is visible" runs inside the active window,
    # merging gaps of GAP_MERGE_FRAMES or fewer (a brief MediaPipe tracking
    # dropout mid-sign, not a real pause between two separate signs). More
    # than one merged run is evidence of continuous/multi-sign content.
    segment_count = 0
    in_run = False
    gap = 0
    for observed in hand_observed[window]:
        if observed:
            if not in_run:
                segment_count += 1
                in_run = True
            gap = 0
        else:
            gap += 1
            if gap > GAP_MERGE_FRAMES:
                in_run = False

    active_frame_count = end - start + 1
    enough_motion = (
        active_frame_count >= MIN_ACTIVE_FRAMES
        and float(hand_observed[window].mean()) >= MIN_HAND_OBSERVED_RATE
    )

    return ContentDiagnostics(
        frame_count=frame_count,
        active_start_frame=start,
        active_end_frame=end,
        active_frame_count=active_frame_count,
        active_to_total_ratio=round(active_frame_count / frame_count, 4),
        segment_count=segment_count,
        left_hand_observed_rate=round(float(left_observed[window].mean()), 4),
        right_hand_observed_rate=round(float(right_observed[window].mean()), 4),
        pose_observed_rate=round(float(pose_observed[window].mean()), 4),
        hand_observed_rate=round(float(hand_observed[window].mean()), 4),
        enough_motion=enough_motion,
    )


def ctc_non_blank_fraction(logits: torch.Tensor, output_length: int) -> float:
    """Fraction of decoded frames whose most likely CTC token is not the
    blank token (id 0). Continuous multi-sign input tends to hold non-blank
    tokens for a larger share of the sequence than a single isolated sign
    padded/surrounded by blank-dominated frames.
    """
    if output_length <= 0:
        return 0.0
    frame_tokens = logits[0, :output_length].argmax(dim=-1)
    return float((frame_tokens != 0).float().mean().item())


def _top2_margin(top_k: list[tuple[str, float]]) -> float:
    if len(top_k) < 2:
        return float(top_k[0][1]) if top_k else 0.0
    return float(top_k[0][1] - top_k[1][1])


def _normalized_entropy(confidences: list[float]) -> float:
    """0 = fully confident (all mass on one candidate), 1 = maximally
    spread out across the observed top-k candidates. This is only computed
    over the *returned* top-k, not the full label vocabulary, so it is a
    conservative proxy, not a true distribution entropy."""
    if len(confidences) < 2:
        return 0.0
    probabilities = np.clip(np.array(confidences, dtype=np.float64), 1e-9, 1.0)
    probabilities = probabilities / probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = math.log(len(probabilities))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _quality_score(diagnostics: ContentDiagnostics) -> float:
    return max(0.0, min(1.0, 0.7 * diagnostics.hand_observed_rate + 0.3 * diagnostics.pose_observed_rate))


def score_isolated(top_k: list[tuple[str, float]], diagnostics: ContentDiagnostics) -> tuple[float, dict[str, Any]]:
    """Composite routing score for an isolated-sign candidate (Jordanian IT
    or KArSL) -- explicitly documented, uncalibrated heuristic combining
    top-1 confidence, the margin over the runner-up, how spread out the
    top-k is, video quality, and a mild bonus for a single motion segment
    (isolated signs are usually one continuous motion)."""
    if not top_k:
        return 0.0, {"label": None, "raw_confidence": 0.0, "routing_score": 0.0, "margin": 0.0}

    top1_confidence = float(top_k[0][1])
    margin = _top2_margin(top_k)
    entropy = _normalized_entropy([score for _label, score in top_k])
    quality = _quality_score(diagnostics)
    single_segment_bonus = 1.0 if diagnostics.segment_count <= 1 else 0.0

    routing_score = (
        0.45 * top1_confidence
        + 0.25 * margin
        + 0.15 * (1.0 - entropy)
        + 0.10 * quality
        + 0.05 * single_segment_bonus
    )
    return routing_score, {
        "label": top_k[0][0],
        "raw_confidence": round(top1_confidence, 4),
        "routing_score": round(routing_score, 4),
        "margin": round(margin, 4),
    }


def score_continuous(
    decoded_gloss_sequence: list[str],
    decoded_gloss_confidences: list[float],
    non_blank_fraction: float,
    diagnostics: ContentDiagnostics,
) -> tuple[float, dict[str, Any]]:
    """Composite routing score for the continuous CTC candidate --
    explicitly documented, uncalibrated heuristic combining average decoded
    token confidence, how many separate motion segments were observed
    (more segments -> more sign-like of a sequence), how much of the
    active window the CTC decoder considered non-blank, and decoded
    sequence length (a single decoded token is weak continuous evidence)."""
    if not decoded_gloss_sequence:
        return 0.0, {"decoded_gloss_sequence": [], "routing_score": 0.0}

    average_confidence = float(np.mean(decoded_gloss_confidences)) if decoded_gloss_confidences else 0.0
    segment_evidence = min(1.0, diagnostics.segment_count / 3.0)
    length_evidence = min(1.0, len(decoded_gloss_sequence) / 3.0)
    quality = _quality_score(diagnostics)

    routing_score = (
        0.35 * average_confidence
        + 0.25 * segment_evidence
        + 0.20 * max(0.0, min(1.0, non_blank_fraction))
        + 0.15 * length_evidence
        + 0.05 * quality
    )
    return routing_score, {
        "decoded_gloss_sequence": list(decoded_gloss_sequence),
        "routing_score": round(routing_score, 4),
        "average_token_confidence": round(average_confidence, 4),
    }


def decide_automatic_mode(
    it_top_k: list[tuple[str, float]],
    karsl_top_k: list[tuple[str, float]],
    decoded_gloss_sequence: list[str],
    decoded_gloss_confidences: list[float],
    ctc_non_blank_fraction_value: float,
    diagnostics: ContentDiagnostics,
) -> dict[str, Any]:
    """Stage B/C -- decide which trained head's result to trust, or reject.

    Pure function of already-computed evidence (no video/model access), so
    it is directly unit-testable with synthetic evidence. Never picks the
    largest raw softmax value across heads naively: each candidate gets its
    own composite score (see score_isolated/score_continuous above) built
    from information that is meaningful *within* that head, then the
    normalized composite scores -- not the raw confidences -- are compared
    against each other and against explicit acceptance/margin thresholds.
    """
    # Percentage scale (0-100), matching the top-level
    # left_hand_observed_rate/right_hand_observed_rate convention already
    # used elsewhere in the recognition result contract.
    quality = {
        "left_hand_observed_rate": round(diagnostics.left_hand_observed_rate * 100, 2),
        "right_hand_observed_rate": round(diagnostics.right_hand_observed_rate * 100, 2),
        "pose_observed_rate": round(diagnostics.pose_observed_rate * 100, 2),
        "active_frame_count": diagnostics.active_frame_count,
        "active_to_total_ratio": diagnostics.active_to_total_ratio,
        "segment_count": diagnostics.segment_count,
    }

    if not diagnostics.enough_motion:
        return {
            "resolved_mode": "rejected",
            "accepted": False,
            "routing_reason": (
                "Insufficient signing motion detected: no hand was reliably "
                "visible for a long enough stretch of the recording."
            ),
            "candidate_scores": {},
            "quality": quality,
        }

    it_score, it_info = score_isolated(it_top_k, diagnostics)
    karsl_score, karsl_info = score_isolated(karsl_top_k, diagnostics)
    continuous_score, continuous_info = score_continuous(
        decoded_gloss_sequence, decoded_gloss_confidences, ctc_non_blank_fraction_value, diagnostics
    )

    candidate_scores = {"it": it_info, "karsl": karsl_info, "continuous": continuous_info}
    ranked = sorted(
        (("it", it_score), ("karsl", karsl_score), ("continuous", continuous_score)),
        key=lambda item: item[1],
        reverse=True,
    )
    winner_mode, winner_score = ranked[0]
    runner_up_score = ranked[1][1]

    if winner_score < MIN_ACCEPT_SCORE:
        return {
            "resolved_mode": "rejected",
            "accepted": False,
            "routing_reason": (
                f"All candidate recognition paths scored below the acceptance "
                f"threshold ({MIN_ACCEPT_SCORE}); the strongest was {winner_mode!r} "
                f"at {winner_score:.2f}."
            ),
            "candidate_scores": candidate_scores,
            "quality": quality,
        }

    if winner_score - runner_up_score < MIN_WINNER_MARGIN:
        return {
            "resolved_mode": "rejected",
            "accepted": False,
            "routing_reason": (
                f"Isolated-vs-continuous evidence was too ambiguous: top candidate "
                f"{winner_mode!r} ({winner_score:.2f}) was not clearly separated "
                f"from the runner-up ({runner_up_score:.2f})."
            ),
            "candidate_scores": candidate_scores,
            "quality": quality,
        }

    reasons = {
        "it": "isolated sign; the Jordanian IT head had the strongest accepted normalized evidence",
        "karsl": "isolated sign; the KArSL head had the strongest accepted normalized evidence",
        "continuous": (
            "continuous signing evidence (motion segmentation and/or the decoded "
            "CTC sequence) outweighed the isolated-sign candidates"
        ),
    }

    return {
        "resolved_mode": winner_mode,
        "accepted": True,
        "routing_reason": reasons[winner_mode],
        "candidate_scores": candidate_scores,
        "quality": quality,
    }
