"""Video -> recognition -> gloss -> question -> RAG glue.

This intentionally does NOT reuse ``signbridge_video_to_rag.py`` from the
teammates' zip: that module is built around ``predict_unified_video.py`` and
``best_unified_ctc.pt``, an older checkpoint that predates KArSL support.
Instead this combines the new ``RecognitionService`` (the traced,
checkpoint-compatible path) with the teammates' still-current
``glosses_to_question`` bridge and RAG router.
"""

from __future__ import annotations

from pathlib import Path

from app.errors import ConfigurationError, SignBridgeError
from app.motions.manifest import motion_manifest
from app.rag import service as rag_service
from app.recognition.service import recognition_service
from app.schemas import (
    AnswerResponse,
    RecognitionResult,
    RecognizedMotion,
    RecognizeAndAnswerResponse,
    TopKPrediction,
)

import signbridge_gloss_to_rag  # noqa: E402


def _recognition_to_gloss_text(result: dict) -> str | None:
    mode = result["resolved_mode"]
    if mode in {"it", "karsl"} and result.get("predicted_label"):
        return str(result["predicted_label"]).split(" (")[0]
    if result.get("decoded_gloss_sequence"):
        return " ".join(result["decoded_gloss_sequence"])
    return None


def _resolve_path_a_motions(raw_result: dict) -> list[RecognizedMotion]:
    """Path A: recognized dataset + recognized token(s) -> exact dataset
    motion entry, in order. Independent of the RAG/semantic-planner path.
    """
    resolved_mode = raw_result["resolved_mode"]
    dataset = raw_result.get("motion_dataset_namespace")
    if dataset is None:
        return []

    if resolved_mode == "continuous":
        raw_tokens = raw_result.get("decoded_gloss_sequence", [])
    else:
        token = raw_result.get("predicted_token")
        raw_tokens = [token] if token else []

    resolved: list[RecognizedMotion] = []
    for raw_token in raw_tokens:
        entry = motion_manifest.resolve_recognized_token(resolved_mode, raw_token)
        if entry is not None:
            path = motion_manifest.resolve_file(entry.dataset, entry.token)
        else:
            path = None
        if entry is not None and path is not None:
            resolved.append(
                RecognizedMotion(
                    token=entry.token,
                    dataset=entry.dataset,
                    status="ready",
                    motion_url=f"/api/motions/{entry.dataset}/{entry.token}",
                )
            )
        else:
            resolved.append(RecognizedMotion(token=raw_token, dataset=dataset, status="missing"))
    return resolved


def recognize_and_answer(
    video_path: Path,
    mode: str = "auto",
    top_k: int = 5,
    per_domain_k: int = 2,
    cpu: bool = False,
) -> RecognizeAndAnswerResponse:
    raw_result = recognition_service.recognize(video_path, mode=mode, top_k=top_k, cpu=cpu)

    recognition = RecognitionResult(
        requested_mode=raw_result["requested_mode"],
        resolved_mode=raw_result["resolved_mode"],
        accepted=raw_result.get("accepted", True),
        routing_reason=raw_result.get("routing_reason"),
        candidate_scores=raw_result.get("candidate_scores", {}),
        quality=raw_result.get("quality", {}),
        predicted_label=raw_result.get("predicted_label"),
        predicted_token=raw_result.get("predicted_token"),
        decoded_gloss_sequence=raw_result.get("decoded_gloss_sequence", []),
        decoded_gloss_confidences=raw_result.get("decoded_gloss_confidences", []),
        top_k=[TopKPrediction(**item) for item in raw_result.get("top_k", [])],
        temporal_token_evidence=[
            TopKPrediction(**item) for item in raw_result.get("temporal_token_evidence", [])
        ],
        confidence=raw_result.get("confidence"),
        frame_count=raw_result["frame_count"],
        active_start_frame=raw_result["active_start_frame"],
        active_end_frame=raw_result["active_end_frame"],
        active_source=raw_result["active_source"],
        left_hand_observed_rate=raw_result["left_hand_observed_rate"],
        right_hand_observed_rate=raw_result["right_hand_observed_rate"],
        dataset_source=raw_result["dataset_source"],
        motion_dataset_namespace=raw_result.get("motion_dataset_namespace"),
        device=raw_result["device"],
        warnings=raw_result.get("warnings", []),
    )

    recognized_motion_sequence = _resolve_path_a_motions(raw_result)
    gloss_text = _recognition_to_gloss_text(raw_result)
    question: str | None = None
    answer: AnswerResponse

    if not gloss_text:
        if raw_result.get("resolved_mode") == "rejected":
            answer = rag_service.unavailable_answer(
                "The recording was rejected by the recognizer: "
                f"{raw_result.get('routing_reason', 'insufficient or ambiguous signing evidence')}."
            )
        else:
            answer = rag_service.unavailable_answer(
                "The recognizer produced no usable gloss/label for this clip."
            )
    else:
        try:
            question = signbridge_gloss_to_rag.glosses_to_question(gloss_text)
            answer = rag_service.answer_question(question, per_domain_k=per_domain_k)
        except ConfigurationError as exc:
            answer = rag_service.unavailable_answer(str(exc.detail))
        except (ValueError, SignBridgeError) as exc:
            answer = rag_service.unavailable_answer(
                f"Recognized '{gloss_text}' but it does not map to a supported "
                f"educational topic: {exc}"
            )

    return RecognizeAndAnswerResponse(
        recognition=recognition,
        recognized_gloss=gloss_text or "",
        question=question,
        answer=answer,
        recognized_motion_sequence=recognized_motion_sequence,
    )
