"""Wraps the teammates' RAG/LLM/avatar-planner pipeline for the API layer.

``signbridge_router.answer_question`` already returns educational_answer,
simplified_text, avatar_text, sources, and (via ``avatar_sign_mapper``)
sign_tokens/semantic_coverage. Its own "motion_coverage" is always 0.0
today because every entry in the copied ``avatar_sign_lexicon.json`` has
``motion_file: null`` — so this module ignores that field and computes
physical motion coverage itself from the real motion manifest instead of
trusting/propagating a number that can never be anything but zero.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.errors import ConfigurationError, SignBridgeError
from app.motions.manifest import motion_manifest
from app.schemas import AnswerResponse, MotionQueueEntry

import signbridge_router  # noqa: E402  (resolved via legacy_runtime/rag on sys.path)


class RagUnavailableError(SignBridgeError):
    code = "rag_unavailable"
    status_code = 200  # never raised across the recognize path as an HTTP error


def rag_content_available() -> bool:
    try:
        # A cheap check: the Stack domain module and its default index/chunks
        # must be resolvable, since it is always part of the route fallback.
        module = signbridge_router.load_module("stack")
        return bool(module.DEFAULT_CHUNKS.exists() or module.DEFAULT_INDEX.exists())
    except Exception:
        return False


def _build_motion_queue(sign_tokens: list[str]) -> tuple[list[MotionQueueEntry], list[str]]:
    queue: list[MotionQueueEntry] = []
    missing: list[str] = []
    seen: set[str] = set()

    for token in sign_tokens:
        if token in seen:
            continue
        seen.add(token)

        entry = motion_manifest.resolve_lexicon_token(token)
        if entry is not None:
            path = motion_manifest.resolve_file(entry.dataset, entry.token)
            if path is not None:
                queue.append(
                    MotionQueueEntry(
                        token=token,
                        motion_id=entry.motion_id,
                        status="ready",
                        motion_url=f"/api/motions/{entry.dataset}/{entry.token}",
                    )
                )
                continue

        queue.append(MotionQueueEntry(token=token, motion_id=token, status="missing"))
        missing.append(token)

    return queue, missing


def answer_question(question: str, per_domain_k: int = 2) -> AnswerResponse:
    if not settings.groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not configured. Create backend/.env from "
            ".env.example and set a real key to enable educational answers."
        )

    try:
        response: dict[str, Any] = signbridge_router.answer_question(
            question=question, per_domain_k=per_domain_k
        )
    except RuntimeError as exc:
        raise ConfigurationError(str(exc)) from exc

    return _to_answer_response(response)


def _to_answer_response(response: dict[str, Any]) -> AnswerResponse:
    sign_tokens = list(response.get("sign_tokens", []))
    queue, missing = _build_motion_queue(sign_tokens)
    physical_coverage = (
        (len(queue) - len(missing)) / len(queue) if queue else 0.0
    )

    return AnswerResponse(
        educational_answer=response.get("educational_answer"),
        simplified_text=response.get("simplified_text"),
        avatar_text=response.get("avatar_text"),
        routed_subject=response.get("routed_domains", []),
        sources=response.get("sources", []),
        sign_tokens=sign_tokens,
        resolved_motion_sequence=queue,
        missing_motion_tokens=missing,
        semantic_coverage=float(response.get("semantic_coverage", 0.0)),
        physical_motion_coverage=round(physical_coverage, 3),
        avatar_ready=bool(queue) and not missing,
        warnings=response.get("planner_warnings", []),
        educational_answer_available=True,
        unavailable_reason=None,
    )


def unavailable_answer(reason: str) -> AnswerResponse:
    return AnswerResponse(
        educational_answer=None,
        simplified_text=None,
        avatar_text=None,
        routed_subject=[],
        sources=[],
        sign_tokens=[],
        resolved_motion_sequence=[],
        missing_motion_tokens=[],
        semantic_coverage=0.0,
        physical_motion_coverage=0.0,
        avatar_ready=False,
        warnings=[reason],
        educational_answer_available=False,
        unavailable_reason=reason,
    )
