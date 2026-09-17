"""Pydantic response/request models for the SignBridge integration API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RecognitionMode = Literal["auto", "it", "karsl", "continuous"]
# "rejected" is only ever a *resolved* value (never something a caller can
# request): the genuinely-automatic router's safe-rejection outcome when
# evidence is too weak or too ambiguous to trust any candidate path.
ResolvedRecognitionMode = Literal["it", "karsl", "continuous", "rejected"]


class HealthComponent(BaseModel):
    ok: bool
    detail: str = ""


class HealthResponse(BaseModel):
    backend: HealthComponent
    checkpoint: HealthComponent
    mediapipe_model: HealthComponent
    karsl_labels: HealthComponent
    groq_configured: HealthComponent
    rag_content: HealthComponent
    motion_library: HealthComponent
    recognition_model_loaded: bool


class ConfigResponse(BaseModel):
    groq_configured: bool
    groq_model: str
    frontend_origins: list[str]
    recognition_modes: list[RecognitionMode]
    supported_subjects: list[str]
    supported_datasets: list[str]


class TopKPrediction(BaseModel):
    label: str
    confidence: float


class RecognitionResult(BaseModel):
    requested_mode: RecognitionMode
    resolved_mode: ResolvedRecognitionMode
    # True unless the automatic router safely rejected the recording
    # (resolved_mode == "rejected"). Explicit (non-"auto") requests are
    # always accepted, since the caller chose the path deliberately.
    accepted: bool = True
    # Plain-English explanation of why the automatic router chose this
    # resolved_mode (or rejected the recording) - developer/diagnostic
    # detail, safe to omit from the normal student-facing UI.
    routing_reason: str | None = None
    # Per-candidate-path routing evidence (it/karsl/continuous), only
    # populated for "auto" requests. Diagnostic detail, not for the normal
    # student-facing UI.
    candidate_scores: dict[str, object] = Field(default_factory=dict)
    # Video-quality/temporal diagnostics (hand/pose observation rates,
    # active-segment count, etc.), only populated for "auto" requests.
    quality: dict[str, object] = Field(default_factory=dict)
    predicted_label: str | None = None
    # Raw, unformatted token (IT label / KArSL sign_id) used for exact
    # Path A motion resolution. Continuous mode has no single token — see
    # decoded_gloss_sequence, resolved one gloss at a time, in order.
    predicted_token: str | None = None
    decoded_gloss_sequence: list[str] = Field(default_factory=list)
    decoded_gloss_confidences: list[float] = Field(default_factory=list)
    top_k: list[TopKPrediction] = Field(default_factory=list)
    temporal_token_evidence: list[TopKPrediction] = Field(default_factory=list)
    confidence: float | None = None
    frame_count: int
    active_start_frame: int
    active_end_frame: int
    active_source: str
    left_hand_observed_rate: float
    right_hand_observed_rate: float
    dataset_source: str
    motion_dataset_namespace: str | None = None
    device: str
    warnings: list[str] = Field(default_factory=list)


class RecognizedMotion(BaseModel):
    """Path A: the motion for the sign that was actually recognized —
    distinct from Path B's RAG-derived AnswerResponse.resolved_motion_sequence.
    """
    token: str
    dataset: str
    status: Literal["ready", "missing"]
    motion_url: str | None = None


class MotionQueueEntry(BaseModel):
    token: str
    motion_id: str
    status: Literal["ready", "missing"]
    motion_url: str | None = None


class AnswerResponse(BaseModel):
    educational_answer: str | None = None
    simplified_text: str | None = None
    avatar_text: str | None = None
    routed_subject: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    sign_tokens: list[str] = Field(default_factory=list)
    resolved_motion_sequence: list[MotionQueueEntry] = Field(default_factory=list)
    missing_motion_tokens: list[str] = Field(default_factory=list)
    semantic_coverage: float = 0.0
    physical_motion_coverage: float = 0.0
    avatar_ready: bool = False
    warnings: list[str] = Field(default_factory=list)
    educational_answer_available: bool = True
    unavailable_reason: str | None = None


class AskRequest(BaseModel):
    question: str
    per_domain_k: int = 2


class RecognizeAndAnswerResponse(BaseModel):
    recognition: RecognitionResult
    recognized_gloss: str
    question: str | None = None
    answer: AnswerResponse
    # Path A: motion(s) for the sign(s) actually recognized, resolved
    # directly against the recognized dataset — independent of whether the
    # RAG/semantic-planner path (answer.resolved_motion_sequence, Path B)
    # found anything. In continuous mode this has one entry per decoded
    # gloss, in order.
    recognized_motion_sequence: list[RecognizedMotion] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str
