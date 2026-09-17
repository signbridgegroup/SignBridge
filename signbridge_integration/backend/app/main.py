"""SignBridge integration backend (FastAPI).

Endpoints are intentionally thin: all real logic lives in
app.recognition, app.rag, app.motions, and app.services.pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.errors import (
    InvalidTokenError,
    MotionNotFoundError,
    RecognitionError,
    register_error_handlers,
)
from app.motions.manifest import motion_manifest
from app.rag import service as rag_service
from app.recognition.service import recognition_service
from app.schemas import (
    AnswerResponse,
    AskRequest,
    ConfigResponse,
    HealthComponent,
    HealthResponse,
    RecognizeAndAnswerResponse,
)
from app.services.pipeline import recognize_and_answer


app = FastAPI(title="SignBridge Integration API", version="1.0.0")
register_error_handlers(app)


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
#
# SignBridge frontend normally runs on port 1573.
#
# Also allow port 5173 as a fallback Vite development port.
#
# Supported examples:
#
#   http://localhost:1573
#   http://127.0.0.1:1573
#   http://192.168.1.15:1573
#
#   http://localhost:5173
#   http://127.0.0.1:5173
#   http://192.168.1.15:5173
#
# This lets another device on the same local Wi-Fi network access the
# frontend while still allowing it to communicate with this backend.
# --------------------------------------------------------------------------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_origin_regex=(
        r"^http://("
        r"localhost|"
        r"127\.0\.0\.1|"
        r"192\.168\.\d{1,3}\.\d{1,3}"
        r"):(1573|5173)$"
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


SUPPORTED_SUBJECTS = ["Stack", "Queue", "Pointers", "Database"]
SUPPORTED_DATASETS = ["jordanian_it", "karsl", "isharah"]

UPLOAD_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

DIRECTLY_DECODABLE_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}


# --------------------------------------------------------------------------- #
# Health / config
# --------------------------------------------------------------------------- #


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checkpoint_ok = settings.checkpoint.is_file()
    mediapipe_ok = settings.mediapipe_model.is_file()
    karsl_labels_ok = settings.karsl_labels.is_file()
    rag_ok = rag_service.rag_content_available()
    manifest_summary = motion_manifest.status_summary()

    return HealthResponse(
        backend=HealthComponent(
            ok=True,
            detail="running",
        ),
        checkpoint=HealthComponent(
            ok=checkpoint_ok,
            detail=str(settings.checkpoint),
        ),
        mediapipe_model=HealthComponent(
            ok=mediapipe_ok,
            detail=str(settings.mediapipe_model),
        ),
        karsl_labels=HealthComponent(
            ok=karsl_labels_ok,
            detail=str(settings.karsl_labels),
        ),
        groq_configured=HealthComponent(
            ok=bool(settings.groq_api_key),
            detail=(
                "configured"
                if settings.groq_api_key
                else "GROQ_API_KEY not set"
            ),
        ),
        rag_content=HealthComponent(
            ok=rag_ok,
            detail=(
                "stack domain content resolvable"
                if rag_ok
                else "stack domain content missing"
            ),
        ),
        motion_library=HealthComponent(
            ok=manifest_summary["manifest_exists"],
            detail=f"{manifest_summary['total_entries']} manifest entries",
        ),
        recognition_model_loaded=recognition_service.is_loaded,
    )


@app.get("/api/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    return ConfigResponse(
        groq_configured=bool(settings.groq_api_key),
        groq_model=settings.groq_model,
        frontend_origins=settings.frontend_origins,
        recognition_modes=[
            "auto",
            "it",
            "karsl",
            "continuous",
        ],
        supported_subjects=SUPPORTED_SUBJECTS,
        supported_datasets=SUPPORTED_DATASETS,
    )


# --------------------------------------------------------------------------- #
# Motions
# --------------------------------------------------------------------------- #


@app.get("/api/motions")
def list_motions() -> dict:
    entries = motion_manifest.all_entries()

    ready = [
        e
        for e in entries
        if e.status == "ready"
    ]

    pending = [
        e
        for e in entries
        if e.status == "pending_generation"
    ]

    unavailable = [
        e
        for e in entries
        if e.status == "source_unavailable"
    ]

    failed = [
        e
        for e in entries
        if e.status
        in {
            "extraction_failed",
            "incompatible_source",
            "invalid_motion",
        }
    ]

    datasets = sorted(
        {
            e.dataset
            for e in entries
        }
    )

    return {
        "total_manifest_entries": len(entries),
        "ready_count": len(ready),
        "pending_count": len(pending),
        "unavailable_count": len(unavailable),
        "failed_count": len(failed),
        "datasets": datasets,
        "tokens": [
            {
                "token": e.token,
                "dataset": e.dataset,
                "label": e.label,
                "status": e.status,
                "motion_id": e.motion_id,
            }
            for e in entries
        ],
    }


@app.get("/api/motions/status")
def motions_status() -> dict:
    return motion_manifest.status_summary()


@app.get("/api/motions/{dataset}/{token}")
def get_motion(dataset: str, token: str) -> FileResponse:
    if not dataset.isalnum() and "_" not in dataset:
        raise InvalidTokenError(
            "Invalid dataset identifier."
        )

    if not all(
        ch.isalnum() or ch in "_-"
        for ch in token
    ):
        raise InvalidTokenError(
            "Invalid token identifier."
        )

    path = motion_manifest.resolve_file(
        dataset,
        token,
    )

    if path is None:
        raise MotionNotFoundError(
            f"No ready motion for {dataset}:{token}."
        )

    return FileResponse(
        path,
        media_type="application/json",
    )


# --------------------------------------------------------------------------- #
# Ask (typed question)
# --------------------------------------------------------------------------- #


@app.post(
    "/api/ask",
    response_model=AnswerResponse,
)
def ask(
    request: AskRequest,
) -> AnswerResponse:
    return rag_service.answer_question(
        request.question,
        per_domain_k=request.per_domain_k,
    )


# --------------------------------------------------------------------------- #
# Recognize (camera recording or uploaded video)
# --------------------------------------------------------------------------- #


def _ensure_decodable(
    source: Path,
    workdir: Path,
) -> Path:
    """Convert a browser recording to a decodable temp file if needed.

    Only ever touches the one temporary uploaded clip; never the training
    dataset. Falls back clearly if ffmpeg is unavailable.
    """

    if (
        source.suffix.lower()
        in DIRECTLY_DECODABLE_EXTENSIONS
    ):
        return source

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RecognitionError(
            "This browser recording format could not be decoded directly and "
            "ffmpeg is not installed on the server to convert it. Install "
            "ffmpeg or upload an .mp4/.mov file instead."
        )

    converted = (
        workdir
        / f"{source.stem}_converted.mp4"
    )

    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-an",
            str(converted),
        ],
        capture_output=True,
        text=True,
    )

    if (
        result.returncode != 0
        or not converted.is_file()
    ):
        raise RecognitionError(
            "The uploaded recording could not be converted to a supported "
            "format. ffmpeg said: "
            f"{result.stderr[-500:] if result.stderr else 'unknown error'}"
        )

    return converted


@app.post(
    "/api/recognize",
    response_model=RecognizeAndAnswerResponse,
)
async def recognize(
    video: UploadFile = File(...),
    mode: str = Form("auto"),
    top_k: int = Form(5),
) -> RecognizeAndAnswerResponse:

    suffix = Path(
        video.filename or "clip.webm"
    ).suffix.lower()

    if (
        suffix
        not in UPLOAD_VIDEO_EXTENSIONS
    ):
        raise InvalidTokenError(
            f"Unsupported video extension: {suffix}"
        )

    with tempfile.TemporaryDirectory(
        prefix="signbridge_upload_"
    ) as tmp:

        workdir = Path(tmp)

        upload_path = (
            workdir
            / f"upload_{uuid.uuid4().hex}{suffix}"
        )

        try:
            with upload_path.open("wb") as handle:
                shutil.copyfileobj(
                    video.file,
                    handle,
                )

            decodable_path = _ensure_decodable(
                upload_path,
                workdir,
            )

            return recognize_and_answer(
                decodable_path,
                mode=mode,
                top_k=top_k,
            )

        finally:
            await video.close()

            # TemporaryDirectory cleans itself up; this block guarantees no
            # partially written file is left behind even on an exception
            # raised before the context manager exits normally.


# --------------------------------------------------------------------------- #
# Admin: single-token motion generation
# (local/admin only, one token at a time)
# --------------------------------------------------------------------------- #


@app.post(
    "/api/motions/generate/{dataset}/{token}"
)
def generate_motion(
    dataset: str,
    token: str,
) -> dict:

    if dataset not in SUPPORTED_DATASETS:
        raise InvalidTokenError(
            f"Unknown dataset source: {dataset}"
        )

    if (
        not token
        or not all(
            ch.isalnum() or ch in "_-"
            for ch in token
        )
    ):
        raise InvalidTokenError(
            "Invalid token identifier."
        )

    entry = motion_manifest.get(
        dataset,
        token,
    )

    if entry is None:
        raise MotionNotFoundError(
            f"{dataset}:{token} "
            "is not in the manifest inventory."
        )

    # Local/admin single-token generation delegates to the motion library
    # builder's own generate-token command rather than duplicating its logic
    # here, and rather than exposing any batch/unrestricted path.

    import sys as _sys

    tools_dir = str(
        settings.project_root
        / "signbridge_integration"
        / "tools"
        / "motion_library_builder"
    )

    if tools_dir not in _sys.path:
        _sys.path.insert(
            0,
            tools_dir,
        )

    from builder import generate_single_token  # noqa: E402

    result = generate_single_token(
        dataset,
        token,
    )

    motion_manifest.reload()

    return result