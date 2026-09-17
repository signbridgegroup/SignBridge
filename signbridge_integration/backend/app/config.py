"""Central configuration and bootstrap for the SignBridge integration backend.

This version is portable:
- It resolves paths relative to the GitHub repository by default.
- Environment variables can still override any path.
- Secrets remain loaded from backend/.env.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent.parent
INTEGRATION_ROOT = BACKEND_DIR.parent
REPO_ROOT = INTEGRATION_ROOT.parent

LEGACY_RUNTIME_DIR = BACKEND_DIR / "legacy_runtime"
LEGACY_RAG_DIR = LEGACY_RUNTIME_DIR / "rag"
MOTION_BUILDER_TOOL_DIR = INTEGRATION_ROOT / "tools" / "motion_library_builder"

load_dotenv(BACKEND_DIR / ".env")

for path in (
    str(LEGACY_RUNTIME_DIR),
    str(LEGACY_RAG_DIR),
    str(MOTION_BUILDER_TOOL_DIR),
):
    if path not in sys.path:
        sys.path.insert(0, path)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    checkpoint: Path
    mediapipe_model: Path
    karsl_labels: Path
    motion_library: Path
    groq_api_key: str | None
    groq_model: str
    frontend_origins: list[str]


def load_settings() -> Settings:
    project_root = Path(
        os.getenv("SIGNBRIDGE_PROJECT_ROOT", str(REPO_ROOT))
    )

    checkpoint = _env_path(
        "SIGNBRIDGE_CHECKPOINT",
        project_root / "models" / "best_unified_karsl_multitask.pt",
    )

    mediapipe_model = _env_path(
        "SIGNBRIDGE_MEDIAPIPE_MODEL",
        project_root / "models" / "holistic_landmarker.task",
    )

    karsl_labels = _env_path(
        "SIGNBRIDGE_KARSL_LABELS",
        project_root / "metadata" / "unified" / "karsl_sign_mapping.csv",
    )

    motion_library = _env_path(
        "SIGNBRIDGE_MOTION_LIBRARY",
        project_root / "motion_library",
    )

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

    origins_raw = os.getenv(
        "SIGNBRIDGE_FRONTEND_ORIGINS",
        "http://localhost:5173",
    )

    frontend_origins = [
        origin.strip()
        for origin in origins_raw.split(",")
        if origin.strip()
    ]

    return Settings(
        project_root=project_root,
        checkpoint=checkpoint,
        mediapipe_model=mediapipe_model,
        karsl_labels=karsl_labels,
        motion_library=motion_library,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        frontend_origins=frontend_origins,
    )


settings = load_settings()
