"""Motion manifest loading and safe resolution.

The manifest is produced by ``tools/motion_library_builder`` and describes,
for every token in all three recognition datasets, whether a real,
compatible avatar motion JSON exists. This module never invents an entry:
if the manifest has no "ready" record for a token, the token is reported as
missing. It also enforces that every file actually served comes from one of
two allow-listed roots, so a client can never request an arbitrary
filesystem path.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import INTEGRATION_ROOT, settings

from inventory import jordanian_it_token, karsl_token  # noqa: E402

DEFAULT_MANIFEST_PATH = INTEGRATION_ROOT / "data_manifests" / "motion_manifest.json"

# Approved baseline motion files copied verbatim from avatar_web/public/motions.
# Served read-only from this location; never written to by the builder.
BASELINE_MOTIONS_DIR = INTEGRATION_ROOT / "frontend" / "public" / "motions"

VALID_STATUSES = {
    "ready",
    "pending_generation",
    "source_unavailable",
    "extraction_failed",
    "incompatible_source",
    "invalid_motion",
    "validation_required",
}

# Path A (recognized video) namespace bridge: the recognizer's own
# resolved_mode values are NOT the same strings as the manifest's dataset
# namespaces (e.g. "it" vs "jordanian_it") — this is the one place that
# mapping is defined, so recognition.service and motions.manifest can never
# silently disagree about it.
RECOGNITION_MODE_TO_DATASET = {
    "it": "jordanian_it",
    "karsl": "karsl",
    "continuous": "isharah",
}

# The ONLY confirmed, non-guessed bridges from an avatar_sign_lexicon.json
# semantic token to a real motion-library entry. Every other lexicon token
# stays unmapped until a real correspondence is verified — see Section 14/15
# of the integration brief: "START must not automatically become INITIALIZE",
# etc. Do not extend this table by similarity.
#
# Computed by recomputing the exact intersection between the 58 lexicon
# tokens (their own token name AND their own human-curated aliases,
# normalized with avatar_sign_mapper.py's own normalize() for Arabic and a
# plain casefold for English — never a new/guessed normalization) and each
# dataset's canonical labels — see docs/SEMANTIC_BRIDGE.md for the full
# 58-token report and evidence. A bridge existing here does NOT mean the
# motion is ready yet — resolve_lexicon_token()/resolve_file() still check
# the manifest's actual status; isharah entries below stay validation_required
# (see docs/ISHARAH_ALIGNMENT_INVESTIGATION.md) so they resolve as missing
# until that blocker clears, on purpose, not a bug.
#
# Where a token has an exact match in more than one dataset, jordanian_it is
# preferred (it and karsl are the datasets this session can actually
# generate), then karsl, then isharah only when no other dataset matched.
LEXICON_TOKEN_BRIDGE: dict[str, tuple[str, str]] = {
    "STACK": ("jordanian_it", "STACK"),
    "QUEUE": ("jordanian_it", "QUEUE"),
    "DATA_TYPE": ("jordanian_it", "DATATYPE"),
    "POINTER": ("jordanian_it", "POINTER"),
    "DATABASE": ("jordanian_it", "DATABASE"),
    "TOP": ("jordanian_it", "TOP"),
    "FRONT": ("jordanian_it", "FRONT"),
    "REAR": ("jordanian_it", "REAR"),
    "ADD": ("jordanian_it", "ADD"),
    "ELEMENT": ("jordanian_it", "ELEMENT"),
    "ARRAY": ("jordanian_it", "ARRAY"),
    "STORE": ("jordanian_it", "STORE"),
    "ZERO": ("jordanian_it", "ZERO"),
    "FULL": ("jordanian_it", "FULL"),
    "START": ("jordanian_it", "START"),
    "ADDRESS": ("jordanian_it", "ADDRESS"),
    "MEMORY": ("jordanian_it", "MEMORY"),
    "VARIABLE": ("jordanian_it", "VARIABLE"),
    # Exact match via the lexicon's OWN declared alias "relation" (singular)
    # on the RELATIONSHIP entry — not a similarity guess, the alias was
    # already human-authored in avatar_sign_lexicon.json.
    "RELATIONSHIP": ("jordanian_it", "RELATION"),
    "PRIMARY_KEY": ("jordanian_it", "PRIMARY_KEY"),
    "FOREIGN_KEY": ("jordanian_it", "FOREIGN_KEY"),
    "ROW": ("jordanian_it", "ROW"),
    # Exact match via the lexicon's declared alias "unnecessary" on EXTRA.
    "EXTRA": ("jordanian_it", "UNNECESSARY"),
    "TABLE": ("jordanian_it", "TABLE"),
    "COLUMN": ("jordanian_it", "COLUMN"),
    "NOT": ("karsl", "0069"),
    "SELECT": ("karsl", "0185"),
    # Isharah-only exact matches. These will resolve as "missing" until the
    # Isharah alignment blocker clears (see
    # docs/ISHARAH_ALIGNMENT_INVESTIGATION.md) — kept here because the
    # semantic bridge itself is exact and correct; readiness is a separate,
    # honestly-reported concern.
    "INCREMENT": ("isharah", "زياده"),
    "ATTRIBUTE": ("isharah", "صفه"),
    "READ": ("isharah", "قراءه"),
}


@dataclass
class MotionEntry:
    token: str
    dataset: str
    label: str
    motion_id: str
    motion_file: str | None
    source_type: str
    source_path: str | None
    source_frames: dict[str, Any] = field(default_factory=dict)
    status: str = "pending_generation"
    selection_reason: str = ""
    quality_metadata: dict[str, Any] = field(default_factory=dict)
    generator_version: str = ""
    file_checksum: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.dataset, self.token)


class MotionManifest:
    def __init__(self, manifest_path: Path | None = None) -> None:
        self._path = manifest_path or DEFAULT_MANIFEST_PATH
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], MotionEntry] = {}
        self._loaded_at: str | None = None
        self.reload()

    @property
    def path(self) -> Path:
        return self._path

    def reload(self) -> None:
        with self._lock:
            self._entries = {}
            if not self._path.is_file():
                return
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._loaded_at = data.get("generated_at")
            for raw in data.get("entries", []):
                entry = MotionEntry(
                    token=raw["token"],
                    dataset=raw["dataset"],
                    label=raw.get("label", raw["token"]),
                    motion_id=raw.get("motion_id", f"{raw['dataset']}:{raw['token']}"),
                    motion_file=raw.get("motion_file"),
                    source_type=raw.get("source_type", "unknown"),
                    source_path=raw.get("source_path"),
                    source_frames=raw.get("source_frames", {}),
                    status=raw.get("status", "pending_generation"),
                    selection_reason=raw.get("selection_reason", ""),
                    quality_metadata=raw.get("quality_metadata", {}),
                    generator_version=raw.get("generator_version", ""),
                    file_checksum=raw.get("file_checksum"),
                )
                self._entries[entry.key] = entry

    def all_entries(self) -> list[MotionEntry]:
        return list(self._entries.values())

    def get(self, dataset: str, token: str) -> MotionEntry | None:
        return self._entries.get((dataset, token.upper()))

    def resolve_lexicon_token(self, sign_token: str) -> MotionEntry | None:
        """Path B: an avatar_sign_lexicon.json semantic token -> motion.
        Exact bridge table only (LEXICON_TOKEN_BRIDGE) — never fuzzy."""
        bridge = LEXICON_TOKEN_BRIDGE.get(sign_token.upper())
        if not bridge:
            return None
        return self.get(*bridge)

    def resolve_recognized_token(self, resolved_mode: str, raw_token: str) -> MotionEntry | None:
        """Path A: a token the recognizer actually emitted (an IT label, a
        KArSL sign_id or 'KARSL_XXXX' vocabulary token, or an Isharah gloss)
        -> its own dataset's motion entry. Exact, deterministic
        normalization only (jordanian_it_token / karsl_token, the same
        functions the inventory builder uses) — never fuzzy string matching
        or a cross-dataset guess. Isharah tokens are matched as-is: the
        manifest's isharah token IS the exact gloss string, verified
        (680/680) to equal the checkpoint's own continuous-vocabulary token
        for that gloss.
        """
        dataset = RECOGNITION_MODE_TO_DATASET.get(resolved_mode)
        if dataset is None:
            return None
        if dataset == "jordanian_it":
            token = jordanian_it_token(raw_token)
        elif dataset == "karsl":
            token = karsl_token(raw_token)
        else:  # isharah — exact gloss text, no normalization
            token = raw_token
        return self.get(dataset, token)

    def _resolve_safe_path(self, entry: MotionEntry) -> Path | None:
        """Return an absolute path only if it is inside an allow-listed root.

        The builder writes generated motions to
        ``settings.motion_library / <dataset> / <token>.motion.json`` (see
        ``tools/motion_library_builder/builder.py::generate_single_token``),
        so the per-dataset subfolder must be part of the candidate path —
        omitting it here previously made every generated (non-baseline)
        motion 404 despite the manifest correctly saying "ready".
        """
        if entry.status != "ready" or not entry.motion_file:
            return None

        candidates = [
            (BASELINE_MOTIONS_DIR / entry.motion_file).resolve(),
            (settings.motion_library / entry.dataset / entry.motion_file).resolve(),
        ]
        allowed_roots = [BASELINE_MOTIONS_DIR.resolve(), settings.motion_library.resolve()]

        for candidate, root in zip(candidates, allowed_roots):
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
        return None

    def resolve_file(self, dataset: str, token: str) -> Path | None:
        entry = self.get(dataset, token)
        if entry is None:
            return None
        return self._resolve_safe_path(entry)

    def status_summary(self) -> dict[str, Any]:
        entries = self.all_entries()
        by_status: dict[str, int] = {}
        by_dataset: dict[str, dict[str, int]] = {}
        for entry in entries:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
            bucket = by_dataset.setdefault(entry.dataset, {})
            bucket[entry.status] = bucket.get(entry.status, 0) + 1

        return {
            "manifest_path": str(self._path),
            "manifest_exists": self._path.is_file(),
            "generated_at": self._loaded_at,
            "total_entries": len(entries),
            "by_status": by_status,
            "by_dataset": by_dataset,
        }


motion_manifest = MotionManifest()
