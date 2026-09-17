"""Builds the three-dataset token inventory from existing metadata only.

Nothing here re-runs dataset preparation, re-detects landmarks for
selection scoring, or retrains anything. It reads CSVs that the original
research pipeline already produced and picks one representative source per
token using the objective, documented criteria from the integration brief.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from label_quality import classify_label

PROJECT_ROOT = Path(r"C:\SignBridge_Project")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jordanian_it_token(label: str) -> str:
    """The single normalization rule for a Jordanian IT label <-> manifest
    token, shared by the inventory builder and the recognition resolver so
    the two can never silently drift apart. Exact/deterministic only — no
    fuzzy matching.
    """
    return label.strip().upper().replace(" ", "_")


def karsl_token(raw: str) -> str:
    """Normalize a KArSL sign identifier (e.g. checkpoint vocabulary token
    'KARSL_0001' or a bare sign_id) to the manifest token form '0001'."""
    value = raw.strip()
    if value.upper().startswith("KARSL_"):
        value = value[len("KARSL_"):]
    return value.zfill(4)


# --------------------------------------------------------------------------- #
# Jordanian IT — 165 classes, 967 audited videos
# --------------------------------------------------------------------------- #

def build_jordanian_it_inventory(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    report_path = project_root / "data" / "processed" / "active_length_report.csv"
    rows = _read_csv(report_path)

    by_label: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    entries: list[dict[str, Any]] = []
    for label, takes in sorted(by_label.items()):
        token = jordanian_it_token(label)

        preferred = [t for t in takes if t["signer_id"].strip() == "07"]
        if preferred:
            chosen = max(
                preferred,
                key=lambda t: float(t["active_left_rate"]) + float(t["active_right_rate"]),
            )
            reason = "signer/take 07 preferred (clearest prior Stack avatar motion reference)"
        else:
            chosen = max(
                takes,
                key=lambda t: float(t["active_left_rate"]) + float(t["active_right_rate"])
                - abs(int(t["active_frames"]) - 100) * 0.01,
            )
            reason = (
                "signer 07 unavailable; selected by highest combined "
                "left/right hand observation rate with a valid active segment"
            )

        video_path = (
            project_root / "data" / "raw" / "sign_videos" / label / chosen["video"]
        )

        entries.append(
            {
                "token": token,
                "dataset": "jordanian_it",
                "label": label,
                "motion_id": f"jordanian_it:{token}",
                "motion_file": None,
                "source_type": "video",
                "source_path": str(video_path),
                "source_frames": {
                    "start": int(chosen["start_frame"]),
                    "end": int(chosen["end_frame"]),
                    "end_convention": "inclusive",
                },
                "status": "pending_generation" if video_path.is_file() else "source_unavailable",
                "selection_reason": (
                    f"{reason} (signer {chosen['signer_id']}, take {chosen['repetition_id']}, "
                    f"video {chosen['video']}, left {chosen['active_left_rate']}%, "
                    f"right {chosen['active_right_rate']}%)"
                ),
                "quality_metadata": {
                    "signer_id": chosen["signer_id"],
                    "repetition_id": chosen["repetition_id"],
                    "active_left_rate": float(chosen["active_left_rate"]),
                    "active_right_rate": float(chosen["active_right_rate"]),
                    "active_frames": int(chosen["active_frames"]),
                },
                "generator_version": "",
                "file_checksum": None,
            }
        )
    return entries


# --------------------------------------------------------------------------- #
# KArSL — 502 sign IDs
# --------------------------------------------------------------------------- #

def build_karsl_inventory(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    samples_path = project_root / "data" / "processed" / "karsl_shared198" / "samples.csv"
    mapping_path = (
        project_root / "data" / "processed" / "unified_sign_data_karsl" / "karsl_sign_mapping.csv"
    )
    videos_root = project_root / "data" / "external" / "karsl" / "videos"

    rows = _read_csv(samples_path)
    labels = {r["sign_id"]: r for r in _read_csv(mapping_path)} if mapping_path.is_file() else {}

    by_sign: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_sign.setdefault(row["sign_id"], []).append(row)

    def score(row: dict[str, str]) -> float:
        return sum(
            float(row.get(field, 0) or 0)
            for field in ("face_rate", "pose_rate", "left_hand_rate", "right_hand_rate")
        )

    entries: list[dict[str, Any]] = []
    for sign_id, samples in sorted(by_sign.items()):
        label_row = labels.get(sign_id, {})
        label_en_raw = label_row.get("label_en", "")
        label_ar_raw = label_row.get("label_ar", "")

        # Do not present a numeric placeholder (e.g. "0", "1") as if it were
        # a meaningful English/Arabic word. karsl_sign_mapping.csv marks
        # these with status "new_token" — the sign genuinely has no assigned
        # human label yet, so the honest label is the sign_id itself.
        if classify_label(label_en_raw) not in ("empty", "numeric_only"):
            label, label_source = label_en_raw, "label_en"
        elif classify_label(label_ar_raw) not in ("empty", "numeric_only"):
            label, label_source = label_ar_raw, "label_ar"
        else:
            label, label_source = f"KArSL {sign_id}", "sign_id_fallback_no_human_label_assigned"
        token = karsl_token(sign_id)

        train_rows = [s for s in samples if s["split"] == "train"]
        pool = train_rows or [s for s in samples if s["split"] != "test"]
        reason_prefix = "training split" if train_rows else "validation split (no training sample available)"
        if not pool:
            pool = samples
            reason_prefix = "test split used only because no train/validation sample exists"

        chosen = max(pool, key=score)
        video_path = videos_root / chosen["relative_path"]

        entries.append(
            {
                "token": token,
                "dataset": "karsl",
                "label": str(label),
                "motion_id": f"karsl:{token}",
                "motion_file": None,
                "source_type": "video",
                "source_path": str(video_path),
                "source_frames": {
                    "start": 0,
                    "end": int(chosen["frames"]) - 1,
                    "end_convention": "inclusive",
                },
                "status": "pending_generation" if video_path.is_file() else "source_unavailable",
                "selection_reason": (
                    f"{reason_prefix}; signer {chosen['signer']}, "
                    f"detection score {score(chosen):.3f} "
                    f"(face {chosen.get('face_rate')}, pose {chosen.get('pose_rate')}, "
                    f"left {chosen.get('left_hand_rate')}, right {chosen.get('right_hand_rate')})"
                ),
                "quality_metadata": {
                    "signer": chosen["signer"],
                    "split": chosen["split"],
                    "sample_id": chosen["sample_id"],
                    "detection_score": round(score(chosen), 4),
                    "label_source": label_source,
                },
                "generator_version": "",
                "file_checksum": None,
            }
        )
    return entries


# --------------------------------------------------------------------------- #
# Isharah — glosses covered by already-downloaded source-group 00
# --------------------------------------------------------------------------- #

UNAVAILABLE_ISHARAH_GLOSSES = {
    "اليوم", "بحرهو", "رجوع", "سيجاره", "طاوله", "وفاه",
}


def build_isharah_inventory(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    clips_path = project_root / "data" / "processed" / "isharah_gloss_clip_library" / "clips.csv"
    frames_root = project_root / "data" / "external" / "isharah" / "frames" / "00"

    rows = _read_csv(clips_path)
    all_glosses = sorted({r["gloss"] for r in rows})

    by_gloss_group00: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["source_group"] == "00":
            by_gloss_group00.setdefault(row["gloss"], []).append(row)

    def sort_key(row: dict[str, str]) -> tuple:
        return (
            row["review_required"] == "True",          # False (0) sorts first
            row["sample_exact_match"] != "True",        # True (0) sorts first
            -float(row["candidate_score"]),
            abs(int(row["clip_frames"]) - 40),
        )

    entries: list[dict[str, Any]] = []
    for gloss in all_glosses:
        # The token is the exact gloss string, because it was verified to be
        # exactly, character-for-character, the token the trained CTC model
        # emits for this gloss (680/680 of clips.csv's glosses were found
        # verbatim in the checkpoint's continuous vocabulary — see
        # docs/MOTION_CONTRACT.md). Using an index-based placeholder here
        # would make Path A (recognized gloss -> motion) impossible to
        # resolve without fuzzy matching, which is explicitly disallowed.
        token = gloss
        candidates = by_gloss_group00.get(gloss)

        if not candidates:
            entries.append(
                {
                    "token": token,
                    "dataset": "isharah",
                    "label": gloss,
                    "motion_id": f"isharah:{token}",
                    "motion_file": None,
                    "source_type": "image_sequence",
                    "source_path": None,
                    "source_frames": {},
                    "status": "source_unavailable",
                    "selection_reason": (
                        "No source-group-00 candidate exists for this gloss; "
                        "additional Isharah archives were not requested/downloaded "
                        "per the approved scope."
                    ),
                    "quality_metadata": {},
                    "generator_version": "",
                    "file_checksum": None,
                }
            )
            continue

        chosen = sorted(candidates, key=sort_key)[0]
        sample_dir = frames_root / chosen["sample_id"]

        entries.append(
            {
                "token": token,
                "dataset": "isharah",
                "label": gloss,
                "motion_id": f"isharah:{token}",
                "motion_file": None,
                "source_type": "image_sequence",
                "source_path": str(sample_dir),
                "source_frames": {
                    "start": int(chosen["relative_start_frame"]),
                    "end": int(chosen["relative_end_frame"]),
                    "end_convention": "exclusive_verified_from_source",
                },
                "status": "validation_required" if sample_dir.is_dir() else "source_unavailable",
                "selection_reason": (
                    f"source_group=00 sample {chosen['sample_id']}, "
                    f"review_required={chosen['review_required']}, "
                    f"sample_exact_match={chosen['sample_exact_match']}, "
                    f"candidate_score={chosen['candidate_score']}, "
                    f"clip_frames={chosen['clip_frames']}. BLOCKED (forensically confirmed, "
                    "see docs/ISHARAH_ALIGNMENT_INVESTIGATION.md): relative_start_frame/"
                    "relative_end_frame index the pose-pickle-derived feature sequence "
                    "(prepare_isharah_pose.py -> prepare_isharah_shared_features.py, both "
                    "explicitly documented as performing NO temporal resampling, one row per "
                    "original pose_data_isharah2000_hands_lips_body.pkl frame). The local "
                    "frames/00/<sample>/frame####.jpg images were independently extracted "
                    "by a different, unrelated process (00.zip's own internal paths are "
                    "'Volumes/SarahAlyami/isharah500/00/...' - a separate 'isharah500' frame "
                    "dump, not the isharah2000 pose source). No timestamp, original "
                    "frame-number, or extraction-parameter metadata ties the two together "
                    "anywhere in this project. Confirmed non-resampling rules out a "
                    "proportional/interpolated mapping. Do not generate from this source "
                    "until an external artifact establishes the correspondence."
                ),
                "quality_metadata": {
                    "sample_id": chosen["sample_id"],
                    "clip_id": chosen["clip_id"],
                    "review_required": chosen["review_required"] == "True",
                    "sample_exact_match": chosen["sample_exact_match"] == "True",
                    "candidate_score": float(chosen["candidate_score"]),
                },
                "generator_version": "",
                "file_checksum": None,
            }
        )
    return entries
