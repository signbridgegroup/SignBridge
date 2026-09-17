"""Generate the Isharah avatar-motion library with per-sample DTW alignment.

This is a production companion to ``diagnose_isharah_alignment.py``.  It:

1. Reads only the 509 source samples selected by the existing 680-token
   Isharah manifest (674 available, six intentionally unavailable).
2. Runs MediaPipe once over each selected sample's complete JPG sequence.
3. Aligns that JPG timeline to the stored Isharah 2-D pose timeline with DTW.
4. Maps every selected gloss interval from pose frames to JPG frames.
5. Writes the existing, approved ``signbridge-motion-v1`` JSON schema.
6. Saves the manifest atomically after every completed token, so rerunning the
   command safely resumes instead of starting over.

It never retrains or changes the recognition model, and it never modifies the
frozen Three.js retargeter/avatar code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


TOOL_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = TOOL_DIR.parent.parent
PROJECT_ROOT = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from diagnose_isharah_alignment import (  # noqa: E402
    apply_coordinate_transform,
    calibrated_overlay_error,
    dtw,
    fit_fixed_coordinate_transform,
    frame_cost_matrix,
    list_images,
    map_source_index,
)
GENERATOR_VERSION = "isharah-dtw-motion-builder-1.0"
MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "holistic_landmarker.task"
POSE_DIR = PROJECT_ROOT / "data" / "processed" / "isharah1000_pose"
KEYPOINTS_PATH = POSE_DIR / "keypoints_flat.npy"
SAMPLES_CSV = POSE_DIR / "samples.csv"
MOTION_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "avatar_motion_library" / "isharah"
)
ALIGNMENT_REPORT_DIR = (
    INTEGRATION_ROOT / "debug" / "isharah_batch_alignment_reports"
)
LOG_PATH = TOOL_DIR / "batch_isharah.log"

# Loaded only for real generation.  Keeping these imports lazy allows the
# dependency-free --self-test to run even outside the project's Windows venv.
builder: Any = None
extractor: Any = None
cv2: Any = None
mp: Any = None
build_features: Any = None


def load_runtime_modules() -> None:
    global builder, extractor, cv2, mp, build_features
    import builder as builder_module
    import cv2 as cv2_module
    import extractor as extractor_module
    import mediapipe as mp_module
    from prepare_isharah_shared_features import build_features as build_features_function

    builder = builder_module
    extractor = extractor_module
    cv2 = cv2_module
    mp = mp_module
    build_features = build_features_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aligned Isharah signbridge-motion-v1 files."
    )
    parser.add_argument(
        "--confirm-full-batch",
        action="store_true",
        help="Required to process the complete remaining Isharah batch.",
    )
    parser.add_argument(
        "--sample-id",
        help="Process one selected source sample only (safe test mode).",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Process at most this many source samples (safe staged test mode).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also retry entries marked incompatible_source/extraction_failed.",
    )
    parser.add_argument("--min-hand-confidence", type=float, default=0.25)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-overlay-error", type=float, default=0.08)
    parser.add_argument(
        "--allow-weak-alignment",
        action="store_true",
        help="Generate even when the calibrated body error exceeds the gate.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_sample_ranges() -> dict[str, tuple[int, int]]:
    import csv

    with SAMPLES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, tuple[int, int]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in output:
            raise RuntimeError(f"Duplicate pose sample ID: {sample_id}")
        output[sample_id] = (int(row["start"]), int(row["end"]))
    return output


def entry_motion_path(entry: dict[str, Any]) -> Path:
    return MOTION_OUTPUT_DIR / f"{entry['token']}.motion.json"


def ready_file_is_valid(entry: dict[str, Any]) -> bool:
    if entry.get("status") != "ready" or not entry.get("motion_file"):
        return False
    path = MOTION_OUTPUT_DIR / entry["motion_file"]
    if not path.is_file():
        return False
    checksum = entry.get("file_checksum")
    return not checksum or builder._sha256(path) == checksum


def selected_entries(
    entries: dict[tuple, dict[str, Any]], retry_failed: bool
) -> list[dict[str, Any]]:
    permitted = {"validation_required"}
    if retry_failed:
        permitted.update({"incompatible_source", "extraction_failed"})
    output = []
    for (dataset, _token), entry in entries.items():
        if dataset != "isharah":
            continue
        if ready_file_is_valid(entry):
            continue
        if entry.get("status") in permitted:
            output.append(entry)
    return output


def group_by_sample(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        sample_id = entry.get("quality_metadata", {}).get("sample_id")
        if not sample_id:
            raise RuntimeError(f"Isharah entry has no sample_id: {entry['token']}")
        groups.setdefault(str(sample_id), []).append(entry)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["source_frames"]["start"], row["token"]))
    return dict(sorted(groups.items()))


def scan_full_jpg_sequence(
    image_paths: list[Path],
    model_path: Path,
    min_hand_confidence: float,
    fps: float,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    pose: list[np.ndarray] = []
    pose_world: list[np.ndarray] = []
    left_hand: list[np.ndarray] = []
    left_hand_world: list[np.ndarray] = []
    right_hand: list[np.ndarray] = []
    right_hand_world: list[np.ndarray] = []
    face_weights: list[list[float]] = []
    face_names: list[str] | None = None
    left_detected: list[bool] = []
    right_detected: list[bool] = []
    pose_detected: list[bool] = []
    points_2d = np.zeros((len(image_paths), 86, 2), dtype=np.float32)
    width = height = 0

    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=min_hand_confidence,
        output_face_blendshapes=False,
    )

    with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(index * 1000.0 / fps))
            result = landmarker.detect_for_video(image, timestamp_ms)

            face_names = extractor._accumulate(
                result,
                pose,
                pose_world,
                left_hand,
                left_hand_world,
                right_hand,
                right_hand_world,
                left_detected,
                right_detected,
                pose_detected,
                face_weights,
                face_names,
            )

            if len(result.pose_landmarks) == 33:
                for joint, landmark in enumerate(result.pose_landmarks[:25]):
                    points_2d[index, 61 + joint] = (
                        landmark.x * width,
                        landmark.y * height,
                    )
            if len(result.right_hand_landmarks) == 21:
                for joint, landmark in enumerate(result.right_hand_landmarks):
                    points_2d[index, joint] = (
                        landmark.x * width,
                        landmark.y * height,
                    )
            if len(result.left_hand_landmarks) == 21:
                for joint, landmark in enumerate(result.left_hand_landmarks):
                    points_2d[index, 21 + joint] = (
                        landmark.x * width,
                        landmark.y * height,
                    )

    return {
        "pose": pose,
        "pose_world": pose_world,
        "left_hand": left_hand,
        "left_hand_world": left_hand_world,
        "right_hand": right_hand,
        "right_hand_world": right_hand_world,
        "face_weights": face_weights,
        "face_names": face_names,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "pose_detected": pose_detected,
        "points_2d": points_2d,
        "width": width,
        "height": height,
    }


def slice_list(values: list[Any], start: int, end: int) -> list[Any]:
    selected = values[start:end]
    if not selected:
        raise ValueError(f"Empty selected frame interval [{start}, {end})")
    return selected


def write_motion(
    entry: dict[str, Any],
    scan: dict[str, Any],
    jpg_start: int,
    jpg_end: int,
    fps: float,
    smoothing_passes: int,
) -> dict[str, Any]:
    output_path = entry_motion_path(entry)
    stats = extractor._assemble_and_write(
        output=output_path,
        name=entry["label"],
        fps=fps,
        source_file=Path(entry["source_path"]).name,
        start_frame=jpg_start,
        rotation="none",
        width=scan["width"],
        height=scan["height"],
        finger_mode="tracked",
        smoothing_passes=smoothing_passes,
        pose=slice_list(scan["pose"], jpg_start, jpg_end),
        pose_world=slice_list(scan["pose_world"], jpg_start, jpg_end),
        left_hand=slice_list(scan["left_hand"], jpg_start, jpg_end),
        left_hand_world=slice_list(scan["left_hand_world"], jpg_start, jpg_end),
        right_hand=slice_list(scan["right_hand"], jpg_start, jpg_end),
        right_hand_world=slice_list(scan["right_hand_world"], jpg_start, jpg_end),
        left_detected=slice_list(scan["left_detected"], jpg_start, jpg_end),
        right_detected=slice_list(scan["right_detected"], jpg_start, jpg_end),
        pose_detected=slice_list(scan["pose_detected"], jpg_start, jpg_end),
        face_weights=slice_list(scan["face_weights"], jpg_start, jpg_end),
        face_names=scan["face_names"],
    )
    return {**stats, "path": output_path}


def save_alignment_report(sample_id: str, report: dict[str, Any]) -> Path:
    ALIGNMENT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = ALIGNMENT_REPORT_DIR / f"{sample_id}.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    return output


def mark_sample_failed(
    manifest: dict[tuple, dict[str, Any]],
    sample_entries: list[dict[str, Any]],
    reason: str,
) -> None:
    for entry in sample_entries:
        key = ("isharah", entry["token"])
        current = manifest[key]
        current["status"] = "incompatible_source"
        current["quality_metadata"] = {
            **current.get("quality_metadata", {}),
            "alignment_error": reason,
        }
        manifest[key] = current
    builder.save_manifest(manifest)


def process_sample(
    sample_id: str,
    sample_entries: list[dict[str, Any]],
    manifest: dict[tuple, dict[str, Any]],
    source_flat: np.ndarray,
    sample_ranges: dict[str, tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    if sample_id not in sample_ranges:
        raise KeyError(f"No pose range found for {sample_id}")
    source_start, source_end = sample_ranges[sample_id]
    source_points = np.asarray(source_flat[source_start:source_end], dtype=np.float32)
    if source_points.shape != (source_end - source_start, 86, 2):
        raise ValueError(f"Unexpected pose slice shape for {sample_id}: {source_points.shape}")

    frames_dir = Path(sample_entries[0]["source_path"])
    image_paths = list_images(frames_dir)
    log(
        f"{sample_id}: scanning {len(image_paths)} JPG frames for "
        f"{len(sample_entries)} token(s); pose frames={len(source_points)}"
    )
    scan = scan_full_jpg_sequence(
        image_paths,
        MEDIAPIPE_MODEL,
        args.min_hand_confidence,
        args.fps,
    )

    source_features = build_features(source_points)
    jpg_features = build_features(scan["points_2d"])
    alignment_cost, path = dtw(frame_cost_matrix(source_features, jpg_features))
    reversed_cost, _ = dtw(frame_cost_matrix(source_features, jpg_features[::-1]))
    control_ratio = reversed_cost / max(alignment_cost, 1e-12)
    transform = fit_fixed_coordinate_transform(source_points, scan["points_2d"], path)
    transformed_source = apply_coordinate_transform(source_points, transform)
    overlay_error = calibrated_overlay_error(
        transformed_source,
        scan["points_2d"],
        path,
        (scan["width"], scan["height"]),
    )
    if overlay_error is None:
        raise RuntimeError("No body landmarks were available for the alignment gate")

    alignment_report: dict[str, Any] = {
        "schema": "signbridge-isharah-production-alignment-v1",
        "sample_id": sample_id,
        "pose_frames": len(source_points),
        "jpg_frames": len(image_paths),
        "jpg_minus_pose_frames": len(image_paths) - len(source_points),
        "dtw_cost": alignment_cost,
        "reversed_control_cost": reversed_cost,
        "reversed_control_ratio": control_ratio,
        "fixed_coordinate_transform_source_to_jpg": transform,
        "median_calibrated_body_overlay_error_fraction_of_image_diagonal": overlay_error,
        "pose_detection_rate": float(np.mean(scan["pose_detected"])),
        "left_hand_detection_rate": float(np.mean(scan["left_detected"])),
        "right_hand_detection_rate": float(np.mean(scan["right_detected"])),
        "quality_gate": {
            "maximum_overlay_error": args.max_overlay_error,
            "passed": overlay_error <= args.max_overlay_error,
            "overridden": bool(args.allow_weak_alignment and overlay_error > args.max_overlay_error),
        },
        "tokens": [],
    }

    if overlay_error > args.max_overlay_error and not args.allow_weak_alignment:
        report_path = save_alignment_report(sample_id, alignment_report)
        reason = (
            f"calibrated overlay error {overlay_error:.6f} exceeds "
            f"gate {args.max_overlay_error:.6f}; report={report_path}"
        )
        mark_sample_failed(manifest, sample_entries, reason)
        log(f"{sample_id}: ALIGNMENT FAILED — {reason}")
        return 0, len(sample_entries)

    ready = failed = 0
    for entry in sample_entries:
        key = ("isharah", entry["token"])
        pose_clip_start = int(entry["source_frames"]["start"])
        pose_clip_end = int(entry["source_frames"]["end"])
        if not (0 <= pose_clip_start < pose_clip_end <= len(source_points)):
            reason = (
                f"Invalid pose interval [{pose_clip_start}, {pose_clip_end}) "
                f"for {len(source_points)} frames"
            )
            current = manifest[key]
            current["status"] = "incompatible_source"
            current["quality_metadata"] = {
                **current.get("quality_metadata", {}),
                "alignment_error": reason,
            }
            manifest[key] = current
            builder.save_manifest(manifest)
            failed += 1
            continue

        jpg_start = map_source_index(path, pose_clip_start)
        jpg_last = map_source_index(path, pose_clip_end - 1)
        jpg_end = min(len(image_paths), jpg_last + 1)
        if jpg_end <= jpg_start:
            jpg_end = min(len(image_paths), jpg_start + 1)

        try:
            stats = write_motion(
                entry,
                scan,
                jpg_start,
                jpg_end,
                args.fps,
                args.smoothing_passes,
            )
        except Exception as exc:  # noqa: BLE001
            current = manifest[key]
            current["status"] = "extraction_failed"
            current["quality_metadata"] = {
                **current.get("quality_metadata", {}),
                "extraction_error": str(exc),
            }
            manifest[key] = current
            builder.save_manifest(manifest)
            alignment_report["tokens"].append(
                {"token": entry["token"], "status": "extraction_failed", "error": str(exc)}
            )
            failed += 1
            log(f"{sample_id}/{entry['token']}: EXTRACTION FAILED — {exc}")
            continue

        output_path = stats.pop("path")
        current = manifest[key]
        current["motion_file"] = output_path.name
        current["status"] = "ready"
        current["generator_version"] = GENERATOR_VERSION
        current["file_checksum"] = builder._sha256(output_path)
        current["selection_reason"] = (
            current.get("selection_reason", "")
            + " | Alignment validated by per-sample DTW against fresh MediaPipe JPG landmarks; "
            + f"mapped pose [{pose_clip_start},{pose_clip_end}) to JPG [{jpg_start},{jpg_end})."
        )
        current["quality_metadata"] = {
            **current.get("quality_metadata", {}),
            **stats,
            "pose_interval_start_inclusive": pose_clip_start,
            "pose_interval_end_exclusive": pose_clip_end,
            "mapped_jpg_start_inclusive": jpg_start,
            "mapped_jpg_end_exclusive": jpg_end,
            "alignment_dtw_cost": alignment_cost,
            "alignment_reversed_control_ratio": control_ratio,
            "alignment_overlay_error": overlay_error,
        }
        manifest[key] = current
        builder.save_manifest(manifest)
        alignment_report["tokens"].append(
            {
                "token": entry["token"],
                "status": "ready",
                "pose_interval": [pose_clip_start, pose_clip_end],
                "jpg_interval": [jpg_start, jpg_end],
                "motion_file": output_path.name,
            }
        )
        ready += 1
        log(
            f"{sample_id}/{entry['token']}: ready — pose "
            f"[{pose_clip_start},{pose_clip_end}) -> JPG [{jpg_start},{jpg_end})"
        )

    save_alignment_report(sample_id, alignment_report)
    return ready, failed


def run_self_test() -> int:
    fake = {
        ("isharah", "a"): {
            "dataset": "isharah",
            "token": "a",
            "status": "validation_required",
            "quality_metadata": {"sample_id": "00_0001"},
            "source_frames": {"start": 0, "end": 3},
        },
        ("isharah", "b"): {
            "dataset": "isharah",
            "token": "b",
            "status": "source_unavailable",
            "quality_metadata": {},
            "source_frames": {},
        },
        ("karsl", "0001"): {
            "dataset": "karsl",
            "token": "0001",
            "status": "ready",
        },
    }
    selected = selected_entries(fake, retry_failed=False)
    grouped = group_by_sample(selected)
    if list(grouped) != ["00_0001"] or grouped["00_0001"][0]["token"] != "a":
        raise AssertionError("Manifest selection/grouping self-test failed")

    cost = np.full((5, 7), 10.0, dtype=np.float64)
    for index in range(5):
        cost[index, min(6, index + 1)] = 0.0
    _score, path = dtw(cost, band=4)
    mapped = [map_source_index(path, index) for index in range(5)]
    if mapped != sorted(mapped):
        raise AssertionError("DTW mapping self-test failed")
    print("Self-test passed")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    load_runtime_modules()
    if not 0.0 <= args.min_hand_confidence <= 1.0:
        raise ValueError("--min-hand-confidence must be between 0 and 1")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be greater than zero")
    if not (args.confirm_full_batch or args.sample_id or args.limit_samples):
        print(
            "Safety gate: choose --sample-id, --limit-samples, or "
            "--confirm-full-batch. Nothing was changed."
        )
        return 2

    for required in (KEYPOINTS_PATH, SAMPLES_CSV, MEDIAPIPE_MODEL):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = builder.load_manifest()
    pending = selected_entries(manifest, args.retry_failed)
    groups = group_by_sample(pending)
    if args.sample_id:
        groups = {
            sample_id: rows
            for sample_id, rows in groups.items()
            if sample_id == args.sample_id
        }
        if not groups:
            log(
                f"No eligible pending entries for sample {args.sample_id}; "
                "it may already be ready or require --retry-failed."
            )
            return 0
    if args.limit_samples is not None:
        groups = dict(list(groups.items())[: args.limit_samples])

    sample_ranges = load_sample_ranges()
    source_flat = np.load(KEYPOINTS_PATH, mmap_mode="r")
    if source_flat.ndim != 3 or source_flat.shape[1:] != (86, 2):
        raise ValueError(f"Unexpected Isharah keypoint shape: {source_flat.shape}")

    total_tokens = sum(len(rows) for rows in groups.values())
    log(
        f"Starting Isharah generation: {len(groups)} source samples, "
        f"{total_tokens} token(s), resume-safe atomic manifest updates"
    )
    started = time.perf_counter()
    ready = failed = 0
    for index, (sample_id, rows) in enumerate(groups.items(), start=1):
        try:
            sample_ready, sample_failed = process_sample(
                sample_id,
                rows,
                manifest,
                source_flat,
                sample_ranges,
                args,
            )
        except KeyboardInterrupt:
            log("Interrupted by user. Completed tokens are preserved; rerun to resume.")
            return 130
        except Exception as exc:  # noqa: BLE001
            reason = f"Unhandled sample error: {type(exc).__name__}: {exc}"
            mark_sample_failed(manifest, rows, reason)
            sample_ready, sample_failed = 0, len(rows)
            log(f"{sample_id}: FAILED — {reason}")
        ready += sample_ready
        failed += sample_failed
        elapsed = time.perf_counter() - started
        average = elapsed / index
        eta = average * (len(groups) - index)
        log(
            f"Progress {index}/{len(groups)} samples | ready tokens={ready} | "
            f"failed tokens={failed} | ETA={eta / 60:.1f} min"
        )

    final_manifest = builder.load_manifest()
    isharah_rows = [
        entry for (dataset, _token), entry in final_manifest.items() if dataset == "isharah"
    ]
    counts: dict[str, int] = {}
    for entry in isharah_rows:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    log(f"Run complete. Isharah status counts: {counts}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
