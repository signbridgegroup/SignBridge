from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANDMARKS_ROOT = PROJECT_ROOT / "data" / "processed" / "landmarks"
DEFAULT_ACTIVE_REPORT = PROJECT_ROOT / "data" / "processed" / "active_length_report.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "model_data"

EXPECTED_RAW_FEATURES = 1629
DEFAULT_TARGET_FRAMES = 200
DEFAULT_MAX_SHORT_GAP = 5

# Layout produced by extract_landmarks.py.
FACE_END = 468 * 3
POSE_END = FACE_END + 33 * 3
LEFT_HAND_END = POSE_END + 21 * 3
RIGHT_HAND_END = LEFT_HAND_END + 21 * 3

# Pose 11..24: shoulders, elbows, wrists, approximate hand points, and hips.
UPPER_BODY_POSE_INDICES = tuple(range(11, 25))

# Official MediaPipe face-contour points: lips, eyes, and eyebrows.
LIP_INDICES = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87,
    88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
    308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405,
    409, 415,
)
LEFT_EYE_INDICES = (
    249, 263, 362, 373, 374, 380, 381, 382,
    384, 385, 386, 387, 388, 390, 398, 466,
)
RIGHT_EYE_INDICES = (
    7, 33, 133, 144, 145, 153, 154, 155,
    157, 158, 159, 160, 161, 163, 173, 246,
)
LEFT_EYEBROW_INDICES = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
RIGHT_EYEBROW_INDICES = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
FACE_INDICES = tuple(
    sorted(
        set(
            LIP_INDICES
            + LEFT_EYE_INDICES
            + RIGHT_EYE_INDICES
            + LEFT_EYEBROW_INDICES
            + RIGHT_EYEBROW_INDICES
        )
    )
)

POSE_FEATURES = len(UPPER_BODY_POSE_INDICES) * 3
HAND_FEATURES = 21 * 3
FACE_FEATURES = len(FACE_INDICES) * 3

# Global pose (42) + global hands (126) + face (276)
# + local hands (126) + two observed-hand masks = 572.
TARGET_FEATURES = POSE_FEATURES + 4 * HAND_FEATURES + FACE_FEATURES + 2

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
FACE_NOSE_ANCHOR_INDEX = 1
RIGHT_EYE_OUTER_INDEX = 33
LEFT_EYE_OUTER_INDEX = 263
HAND_WRIST_INDEX = 0
HAND_MIDDLE_MCP_INDEX = 9

PRESENCE_EPSILON = 1e-8
SCALE_EPSILON = 1e-4


if len(FACE_INDICES) != 92:
    raise RuntimeError(f"Expected 92 selected face landmarks, got {len(FACE_INDICES)}")
if TARGET_FEATURES != 572:
    raise RuntimeError(f"Expected 572 output features, got {TARGET_FEATURES}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare research-based SignBridge landmark sequences."
    )
    parser.add_argument(
        "--landmarks-root",
        type=Path,
        default=DEFAULT_LANDMARKS_ROOT,
    )
    parser.add_argument(
        "--active-report",
        type=Path,
        default=DEFAULT_ACTIVE_REPORT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
    )
    parser.add_argument(
        "--max-short-gap",
        type=int,
        default=DEFAULT_MAX_SHORT_GAP,
        help="Maximum missing-hand gap to interpolate before resampling.",
    )
    return parser.parse_args()


def record_key(label: str, video_name: str) -> tuple[str, str]:
    return label.strip().casefold(), Path(video_name).stem.strip().casefold()


def load_active_report(report_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Active-length report not found: {report_path}")

    records: dict[tuple[str, str], dict[str, str]] = {}
    with report_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "label",
            "video",
            "signer_id",
            "repetition_id",
            "raw_frames",
            "start_frame",
            "end_frame",
            "active_frames",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Active report is missing columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            key = record_key(row["label"], row["video"])
            if key in records:
                raise ValueError(
                    "Duplicate active-report row for "
                    f"label={row['label']!r}, video={row['video']!r}"
                )
            records[key] = row

    if not records:
        raise ValueError("The active-length report contains no data rows.")
    return records


def hand_presence(hand: np.ndarray) -> np.ndarray:
    return np.any(np.abs(hand) > PRESENCE_EPSILON, axis=(1, 2))


def safe_scales(
    raw_scales: np.ndarray,
    valid_frames: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(raw_scales) & (raw_scales > SCALE_EPSILON)
    if valid_frames is not None:
        valid &= valid_frames

    usable = raw_scales[valid]
    fallback = float(np.median(usable)) if usable.size else 1.0
    return np.where(
        np.isfinite(raw_scales) & (raw_scales > SCALE_EPSILON),
        raw_scales,
        fallback,
    ).astype(np.float32)


def shoulder_reference(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = pose[:, LEFT_SHOULDER_INDEX, :]
    right = pose[:, RIGHT_SHOULDER_INDEX, :]
    origin = (left + right) / 2.0
    shoulder_width = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
    return origin.astype(np.float32), safe_scales(shoulder_width)


def normalize_pose(
    pose: np.ndarray,
    shoulder_origin: np.ndarray,
    shoulder_scale: np.ndarray,
) -> np.ndarray:
    selected = pose[:, UPPER_BODY_POSE_INDICES, :].copy()
    selected -= shoulder_origin[:, None, :]
    selected /= shoulder_scale[:, None, None]
    return selected.astype(np.float32)


def normalize_face(
    face: np.ndarray,
    shoulder_origin: np.ndarray,
    shoulder_scale: np.ndarray,
) -> np.ndarray:
    selected = face[:, FACE_INDICES, :].copy()

    # Keep X/Y relative to the body so face/hand positions remain comparable.
    selected[:, :, :2] -= shoulder_origin[:, None, :2]
    selected[:, :, :2] /= shoulder_scale[:, None, None]

    # Face Z is face-local in MediaPipe, so normalize it separately.
    eye_width = np.linalg.norm(
        face[:, RIGHT_EYE_OUTER_INDEX, :2]
        - face[:, LEFT_EYE_OUTER_INDEX, :2],
        axis=1,
    )
    eye_scale = safe_scales(eye_width)
    nose_z = face[:, FACE_NOSE_ANCHOR_INDEX, 2]
    selected[:, :, 2] = (
        selected[:, :, 2] - nose_z[:, None]
    ) / eye_scale[:, None]
    return selected.astype(np.float32)


def normalize_hand_global(
    hand: np.ndarray,
    observed: np.ndarray,
    shoulder_origin: np.ndarray,
    shoulder_scale: np.ndarray,
) -> np.ndarray:
    output = hand.copy()
    output[:, :, :2] -= shoulder_origin[:, None, :2]
    output[:, :, :2] /= shoulder_scale[:, None, None]

    # MediaPipe hand Z is wrist-relative, so only scale it here.
    output[:, :, 2] /= shoulder_scale[:, None]
    output[~observed] = 0.0
    return output.astype(np.float32)


def normalize_hand_local(hand: np.ndarray, observed: np.ndarray) -> np.ndarray:
    wrist = hand[:, HAND_WRIST_INDEX, :]
    palm_scale_raw = np.linalg.norm(
        hand[:, HAND_MIDDLE_MCP_INDEX, :2]
        - hand[:, HAND_WRIST_INDEX, :2],
        axis=1,
    )
    palm_scale = safe_scales(palm_scale_raw, observed)
    output = (hand - wrist[:, None, :]) / palm_scale[:, None, None]
    output[~observed] = 0.0
    return output.astype(np.float32)


def fill_short_gaps(
    values: np.ndarray,
    observed: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill only short bounded gaps; preserve the original observed mask."""
    output = values.copy()
    value_valid = observed.copy()
    if max_gap <= 0 or len(observed) < 3:
        return output, value_valid

    index = 0
    frame_count = len(observed)
    while index < frame_count:
        if observed[index]:
            index += 1
            continue

        gap_start = index
        while index < frame_count and not observed[index]:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1
        left_index = gap_start - 1
        right_index = index

        if (
            gap_length <= max_gap
            and left_index >= 0
            and right_index < frame_count
            and observed[left_index]
            and observed[right_index]
        ):
            denominator = gap_length + 1
            for offset, frame_index in enumerate(
                range(gap_start, gap_end + 1),
                start=1,
            ):
                weight = offset / denominator
                output[frame_index] = (
                    (1.0 - weight) * output[left_index]
                    + weight * output[right_index]
                )
            value_valid[gap_start : gap_end + 1] = True

    return output, value_valid


def linear_resample(values: np.ndarray, target_frames: int) -> np.ndarray:
    source_frames = values.shape[0]
    flat = values.reshape(source_frames, -1)
    if source_frames == target_frames:
        return flat.astype(np.float32, copy=True)
    if source_frames == 1:
        return np.repeat(flat, target_frames, axis=0).astype(np.float32)

    source_time = np.linspace(0.0, 1.0, source_frames)
    target_time = np.linspace(0.0, 1.0, target_frames)
    output = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for feature_index in range(flat.shape[1]):
        output[:, feature_index] = np.interp(
            target_time,
            source_time,
            flat[:, feature_index],
        )
    return output


def resample_hand_features(
    values: np.ndarray,
    observed: np.ndarray,
    max_gap: int,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = values.reshape(values.shape[0], -1)
    filled, value_valid = fill_short_gaps(flat, observed, max_gap)
    output = linear_resample(filled, target_frames)

    nearest = np.rint(
        np.linspace(0, len(observed) - 1, target_frames)
    ).astype(np.int64)
    target_observed = observed[nearest]
    target_value_valid = value_valid[nearest]
    output[~target_value_valid] = 0.0
    return output, target_observed, target_value_valid


def prepare_sequence(
    raw_features: np.ndarray,
    start_frame: int,
    end_frame: int,
    target_frames: int,
    max_short_gap: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    if raw_features.ndim != 2 or raw_features.shape[1] != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"Unexpected landmark shape {raw_features.shape}; "
            f"expected (frames, {EXPECTED_RAW_FEATURES})."
        )
    if raw_features.shape[0] == 0:
        raise ValueError("Landmark array contains no frames.")

    total_frames = raw_features.shape[0]
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    active = raw_features[start_frame : end_frame + 1].astype(np.float32)

    face = active[:, :FACE_END].reshape(-1, 468, 3)
    pose = active[:, FACE_END:POSE_END].reshape(-1, 33, 3)
    left_hand = active[:, POSE_END:LEFT_HAND_END].reshape(-1, 21, 3)
    right_hand = active[:, LEFT_HAND_END:RIGHT_HAND_END].reshape(-1, 21, 3)

    left_observed = hand_presence(left_hand)
    right_observed = hand_presence(right_hand)
    shoulder_origin, shoulder_scale = shoulder_reference(pose)

    pose_global = normalize_pose(pose, shoulder_origin, shoulder_scale)
    face_selected = normalize_face(face, shoulder_origin, shoulder_scale)
    left_global = normalize_hand_global(
        left_hand,
        left_observed,
        shoulder_origin,
        shoulder_scale,
    )
    right_global = normalize_hand_global(
        right_hand,
        right_observed,
        shoulder_origin,
        shoulder_scale,
    )
    left_local = normalize_hand_local(left_hand, left_observed)
    right_local = normalize_hand_local(right_hand, right_observed)

    pose_output = linear_resample(pose_global, target_frames)
    face_output = linear_resample(face_selected, target_frames)
    left_global_output, left_mask, left_valid = resample_hand_features(
        left_global,
        left_observed,
        max_short_gap,
        target_frames,
    )
    right_global_output, right_mask, right_valid = resample_hand_features(
        right_global,
        right_observed,
        max_short_gap,
        target_frames,
    )
    left_local_output, _, _ = resample_hand_features(
        left_local,
        left_observed,
        max_short_gap,
        target_frames,
    )
    right_local_output, _, _ = resample_hand_features(
        right_local,
        right_observed,
        max_short_gap,
        target_frames,
    )

    sequence = np.concatenate(
        [
            pose_output,
            left_global_output,
            right_global_output,
            face_output,
            left_local_output,
            right_local_output,
            left_mask.astype(np.float32)[:, None],
            right_mask.astype(np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    if sequence.shape != (target_frames, TARGET_FEATURES):
        raise RuntimeError(f"Internal output-shape error: {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("Prepared sequence contains NaN or infinite values.")

    stats: dict[str, float | int] = {
        "active_frames_used": int(active.shape[0]),
        "resampled_left_observed_rate": round(float(left_mask.mean() * 100), 2),
        "resampled_right_observed_rate": round(float(right_mask.mean() * 100), 2),
        "resampled_left_imputed_rate": round(
            float(np.mean(left_valid & ~left_mask) * 100),
            2,
        ),
        "resampled_right_imputed_rate": round(
            float(np.mean(right_valid & ~right_mask) * 100),
            2,
        ),
    }
    return sequence, stats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.target_frames <= 1:
        raise ValueError("--target-frames must be greater than 1.")
    if args.max_short_gap < 0:
        raise ValueError("--max-short-gap cannot be negative.")
    if not args.landmarks_root.is_dir():
        raise FileNotFoundError(f"Landmarks folder not found: {args.landmarks_root}")

    report_records = load_active_report(args.active_report)
    landmark_paths = sorted(
        args.landmarks_root.rglob("*.npy"),
        key=lambda path: str(path).casefold(),
    )
    if not landmark_paths:
        raise FileNotFoundError(f"No .npy files found in: {args.landmarks_root}")

    labels = sorted(
        {path.parent.name for path in landmark_paths},
        key=str.casefold,
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    sample_count = len(landmark_paths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_x_path = args.output_dir / "X.building.npy"
    final_x_path = args.output_dir / "X.npy"
    if temporary_x_path.exists():
        temporary_x_path.unlink()

    X = np.lib.format.open_memmap(
        temporary_x_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, args.target_frames, TARGET_FEATURES),
    )
    y = np.empty(sample_count, dtype=np.int64)
    metadata_rows: list[dict[str, object]] = []
    used_report_keys: set[tuple[str, str]] = set()

    try:
        for sample_index, landmark_path in enumerate(landmark_paths):
            label = landmark_path.parent.name
            key = record_key(label, landmark_path.name)
            report_row = report_records.get(key)
            if report_row is None:
                raise KeyError(
                    "No active-report row matches landmark file: "
                    f"{landmark_path}"
                )

            raw_features = np.load(landmark_path, allow_pickle=False)
            sequence, stats = prepare_sequence(
                raw_features,
                int(report_row["start_frame"]),
                int(report_row["end_frame"]),
                args.target_frames,
                args.max_short_gap,
            )
            X[sample_index] = sequence
            y[sample_index] = label_to_id[label]
            used_report_keys.add(key)

            metadata_rows.append(
                {
                    "sample_index": sample_index,
                    "label_id": label_to_id[label],
                    "label": label,
                    "video": report_row["video"],
                    "signer_id": report_row["signer_id"],
                    "repetition_id": report_row["repetition_id"],
                    "raw_frames": raw_features.shape[0],
                    "start_frame": int(report_row["start_frame"]),
                    "end_frame": int(report_row["end_frame"]),
                    "active_frames_used": stats["active_frames_used"],
                    "target_frames": args.target_frames,
                    "left_observed_rate": stats[
                        "resampled_left_observed_rate"
                    ],
                    "right_observed_rate": stats[
                        "resampled_right_observed_rate"
                    ],
                    "left_short_gap_imputed_rate": stats[
                        "resampled_left_imputed_rate"
                    ],
                    "right_short_gap_imputed_rate": stats[
                        "resampled_right_imputed_rate"
                    ],
                    "landmark_file": str(
                        landmark_path.relative_to(args.landmarks_root)
                    ),
                }
            )

            if (sample_index + 1) % 100 == 0 or sample_index + 1 == sample_count:
                print(f"Prepared {sample_index + 1}/{sample_count} sequences...")

        unused_rows = set(report_records).difference(used_report_keys)
        if unused_rows:
            raise ValueError(
                f"{len(unused_rows)} active-report rows have no landmark file. "
                f"Examples: {sorted(unused_rows)[:5]}"
            )

        X.flush()
        del X
        os.replace(temporary_x_path, final_x_path)

    except Exception:
        try:
            X.flush()
            del X
        except Exception:
            pass
        if temporary_x_path.exists():
            temporary_x_path.unlink()
        raise

    np.save(args.output_dir / "y.npy", y)
    write_csv(args.output_dir / "sequence_metadata.csv", metadata_rows)

    label_map = {
        "label_to_id": label_to_id,
        "id_to_label": {
            str(index): label for label, index in label_to_id.items()
        },
    }
    with (args.output_dir / "label_map.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(label_map, file, ensure_ascii=False, indent=2)

    class_counts = Counter(row["label"] for row in metadata_rows)
    config = {
        "samples": sample_count,
        "classes": len(labels),
        "target_frames": args.target_frames,
        "features_per_frame": TARGET_FEATURES,
        "input_shape": [
            sample_count,
            args.target_frames,
            TARGET_FEATURES,
        ],
        "source_feature_count": EXPECTED_RAW_FEATURES,
        "active_segment": (
            "inclusive start_frame:end_frame from active_length_report.csv"
        ),
        "temporal_resampling": (
            "linear interpolation over the complete active segment"
        ),
        "short_hand_gap_interpolation_frames": args.max_short_gap,
        "normalization": {
            "global_xy": "shoulder midpoint and shoulder-width scale",
            "pose_z": "shoulder midpoint and shoulder-width scale",
            "face_z": "nose-relative and inter-eye-width scale",
            "global_hand_z": (
                "MediaPipe wrist-relative Z scaled by shoulder width"
            ),
            "local_hands": (
                "wrist-relative and wrist-to-middle-MCP scale"
            ),
        },
        "selected_pose_indices": list(UPPER_BODY_POSE_INDICES),
        "selected_face_indices": list(FACE_INDICES),
        "feature_blocks_end_exclusive": {
            "upper_pose_global_xyz": [0, 42],
            "left_hand_global_xyz": [42, 105],
            "right_hand_global_xyz": [105, 168],
            "face_lips_eyes_eyebrows_xyz": [168, 444],
            "left_hand_local_xyz": [444, 507],
            "right_hand_local_xyz": [507, 570],
            "left_hand_observed": [570, 571],
            "right_hand_observed": [571, 572],
        },
        "future_training_features": (
            "velocity is derived from consecutive prepared frames "
            "during training"
        ),
        "class_counts": dict(
            sorted(
                class_counts.items(),
                key=lambda item: item[0].casefold(),
            )
        ),
    }
    with (args.output_dir / "preprocessing_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Model sequence preparation completed.")
    print(f"Samples: {sample_count}")
    print(f"Classes: {len(labels)}")
    print(
        f"X shape: "
        f"({sample_count}, {args.target_frames}, {TARGET_FEATURES})"
    )
    print(f"y shape: {y.shape}")
    print("Finite values: True")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
