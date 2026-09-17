"""Create SignBridge-compatible 198-D features from Isharah-1000 pose data.

The output layout matches a 2-D subset of the existing SignBridge 572-D
representation:

  0:28    upper-pose global XY (MediaPipe pose indices 11..24)
  28:70   left-hand global XY
  70:112  right-hand global XY
  112:154 left-hand local XY
  154:196 right-hand local XY
  196     left-hand originally observed flag
  197     right-hand originally observed flag

Coordinates are normalized by shoulder midpoint and shoulder width. Local hand
coordinates are wrist-relative and scaled by wrist-to-middle-MCP distance.
Short missing hand-landmark gaps are linearly interpolated, while the observed
flags preserve whether each hand was detected in the original frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path(r"C:\SignBridge_Project\data\processed\isharah1000_pose")
DEFAULT_OUTPUT = Path(
    r"C:\SignBridge_Project\data\processed\isharah1000_shared198"
)

NUM_JOINTS = 86
FEATURES = 198
MAX_HAND_GAP = 5
CLIP_VALUE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize Isharah-1000 into SignBridge-compatible features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def point_valid(points: np.ndarray) -> np.ndarray:
    return np.any(points != 0.0, axis=-1) & np.isfinite(points).all(axis=-1)


def fill_short_gaps(
    points: np.ndarray, valid: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate bounded gaps of at most max_gap frames per landmark."""
    output = points.astype(np.float32, copy=True)
    filled_valid = valid.copy()
    frames, joints, _ = output.shape

    for joint in range(joints):
        observed = valid[:, joint]
        index = 0
        while index < frames:
            if observed[index]:
                index += 1
                continue
            start = index
            while index < frames and not observed[index]:
                index += 1
            end = index
            gap = end - start
            if start == 0 or end == frames or gap > max_gap:
                continue

            left = output[start - 1, joint]
            right = output[end, joint]
            for offset in range(gap):
                alpha = (offset + 1) / (gap + 1)
                output[start + offset, joint] = (1.0 - alpha) * left + alpha * right
            filled_valid[start:end, joint] = True

    return output, filled_valid


def interpolate_temporal(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill a mostly-observed temporal reference such as shoulder center/scale."""
    values = values.astype(np.float32, copy=True)
    frames = len(values)
    indices = np.arange(frames)
    good = np.flatnonzero(valid)
    if len(good) == 0:
        raise ValueError("No valid shoulder reference exists in this sequence")

    if values.ndim == 1:
        values[:] = np.interp(indices, good, values[good])
    else:
        for coordinate in range(values.shape[1]):
            values[:, coordinate] = np.interp(
                indices, good, values[good, coordinate]
            )
    return values


def normalize_global(
    points: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    normalized = (points - center[:, None, :]) / scale[:, None, None]
    normalized[~valid] = 0.0
    return np.clip(normalized, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)


def normalize_hand_local(
    hand: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    wrist_valid = valid[:, 0]
    middle_mcp_valid = valid[:, 9]
    reference_valid = wrist_valid & middle_mcp_valid

    # A valid continuous-sign sample may legitimately keep one hand outside
    # the frame.  The global stream and observed mask still carry that fact;
    # the unavailable local-hand stream should therefore be zeros, not an
    # exception that stops preparation of the complete dataset.
    if not np.any(wrist_valid) or not np.any(reference_valid):
        return np.zeros_like(hand, dtype=np.float32)

    wrist = hand[:, 0].copy()
    wrist = interpolate_temporal(wrist, wrist_valid)

    local_scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)
    scale_valid = reference_valid & np.isfinite(local_scale) & (local_scale > 1e-6)
    local_scale = interpolate_temporal(local_scale, scale_valid)
    local_scale = np.maximum(local_scale, 1e-6)

    local = (hand - wrist[:, None, :]) / local_scale[:, None, None]
    local[~valid] = 0.0
    return np.clip(local, -CLIP_VALUE, CLIP_VALUE).astype(np.float32)


def build_features(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1:] != (NUM_JOINTS, 2):
        raise ValueError(f"Unexpected Isharah sequence shape: {sequence.shape}")

    # Official old-PKL layout: RH 0:21, LH 21:42, lips 42:61, body 61:86.
    right_raw = np.asarray(sequence[:, 0:21], dtype=np.float32)
    left_raw = np.asarray(sequence[:, 21:42], dtype=np.float32)
    body = np.asarray(sequence[:, 61:86], dtype=np.float32)

    right_valid_original = point_valid(right_raw)
    left_valid_original = point_valid(left_raw)
    body_valid = point_valid(body)

    right_observed = (right_valid_original.mean(axis=1) >= 0.5).astype(np.float32)
    left_observed = (left_valid_original.mean(axis=1) >= 0.5).astype(np.float32)

    right, right_valid = fill_short_gaps(
        right_raw, right_valid_original, MAX_HAND_GAP
    )
    left, left_valid = fill_short_gaps(left_raw, left_valid_original, MAX_HAND_GAP)

    left_shoulder_valid = body_valid[:, 11]
    right_shoulder_valid = body_valid[:, 12]
    shoulders_valid = left_shoulder_valid & right_shoulder_valid

    center = (body[:, 11] + body[:, 12]) * 0.5
    shoulder_width = np.linalg.norm(body[:, 11] - body[:, 12], axis=1)
    scale_valid = shoulders_valid & np.isfinite(shoulder_width) & (shoulder_width > 1e-6)

    # Rare annotation sequences contain no usable shoulder pair.  Fall back
    # to the centre and spatial extent of all available body joints.  This
    # keeps the sample without fabricating hand observations.
    if not np.any(scale_valid):
        fallback_center = np.zeros((len(sequence), 2), dtype=np.float32)
        fallback_scale = np.ones(len(sequence), dtype=np.float32)
        fallback_valid = np.zeros(len(sequence), dtype=bool)

        for frame_index in range(len(sequence)):
            available = body[frame_index, body_valid[frame_index]]
            if len(available) == 0:
                continue
            fallback_center[frame_index] = np.median(available, axis=0)
            extent = np.ptp(available, axis=0)
            candidate_scale = float(np.linalg.norm(extent))
            if np.isfinite(candidate_scale) and candidate_scale > 1e-6:
                fallback_scale[frame_index] = candidate_scale
                fallback_valid[frame_index] = True

        if np.any(fallback_valid):
            center = interpolate_temporal(fallback_center, fallback_valid)
            shoulder_width = interpolate_temporal(fallback_scale, fallback_valid)
        else:
            # Completely absent body landmarks: missing points remain zero.
            center = fallback_center
            shoulder_width = fallback_scale
    else:
        center = interpolate_temporal(center, shoulders_valid)
        shoulder_width = interpolate_temporal(shoulder_width, scale_valid)

    shoulder_width = np.maximum(shoulder_width, 1e-6)

    upper_body = body[:, 11:25]
    upper_body_valid = body_valid[:, 11:25]

    pose_global = normalize_global(
        upper_body, upper_body_valid, center, shoulder_width
    ).reshape(len(sequence), -1)
    left_global = normalize_global(
        left, left_valid, center, shoulder_width
    ).reshape(len(sequence), -1)
    right_global = normalize_global(
        right, right_valid, center, shoulder_width
    ).reshape(len(sequence), -1)

    left_local = normalize_hand_local(left, left_valid).reshape(len(sequence), -1)
    right_local = normalize_hand_local(right, right_valid).reshape(len(sequence), -1)

    features = np.concatenate(
        [
            pose_global,
            left_global,
            right_global,
            left_local,
            right_local,
            left_observed[:, None],
            right_observed[:, None],
        ],
        axis=1,
    )
    if features.shape[1] != FEATURES:
        raise AssertionError(f"Expected {FEATURES} features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Normalized features contain NaN or infinite values")
    return features.astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")

    required = [
        "keypoints_flat.npy",
        "offsets.npy",
        "lengths.npy",
        "sample_ids.npy",
        "samples.csv",
        "gloss_vocabulary.json",
    ]
    for filename in required:
        if not (input_dir / filename).is_file():
            raise FileNotFoundError(f"Missing input file: {input_dir / filename}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_dir}. Use --overwrite intentionally."
            )
        shutil.rmtree(output_dir)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    source = np.load(input_dir / "keypoints_flat.npy", mmap_mode="r")
    offsets = np.load(input_dir / "offsets.npy")
    lengths = np.load(input_dir / "lengths.npy")
    sample_ids = np.load(input_dir / "sample_ids.npy", allow_pickle=False)

    if source.ndim != 3 or source.shape[1:] != (NUM_JOINTS, 2):
        raise ValueError(f"Unexpected source shape: {source.shape}")
    if len(offsets) != len(sample_ids) + 1 or len(lengths) != len(sample_ids):
        raise ValueError("Offsets, lengths, and sample IDs do not align")
    if int(offsets[-1]) != len(source):
        raise ValueError("Final offset does not equal the number of source frames")

    print("Preparing SignBridge-compatible Isharah features")
    print("Samples     :", f"{len(sample_ids):,}")
    print("Total frames:", f"{len(source):,}")
    print("Input       :", input_dir)
    print("Output      :", output_dir)

    output_path = partial_dir / "features_flat.npy"
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(source), FEATURES),
    )

    started = time.perf_counter()
    for index, sample_id in enumerate(sample_ids):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        output[start:end] = build_features(source[start:end])

        if (index + 1) % 250 == 0 or index + 1 == len(sample_ids):
            output.flush()
            print(f"Processed {index + 1:,}/{len(sample_ids):,} samples")

    output.flush()
    del output

    for filename in [
        "offsets.npy",
        "lengths.npy",
        "sample_ids.npy",
        "samples.csv",
        "gloss_vocabulary.json",
    ]:
        shutil.copy2(input_dir / filename, partial_dir / filename)

    with (input_dir / "samples.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    vocab_by_split = {}
    for split in ("train", "dev", "test"):
        vocab_by_split[split] = sorted(
            {
                token
                for row in rows
                if row["split"] == split
                for token in row["gloss"].split()
            }
        )
    train_vocab = set(vocab_by_split["train"])
    oov = {
        "dev": sorted(set(vocab_by_split["dev"]) - train_vocab),
        "test": sorted(set(vocab_by_split["test"]) - train_vocab),
    }

    config = {
        "dataset": "Isharah-1000",
        "features_per_frame": FEATURES,
        "dtype": "float32",
        "normalization": {
            "global_xy": "shoulder midpoint and shoulder-width scale",
            "local_hands": "wrist-relative and wrist-to-middle-MCP scale",
            "short_hand_gap_interpolation_frames": MAX_HAND_GAP,
            "coordinate_clip": [-CLIP_VALUE, CLIP_VALUE],
        },
        "feature_blocks_end_exclusive": {
            "upper_pose_global_xy": [0, 28],
            "left_hand_global_xy": [28, 70],
            "right_hand_global_xy": [70, 112],
            "left_hand_local_xy": [112, 154],
            "right_hand_local_xy": [154, 196],
            "left_hand_observed": [196, 197],
            "right_hand_observed": [197, 198],
        },
        "velocity": "derived during training from consecutive prepared frames",
        "train_vocabulary_size": len(train_vocab),
        "oov_glosses": oov,
        "processing_seconds": time.perf_counter() - started,
    }
    with (partial_dir / "feature_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    partial_dir.replace(output_dir)
    print("\nShared feature preparation completed.")
    print("Features per frame:", FEATURES)
    print("Dev OOV:", oov["dev"])
    print("Test OOV:", oov["test"])
    print("Elapsed seconds:", f"{config['processing_seconds']:.1f}")
    print("Output:", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
