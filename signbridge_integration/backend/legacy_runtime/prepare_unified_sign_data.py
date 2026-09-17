"""Prepare a shared Isharah + Jordanian IT vocabulary and feature source.

This script does not train a model and does not modify source data.  It converts
the existing 572-D Jordanian IT sequences into the same 198-D feature layout
used by Isharah, restores an approximate active duration for each IT sample,
preserves the signer-independent splits, and creates one unified vocabulary.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_ISHARAH_DATA = Path(
    r"C:\SignBridge_Project\data\processed\isharah1000_shared198"
)
DEFAULT_GLOSS_LIBRARY = Path(
    r"C:\SignBridge_Project\data\processed\isharah_gloss_clip_library"
)
DEFAULT_CTC_VOCABULARY = Path(
    r"C:\SignBridge_Project\data\processed\isharah_ctc_training\ctc_vocabulary.json"
)
DEFAULT_IT_DATA = Path(
    r"C:\SignBridge_Project\data\processed\model_data"
)
DEFAULT_IT_SPLITS = DEFAULT_IT_DATA / "splits_signer_independent"
DEFAULT_OUTPUT = Path(
    r"C:\SignBridge_Project\data\processed\unified_sign_data"
)

SOURCE_IT_FEATURES = 572
SHARED_FEATURES = 198
SPECIAL_TOKENS = ("<blank>", "<unk>")
MIN_IT_FRAMES = 12
MAX_IT_FRAMES = 200


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare unified Isharah and Jordanian IT sign sources."
    )
    parser.add_argument("--isharah-data", type=Path, default=DEFAULT_ISHARAH_DATA)
    parser.add_argument("--gloss-library", type=Path, default=DEFAULT_GLOSS_LIBRARY)
    parser.add_argument("--ctc-vocabulary", type=Path, default=DEFAULT_CTC_VOCABULARY)
    parser.add_argument("--it-data", type=Path, default=DEFAULT_IT_DATA)
    parser.add_argument("--it-splits", type=Path, default=DEFAULT_IT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def xyz_to_xy(values: np.ndarray, landmarks: int) -> np.ndarray:
    return values.reshape(len(values), landmarks, 3)[:, :, :2].reshape(len(values), -1)


def jordanian_572_to_shared_198(sequence: np.ndarray) -> np.ndarray:
    """Select exactly the feature blocks shared with Isharah."""
    if sequence.ndim != 2 or sequence.shape[1] != SOURCE_IT_FEATURES:
        raise ValueError(
            f"Expected Jordanian sequence (*, {SOURCE_IT_FEATURES}), got {sequence.shape}"
        )

    pose_xy = xyz_to_xy(sequence[:, 0:42], 14)
    left_global_xy = xyz_to_xy(sequence[:, 42:105], 21)
    right_global_xy = xyz_to_xy(sequence[:, 105:168], 21)
    left_local_xy = xyz_to_xy(sequence[:, 444:507], 21)
    right_local_xy = xyz_to_xy(sequence[:, 507:570], 21)
    observed_masks = sequence[:, 570:572]

    shared = np.concatenate(
        [
            pose_xy,
            left_global_xy,
            right_global_xy,
            left_local_xy,
            right_local_xy,
            observed_masks,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if shared.shape[1] != SHARED_FEATURES:
        raise AssertionError(f"Expected {SHARED_FEATURES} features, got {shared.shape[1]}")
    return shared


def resample_shared(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    """Linearly resample coordinates and nearest-resample observed masks."""
    source_frames = len(sequence)
    if source_frames == target_frames:
        output = sequence.astype(np.float32, copy=True)
    elif source_frames == 1:
        output = np.repeat(sequence, target_frames, axis=0).astype(np.float32)
    else:
        source_time = np.linspace(0.0, 1.0, source_frames)
        target_time = np.linspace(0.0, 1.0, target_frames)
        output = np.empty((target_frames, SHARED_FEATURES), dtype=np.float32)
        for feature_index in range(SHARED_FEATURES - 2):
            output[:, feature_index] = np.interp(
                target_time, source_time, sequence[:, feature_index]
            )
        nearest = np.rint(
            np.linspace(0, source_frames - 1, target_frames)
        ).astype(np.int64)
        output[:, -2:] = sequence[nearest, -2:]
    output[:, -2:] = (output[:, -2:] >= 0.5).astype(np.float32)
    return output


def load_split_assignments(split_dir: Path, sample_count: int) -> dict[int, str]:
    filenames = {
        "train": "train_indices.npy",
        "validation": "validation_indices.npy",
        "test": "test_indices.npy",
    }
    assignments: dict[int, str] = {}
    for split, filename in filenames.items():
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        indices = np.load(path, allow_pickle=False).astype(np.int64)
        for raw_index in indices.tolist():
            index = int(raw_index)
            if index < 0 or index >= sample_count:
                raise ValueError(f"Out-of-range {split} index: {index}")
            if index in assignments:
                raise ValueError(f"Sample {index} occurs in multiple splits")
            assignments[index] = split
    if len(assignments) != sample_count:
        missing = sorted(set(range(sample_count)) - set(assignments))
        raise ValueError(
            f"Split files assign {len(assignments)}/{sample_count} samples. "
            f"Missing examples: {missing[:10]}"
        )
    return assignments


def main() -> int:
    args = arguments()
    started = time.perf_counter()
    isharah_data = args.isharah_data.resolve()
    gloss_library = args.gloss_library.resolve()
    ctc_vocabulary_path = args.ctc_vocabulary.resolve()
    it_data = args.it_data.resolve()
    it_splits = args.it_splits.resolve()
    output_dir = args.output.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")

    required = [
        isharah_data / "features_flat.npy",
        isharah_data / "samples.csv",
        gloss_library / "features_flat.npy",
        gloss_library / "clips.csv",
        gloss_library / "gloss_summary.csv",
        ctc_vocabulary_path,
        it_data / "X.npy",
        it_data / "y.npy",
        it_data / "label_map.json",
        it_data / "sequence_metadata.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --overwrite intentionally."
            )
        shutil.rmtree(output_dir)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    # Validate the complete extracted Isharah vocabulary.
    gloss_summary_rows = read_csv(gloss_library / "gloss_summary.csv")
    extracted_glosses = {row["gloss"] for row in gloss_summary_rows}
    if any(int(row["selected_clips"]) == 0 for row in gloss_summary_rows):
        raise ValueError("At least one Isharah gloss has no extracted clip")

    id_to_token = load_json(ctc_vocabulary_path)
    if id_to_token[:2] != list(SPECIAL_TOKENS):
        raise ValueError(
            f"Expected vocabulary to begin with {SPECIAL_TOKENS}, got {id_to_token[:2]}"
        )
    isharah_tokens = list(id_to_token[2:])
    if set(isharah_tokens) != extracted_glosses:
        missing = sorted(set(isharah_tokens) - extracted_glosses)
        extra = sorted(extracted_glosses - set(isharah_tokens))
        raise ValueError(
            f"Gloss-library mismatch. Missing={missing[:10]}, extra={extra[:10]}"
        )

    label_map = load_json(it_data / "label_map.json")
    raw_id_to_label = label_map.get("id_to_label")
    if not isinstance(raw_id_to_label, dict):
        raise ValueError("label_map.json is missing id_to_label")
    it_labels = [
        raw_id_to_label[str(index)] for index in range(len(raw_id_to_label))
    ]

    new_it_tokens = sorted(
        set(it_labels) - set(id_to_token), key=lambda value: value.casefold()
    )
    unified_vocabulary = [*id_to_token, *new_it_tokens]
    token_to_id = {token: index for index, token in enumerate(unified_vocabulary)}

    X = np.load(it_data / "X.npy", mmap_mode="r")
    y = np.load(it_data / "y.npy", allow_pickle=False).astype(np.int64)
    metadata_rows = read_csv(it_data / "sequence_metadata.csv")
    if X.ndim != 3 or X.shape[1:] != (200, SOURCE_IT_FEATURES):
        raise ValueError(f"Unexpected Jordanian X shape: {X.shape}")
    if len(X) != len(y) or len(X) != len(metadata_rows):
        raise ValueError("X, y, and sequence_metadata.csv do not align")
    for expected_index, row in enumerate(metadata_rows):
        if int(row["sample_index"]) != expected_index:
            raise ValueError(
                f"Metadata row {expected_index} has sample_index={row['sample_index']}"
            )

    split_by_index = load_split_assignments(it_splits, len(X))
    restored_lengths = np.asarray(
        [
            min(
                MAX_IT_FRAMES,
                max(MIN_IT_FRAMES, int(row["active_frames_used"])),
            )
            for row in metadata_rows
        ],
        dtype=np.int64,
    )
    offsets = np.zeros(len(X) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(restored_lengths)

    output_features = np.lib.format.open_memmap(
        partial_dir / "it_features_flat.npy",
        mode="w+",
        dtype=np.float32,
        shape=(int(offsets[-1]), SHARED_FEATURES),
    )
    prepared_rows: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    signer_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }

    print("Preparing unified Isharah + Jordanian IT sources")
    print("Isharah glosses:", f"{len(isharah_tokens):,}")
    print("Jordanian IT labels:", f"{len(it_labels):,}")
    print("New IT tokens:", f"{len(new_it_tokens):,}")
    print("Unified vocabulary:", f"{len(unified_vocabulary) - 2:,}", "plus special tokens")
    print("Jordanian samples:", f"{len(X):,}")

    for sample_index in range(len(X)):
        label_id = int(y[sample_index])
        if label_id < 0 or label_id >= len(it_labels):
            raise ValueError(f"Invalid label ID at sample {sample_index}: {label_id}")
        label = it_labels[label_id]
        if metadata_rows[sample_index]["label"] != label:
            raise ValueError(
                f"Label mismatch at sample {sample_index}: "
                f"y={label!r}, metadata={metadata_rows[sample_index]['label']!r}"
            )

        shared = jordanian_572_to_shared_198(np.asarray(X[sample_index]))
        restored = resample_shared(shared, int(restored_lengths[sample_index]))
        start, end = int(offsets[sample_index]), int(offsets[sample_index + 1])
        output_features[start:end] = restored

        split = split_by_index[sample_index]
        signer = str(metadata_rows[sample_index]["signer_id"])
        split_counts[split] += 1
        signer_counts[split][signer] += 1
        prepared_rows.append(
            {
                "sample_index": sample_index,
                "split": split,
                "label": label,
                "original_label_id": label_id,
                "unified_token_id": token_to_id[label],
                "signer_id": signer,
                "video": metadata_rows[sample_index]["video"],
                "original_active_frames": int(
                    metadata_rows[sample_index]["active_frames_used"]
                ),
                "prepared_frames": int(restored_lengths[sample_index]),
                "start": start,
                "end": end,
            }
        )

        if (sample_index + 1) % 100 == 0 or sample_index + 1 == len(X):
            print(f"Converted {sample_index + 1:,}/{len(X):,} IT samples")

    output_features.flush()
    del output_features
    if not np.isfinite(
        np.load(partial_dir / "it_features_flat.npy", mmap_mode="r")
    ).all():
        raise ValueError("Prepared IT features contain NaN or infinite values")

    np.save(partial_dir / "it_offsets.npy", offsets)
    np.save(partial_dir / "it_lengths.npy", restored_lengths)
    np.save(
        partial_dir / "it_unified_token_ids.npy",
        np.asarray([token_to_id[it_labels[int(value)]] for value in y], dtype=np.int64),
    )
    np.save(partial_dir / "it_original_label_ids.npy", y)
    write_csv(partial_dir / "it_samples.csv", prepared_rows)

    with (partial_dir / "unified_vocabulary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(unified_vocabulary, handle, ensure_ascii=False, indent=2)
    with (partial_dir / "unified_token_to_id.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(token_to_id, handle, ensure_ascii=False, indent=2)

    overlap = sorted(set(isharah_tokens) & set(it_labels))
    summary = {
        "purpose": "shared sources for unified continuous sign recognition",
        "shared_features_per_frame": SHARED_FEATURES,
        "feature_layout": {
            "upper_pose_global_xy": [0, 28],
            "left_hand_global_xy": [28, 70],
            "right_hand_global_xy": [70, 112],
            "left_hand_local_xy": [112, 154],
            "right_hand_local_xy": [154, 196],
            "left_hand_observed": [196, 197],
            "right_hand_observed": [197, 198],
        },
        "isharah_glosses": len(isharah_tokens),
        "jordanian_it_labels": len(it_labels),
        "overlapping_tokens": overlap,
        "unified_real_tokens": len(unified_vocabulary) - 2,
        "vocabulary_including_special_tokens": len(unified_vocabulary),
        "jordanian_samples": len(X),
        "jordanian_total_prepared_frames": int(offsets[-1]),
        "jordanian_split_counts": dict(split_counts),
        "jordanian_signer_counts_by_split": {
            split: dict(counts) for split, counts in signer_counts.items()
        },
        "it_duration_restoration": {
            "source_fixed_frames": 200,
            "target": "active_frames_used from sequence_metadata.csv",
            "minimum_frames": MIN_IT_FRAMES,
            "maximum_frames": MAX_IT_FRAMES,
            "coordinate_resampling": "linear",
            "observed_mask_resampling": "nearest",
        },
        "test_policy": (
            "test samples are converted and indexed but remain assigned to the "
            "untouched signer-independent test split"
        ),
        "source_paths": {
            "isharah_data": str(isharah_data),
            "isharah_gloss_library": str(gloss_library),
            "jordanian_it_data": str(it_data),
            "jordanian_splits": str(it_splits),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (partial_dir / "preparation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    partial_dir.replace(output_dir)
    print("\nUnified source preparation completed.")
    print("Shared feature count:", SHARED_FEATURES)
    print("Unified real tokens:", summary["unified_real_tokens"])
    print("Vocabulary with special tokens:", len(unified_vocabulary))
    print("IT split counts:", dict(split_counts))
    print("Finite values: True")
    print("Elapsed seconds:", f"{summary['elapsed_seconds']:.1f}")
    print("Output:", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
