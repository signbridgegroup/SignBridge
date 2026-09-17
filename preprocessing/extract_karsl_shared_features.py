"""Extract resumable SignBridge-compatible features from KArSL-502 videos.

The script scans all three KArSL signers, assigns signer-independent splits,
extracts MediaPipe Holistic landmarks, converts every frame directly to the
same 198-D representation used by the existing Isharah/SignBridge pipeline,
and writes atomic shards. Re-running the command resumes from saved shards.

Default split policy:
  train       = signer 01/02 official train folders
  validation  = signer 01/02 official test folders
  test        = signer 03 official train + test folders (untouched signer)

For a quick smoke test first:
  python extract_karsl_shared_features.py --limit 20 --workers 2

Then resume the complete extraction:
  python extract_karsl_shared_features.py --workers 2
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import multiprocessing as multiprocessing
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT = Path(r"C:\SignBridge_Project")
DEFAULT_VIDEO_ROOT = PROJECT / "data" / "external" / "karsl" / "videos"
DEFAULT_LABELS = PROJECT / "data" / "external" / "karsl" / "KARSL-502_Labels.xlsx"
DEFAULT_MODEL = PROJECT / "models" / "holistic_landmarker.task"
DEFAULT_OUTPUT = PROJECT / "data" / "processed" / "karsl_shared198"

FEATURE_COUNT = 198
SIGNERS = ("01", "02", "03")
SOURCE_SPLITS = ("train", "test")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

_LANDMARKER = None
_MP = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract resumable 198-D SignBridge features from KArSL-502."
    )
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Independent MediaPipe worker processes. Two is safe for ~9 GB free RAM.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=250,
        help="Successful videos saved atomically per resumable shard.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many remaining videos; 0 means all.",
    )
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=0.35,
        help="Minimum MediaPipe hand-landmark confidence.",
    )
    parser.add_argument(
        "--rebuild-final",
        action="store_true",
        help="Rebuild final flat arrays after all videos are complete.",
    )
    return parser.parse_args()


def normalized_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def choose_column(columns, accepted: set[str]):
    lookup = {normalized_column(column): column for column in columns}
    for candidate in accepted:
        if normalized_column(candidate) in lookup:
            return lookup[normalized_column(candidate)]
    return None


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is missing. Run: python -m pip install pandas openpyxl"
        ) from exc

    table = pd.read_excel(path)
    id_column = choose_column(table.columns, {"SignID", "Sign ID", "ID"})
    ar_column = choose_column(
        table.columns, {"Sign-Arabic", "Sign Arabic", "Arabic", "Arabic Sign"}
    )
    en_column = choose_column(
        table.columns, {"Sign-English", "Sign English", "English", "English Sign"}
    )
    if id_column is None:
        raise ValueError(f"No sign-ID column in {list(table.columns)}")

    labels: dict[str, dict[str, str]] = {}
    for _, row in table.iterrows():
        raw_id = row.get(id_column)
        try:
            sign_id = f"{int(float(raw_id)):04d}"
        except (TypeError, ValueError):
            continue
        labels[sign_id] = {
            "label_ar": clean_cell(row.get(ar_column)) if ar_column is not None else "",
            "label_en": clean_cell(row.get(en_column)) if en_column is not None else "",
        }
    return labels


def model_split(signer: str, source_split: str) -> str:
    if signer == "03":
        return "test"
    return "train" if source_split == "train" else "validation"


def scan_inventory(video_root: Path, labels: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for signer in SIGNERS:
        for source_split in SOURCE_SPLITS:
            split_root = video_root / signer / source_split
            if not split_root.is_dir():
                raise FileNotFoundError(f"Missing KArSL folder: {split_root}")
            for sign_folder in sorted(path for path in split_root.iterdir() if path.is_dir()):
                sign_id = sign_folder.name.zfill(4)
                label = labels.get(sign_id, {})
                label_ar = label.get("label_ar", "") or f"KARSL_{sign_id}"
                label_en = label.get("label_en", "")
                for path in sorted(sign_folder.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                        continue
                    relative = path.relative_to(video_root).as_posix()
                    rows.append(
                        {
                            "sample_id": "karsl_" + re.sub(r"[^A-Za-z0-9]+", "_", relative).strip("_"),
                            "relative_path": relative,
                            "absolute_path": str(path),
                            "signer": signer,
                            "source_split": source_split,
                            "model_split": model_split(signer, source_split),
                            "sign_id": sign_id,
                            "label_ar": label_ar,
                            "label_en": label_en,
                        }
                    )
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Generated sample IDs are not unique")
    return rows


def _close_worker() -> None:
    global _LANDMARKER
    if _LANDMARKER is not None:
        try:
            _LANDMARKER.close()
        finally:
            _LANDMARKER = None


def _worker_init(model_path: str, hand_confidence: float) -> None:
    global _LANDMARKER, _MP
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    import mediapipe as mp

    _MP = mp
    cv2.setNumThreads(1)
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=hand_confidence,
    )
    _LANDMARKER = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
    atexit.register(_close_worker)


def _xy(landmarks, expected: int) -> np.ndarray:
    output = np.zeros((expected, 2), dtype=np.float32)
    if landmarks:
        for index, landmark in enumerate(landmarks[:expected]):
            output[index] = (landmark.x, landmark.y)
    return output


def _extract_one(row: dict[str, str]) -> dict[str, Any]:
    try:
        from prepare_isharah_shared_features import build_features

        capture = cv2.VideoCapture(row["absolute_path"])
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the video")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[np.ndarray] = []
        face_frames = pose_frames = left_frames = right_frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = _MP.Image(image_format=_MP.ImageFormat.SRGB, data=rgb)
                result = _LANDMARKER.detect(image)

                # Match the official Isharah layout used by build_features:
                # RH 0:21, LH 21:42, lips 42:61, body 61:86.
                pose86 = np.zeros((86, 2), dtype=np.float32)
                pose86[0:21] = _xy(result.right_hand_landmarks, 21)
                pose86[21:42] = _xy(result.left_hand_landmarks, 21)
                pose86[61:86] = _xy(result.pose_landmarks, 25)
                frames.append(pose86)

                face_frames += int(bool(result.face_landmarks))
                pose_frames += int(bool(result.pose_landmarks))
                left_frames += int(bool(result.left_hand_landmarks))
                right_frames += int(bool(result.right_hand_landmarks))
        finally:
            capture.release()

        if not frames:
            raise RuntimeError("No frames were decoded")
        raw = np.stack(frames).astype(np.float32, copy=False)
        features = build_features(raw)
        if features.shape != (len(frames), FEATURE_COUNT):
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinity")

        count = len(frames)
        metadata = {
            **{key: value for key, value in row.items() if key != "absolute_path"},
            "frames": count,
            "fps": fps,
            "width": width,
            "height": height,
            "face_rate": face_frames / count,
            "pose_rate": pose_frames / count,
            "left_hand_rate": left_frames / count,
            "right_hand_rate": right_frames / count,
        }
        return {"ok": True, "metadata": metadata, "features": features}
    except Exception as exc:
        return {
            "ok": False,
            "sample_id": row["sample_id"],
            "relative_path": row["relative_path"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def existing_progress(shard_dir: Path) -> tuple[set[str], int]:
    processed: set[str] = set()
    max_index = -1
    for shard_path in sorted(shard_dir.glob("shard_*.npz")):
        try:
            index = int(shard_path.stem.split("_")[-1])
            max_index = max(max_index, index)
            with np.load(shard_path, allow_pickle=False) as shard:
                processed.update(str(value) for value in shard["sample_ids"].tolist())
        except Exception as exc:
            raise RuntimeError(f"Unreadable shard {shard_path}: {exc}") from exc
    return processed, max_index + 1


def save_shard(shard_dir: Path, shard_index: int, items: list[dict[str, Any]]) -> Path:
    features = np.concatenate([item["features"] for item in items], axis=0)
    lengths = np.asarray([len(item["features"]) for item in items], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    metadata = [item["metadata"] for item in items]

    payload: dict[str, np.ndarray] = {
        "features": features.astype(np.float32, copy=False),
        "lengths": lengths,
        "offsets": offsets,
    }
    string_fields = (
        "sample_id", "relative_path", "signer", "source_split", "model_split",
        "sign_id", "label_ar", "label_en",
    )
    numeric_fields = (
        "frames", "fps", "width", "height", "face_rate", "pose_rate",
        "left_hand_rate", "right_hand_rate",
    )
    for field in string_fields:
        key = "sample_ids" if field == "sample_id" else field
        payload[key] = np.asarray([str(row[field]) for row in metadata])
    for field in numeric_fields:
        dtype = np.int32 if field in {"frames", "width", "height"} else np.float32
        payload[field] = np.asarray([row[field] for row in metadata], dtype=dtype)

    destination = shard_dir / f"shard_{shard_index:05d}.npz"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def append_failures(path: Path, failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    with path.open("a", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


def write_progress(
    output: Path,
    inventory_count: int,
    processed_count: int,
    failed_this_run: int,
    split_counts: Counter,
) -> None:
    summary = {
        "inventory_videos": inventory_count,
        "successfully_sharded": processed_count,
        "remaining": inventory_count - processed_count,
        "failed_attempts_this_run": failed_this_run,
        "saved_split_counts": dict(split_counts),
        "complete": processed_count == inventory_count,
        "updated_unix_time": time.time(),
    }
    temporary = output / "progress_summary.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output / "progress_summary.json")


def read_all_shard_metadata(shard_paths: list[Path]):
    rows: list[dict[str, Any]] = []
    total_frames = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            count = len(shard["lengths"])
            total_frames += int(shard["lengths"].sum())
            for index in range(count):
                rows.append(
                    {
                        "sample_id": str(shard["sample_ids"][index]),
                        "relative_path": str(shard["relative_path"][index]),
                        "signer": str(shard["signer"][index]),
                        "source_split": str(shard["source_split"][index]),
                        "split": str(shard["model_split"][index]),
                        "sign_id": str(shard["sign_id"][index]),
                        "label_ar": str(shard["label_ar"][index]),
                        "label_en": str(shard["label_en"][index]),
                        "frames": int(shard["frames"][index]),
                        "fps": float(shard["fps"][index]),
                        "width": int(shard["width"][index]),
                        "height": int(shard["height"][index]),
                        "face_rate": float(shard["face_rate"][index]),
                        "pose_rate": float(shard["pose_rate"][index]),
                        "left_hand_rate": float(shard["left_hand_rate"][index]),
                        "right_hand_rate": float(shard["right_hand_rate"][index]),
                        "shard": shard_path.name,
                        "shard_row": index,
                    }
                )
    return rows, total_frames


def finalize(output: Path, expected_samples: int, force: bool) -> None:
    summary_path = output / "preparation_summary.json"
    if summary_path.is_file() and not force:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if int(summary.get("samples", -1)) == expected_samples:
            print("Final flat dataset already exists; skipping rebuild.")
            return

    shard_paths = sorted((output / "shards").glob("shard_*.npz"))
    rows, total_frames = read_all_shard_metadata(shard_paths)
    if len(rows) != expected_samples:
        raise RuntimeError(
            f"Cannot finalize: expected {expected_samples} samples, found {len(rows)}"
        )

    print("Combining shards into final memory-mapped arrays...")
    feature_temp = output / "features_flat.tmp.npy"
    features_out = np.lib.format.open_memmap(
        feature_temp, mode="w+", dtype=np.float32, shape=(total_frames, FEATURE_COUNT)
    )
    lengths = np.empty(expected_samples, dtype=np.int32)
    offsets = np.empty(expected_samples, dtype=np.int64)
    sample_ids = np.empty(expected_samples, dtype=f"<U{max(len(row['sample_id']) for row in rows)}")

    frame_cursor = sample_cursor = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            shard_features = shard["features"]
            count = len(shard["lengths"])
            next_frame = frame_cursor + len(shard_features)
            features_out[frame_cursor:next_frame] = shard_features
            lengths[sample_cursor:sample_cursor + count] = shard["lengths"]
            offsets[sample_cursor:sample_cursor + count] = (
                frame_cursor + shard["offsets"].astype(np.int64)
            )
            sample_ids[sample_cursor:sample_cursor + count] = shard["sample_ids"]
            frame_cursor = next_frame
            sample_cursor += count
    features_out.flush()
    del features_out
    os.replace(feature_temp, output / "features_flat.npy")
    np.save(output / "lengths.npy", lengths, allow_pickle=False)
    np.save(output / "offsets.npy", offsets, allow_pickle=False)
    np.save(output / "sample_ids.npy", sample_ids, allow_pickle=False)

    with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": expected_samples,
        "total_frames": total_frames,
        "features_per_frame": FEATURE_COUNT,
        "dtype": "float32",
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "signer_counts": dict(Counter(row["signer"] for row in rows)),
        "sign_ids": len({row["sign_id"] for row in rows}),
        "finite_values": True,
        "split_policy": {
            "train": "01/train + 02/train",
            "validation": "01/test + 02/test",
            "test": "03/train + 03/test (untouched signer)",
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = arguments()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    if not 0.0 <= args.hand_confidence <= 1.0:
        raise ValueError("--hand-confidence must be between 0 and 1")

    for required in (args.video_root, args.labels, args.model):
        if not required.exists():
            raise FileNotFoundError(required)

    labels = load_labels(args.labels)
    inventory = scan_inventory(args.video_root, labels)
    args.output.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    processed, next_shard_index = existing_progress(shard_dir)
    remaining = [row for row in inventory if row["sample_id"] not in processed]
    if args.limit > 0:
        remaining = remaining[: args.limit]

    print("KArSL resumable feature extraction")
    print("Inventory videos :", f"{len(inventory):,}")
    print("Already saved    :", f"{len(processed):,}")
    print("This run         :", f"{len(remaining):,}")
    print("Workers          :", args.workers)
    print("Output           :", args.output)

    pending: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failed_attempts = 0
    new_success = 0
    started = time.time()

    if remaining:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(str(args.model), float(args.hand_confidence)),
        ) as pool:
            iterator = pool.imap_unordered(_extract_one, remaining, chunksize=4)
            for attempted, result in enumerate(iterator, start=1):
                if result["ok"]:
                    pending.append(result)
                    new_success += 1
                    processed.add(result["metadata"]["sample_id"])
                else:
                    failures.append(result)
                    failed_attempts += 1

                if len(pending) >= args.shard_size:
                    path = save_shard(shard_dir, next_shard_index, pending)
                    print(
                        f"Saved {path.name}: {len(pending)} videos | "
                        f"run progress {attempted:,}/{len(remaining):,}"
                    )
                    next_shard_index += 1
                    pending.clear()
                    append_failures(args.output / "failures.jsonl", failures)
                    failures.clear()

                if attempted % 50 == 0:
                    rate = attempted / max(time.time() - started, 1e-6)
                    print(
                        f"Processed {attempted:,}/{len(remaining):,} | "
                        f"success {new_success:,} | {rate:.2f} videos/s"
                    )

        if pending:
            path = save_shard(shard_dir, next_shard_index, pending)
            print(f"Saved {path.name}: {len(pending)} videos")
        append_failures(args.output / "failures.jsonl", failures)

    # Re-read committed shards; only atomically saved data counts as progress.
    processed, _ = existing_progress(shard_dir)
    saved_split_counts = Counter(
        row["model_split"] for row in inventory if row["sample_id"] in processed
    )
    write_progress(
        args.output,
        len(inventory),
        len(processed),
        failed_attempts,
        saved_split_counts,
    )

    print("\nExtraction run completed.")
    print("Successfully saved:", f"{len(processed):,}/{len(inventory):,}")
    print("Remaining         :", f"{len(inventory) - len(processed):,}")
    if len(processed) == len(inventory):
        finalize(args.output, len(inventory), args.rebuild_final)
        print("Final dataset     :", args.output)
    else:
        print("Run the same command again to resume all remaining videos.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Completed shards are safe; run again to resume.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
