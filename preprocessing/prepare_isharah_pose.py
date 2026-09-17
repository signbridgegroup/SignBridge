"""Prepare the Isharah-1000 pose data for SignBridge.

The official Isharah pose file is one large pickle containing all 30,000
Isharah-2000 samples as variable-length arrays.  This script keeps only the
sample IDs referenced by the Isharah-1000 signer-independent annotations and
writes a compact, training-friendly representation:

* keypoints_flat.npy  - all frames concatenated, float32, memory-mapped
* offsets.npy         - start/end offsets for each sample
* lengths.npy         - frame count for each sample
* sample_ids.npy      - sample IDs in annotation order
* samples.csv         - ID, split, gloss, Arabic text, length, offsets
* preparation_summary.json

No temporal resampling is performed here.  Keeping the original sequence
lengths is important for continuous sign-language recognition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(
    r"C:\SignBridge_Project\data\external\isharah\pose_data_isharah2000_hands_lips_body.pkl"
)
DEFAULT_ANNOTATIONS = Path(
    r"C:\SignBridge_Project\data\external\isharah\annotations\si_1000"
)
DEFAULT_OUTPUT = Path(
    r"C:\SignBridge_Project\data\processed\isharah1000_pose"
)

EXPECTED_JOINTS = 86
EXPECTED_DIMS = 2


@dataclass(frozen=True)
class Annotation:
    sample_id: str
    split: str
    gloss: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the Isharah-1000 SI subset from the full pose PKL."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a previous completed output directory.",
    )
    return parser.parse_args()


def human_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def available_memory_bytes() -> int | None:
    """Return currently available physical memory without extra packages."""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)
        except Exception:
            return None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, OSError, ValueError):
        return None


def read_annotations(annotation_dir: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    seen_ids: set[str] = set()

    for split in ("train", "dev", "test"):
        path = annotation_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            required = {"id", "gloss", "text"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Unexpected columns in {path}: {reader.fieldnames}; "
                    f"expected {sorted(required)}"
                )

            for row_number, row in enumerate(reader, start=2):
                sample_id = row["id"].strip()
                if not sample_id:
                    raise ValueError(f"Blank sample ID in {path}, line {row_number}")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample ID across splits: {sample_id}")
                seen_ids.add(sample_id)
                annotations.append(
                    Annotation(
                        sample_id=sample_id,
                        split=split,
                        gloss=row["gloss"].strip(),
                        text=row["text"].strip(),
                    )
                )

    return annotations


def validate_source_entry(sample_id: str, sample: object) -> np.ndarray:
    if not isinstance(sample, dict) or "keypoints" not in sample:
        raise ValueError(f"Sample {sample_id!r} does not contain a keypoints field")

    points = sample["keypoints"]
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Sample {sample_id!r} keypoints are not a NumPy array")
    if points.ndim != 3 or points.shape[1:] != (EXPECTED_JOINTS, EXPECTED_DIMS):
        raise ValueError(
            f"Sample {sample_id!r} has shape {points.shape}; "
            f"expected (T, {EXPECTED_JOINTS}, {EXPECTED_DIMS})"
        )
    if points.shape[0] == 0:
        raise ValueError(f"Sample {sample_id!r} has zero frames")
    if not np.isfinite(points).all():
        raise ValueError(f"Sample {sample_id!r} contains NaN or infinite values")
    return points


def make_output_directory(output_dir: Path, overwrite: bool) -> Path:
    partial_dir = output_dir.with_name(output_dir.name + ".partial")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}\n"
                "Use --overwrite only if you intentionally want to replace it."
            )
        shutil.rmtree(output_dir)

    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)
    return partial_dir


def write_metadata(
    partial_dir: Path,
    annotations: list[Annotation],
    lengths: np.ndarray,
    offsets: np.ndarray,
) -> None:
    sample_ids = np.asarray([item.sample_id for item in annotations])
    np.save(partial_dir / "sample_ids.npy", sample_ids, allow_pickle=False)
    np.save(partial_dir / "lengths.npy", lengths, allow_pickle=False)
    np.save(partial_dir / "offsets.npy", offsets, allow_pickle=False)

    with (partial_dir / "samples.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "split", "gloss", "text", "frames", "start", "end"]
        )
        for index, item in enumerate(annotations):
            writer.writerow(
                [
                    item.sample_id,
                    item.split,
                    item.gloss,
                    item.text,
                    int(lengths[index]),
                    int(offsets[index]),
                    int(offsets[index + 1]),
                ]
            )


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    annotation_dir = args.annotations.resolve()
    output_dir = args.output.resolve()

    print("Preparing Isharah-1000 pose data")
    print("Source      :", source)
    print("Annotations :", annotation_dir)
    print("Output      :", output_dir)

    if not source.is_file():
        raise FileNotFoundError(f"Pose PKL not found: {source}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    source_size = source.stat().st_size
    print("Source size :", human_gib(source_size))

    memory = available_memory_bytes()
    if memory is not None:
        print("Available physical memory before loading:", human_gib(memory))
        if memory < 7 * 1024**3:
            print(
                "WARNING: Less than 7 GiB of physical memory is currently free.\n"
                "Close browsers and other large applications before continuing.\n"
                "Windows may use virtual memory, so this step can become slow."
            )

    annotations = read_annotations(annotation_dir)
    split_counts = {
        split: sum(item.split == split for item in annotations)
        for split in ("train", "dev", "test")
    }
    selected_ids = {item.sample_id for item in annotations}
    print("Requested samples:", len(annotations), split_counts)

    print("\nLoading the official PKL. This is the memory-heavy step...")
    started = time.perf_counter()
    try:
        with source.open("rb") as handle:
            pose_dict = pickle.load(handle)
    except MemoryError as exc:
        raise MemoryError(
            "The PKL could not fit in memory. Close other applications, restart "
            "Windows if needed, and run the script again. The source file was not changed."
        ) from exc

    print(
        f"Loaded {len(pose_dict):,} source samples in "
        f"{time.perf_counter() - started:.1f} seconds."
    )
    if not isinstance(pose_dict, dict):
        raise TypeError(f"Expected a dictionary in the PKL, got {type(pose_dict)!r}")

    missing_ids = sorted(selected_ids.difference(pose_dict))
    if missing_ids:
        preview = ", ".join(missing_ids[:20])
        raise KeyError(
            f"{len(missing_ids)} annotated IDs are missing from the pose file. "
            f"First IDs: {preview}"
        )

    # Discard the 15,000 Isharah-2000 samples that are not part of the chosen
    # Isharah-1000 split before creating the output arrays.
    for sample_id in list(pose_dict):
        if sample_id not in selected_ids:
            del pose_dict[sample_id]
    gc.collect()
    print("Samples retained after filtering:", len(pose_dict))

    lengths = np.empty(len(annotations), dtype=np.int32)
    for index, item in enumerate(annotations):
        lengths[index] = validate_source_entry(
            item.sample_id, pose_dict[item.sample_id]
        ).shape[0]

    offsets = np.empty(len(annotations) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
    total_frames = int(offsets[-1])

    estimated_bytes = total_frames * EXPECTED_JOINTS * EXPECTED_DIMS * 4
    free_disk = shutil.disk_usage(output_dir.parent).free
    required_disk = int(estimated_bytes * 1.10) + 100 * 1024**2
    print("Total selected frames:", f"{total_frames:,}")
    print("Estimated float32 output:", human_gib(estimated_bytes))
    print("Free disk space:", human_gib(free_disk))
    if free_disk < required_disk:
        raise OSError(
            f"Insufficient disk space. Need about {human_gib(required_disk)}, "
            f"but only {human_gib(free_disk)} is free."
        )

    partial_dir = make_output_directory(output_dir, args.overwrite)
    try:
        flat_path = partial_dir / "keypoints_flat.npy"
        flat = np.lib.format.open_memmap(
            flat_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, EXPECTED_JOINTS, EXPECTED_DIMS),
        )

        for index, item in enumerate(annotations):
            points = validate_source_entry(
                item.sample_id, pose_dict[item.sample_id]
            )
            start = int(offsets[index])
            end = int(offsets[index + 1])
            flat[start:end] = points.astype(np.float32, copy=False)
            del pose_dict[item.sample_id]

            if (index + 1) % 500 == 0 or index + 1 == len(annotations):
                flat.flush()
                gc.collect()
                print(f"Written {index + 1:,}/{len(annotations):,} samples")

        flat.flush()
        del flat
        del pose_dict
        gc.collect()

        write_metadata(partial_dir, annotations, lengths, offsets)

        gloss_vocabulary = sorted(
            {token for item in annotations for token in item.gloss.split()}
        )
        with (partial_dir / "gloss_vocabulary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(gloss_vocabulary, handle, ensure_ascii=False, indent=2)

        summary = {
            "dataset": "Isharah-1000",
            "evaluation_protocol": "signer-independent",
            "samples": len(annotations),
            "split_counts": split_counts,
            "total_frames": total_frames,
            "minimum_frames": int(lengths.min()),
            "maximum_frames": int(lengths.max()),
            "mean_frames": float(lengths.mean()),
            "joints": EXPECTED_JOINTS,
            "coordinates_per_joint": EXPECTED_DIMS,
            "output_dtype": "float32",
            "layout": {
                "right_hand": [0, 21],
                "left_hand": [21, 42],
                "lips": [42, 61],
                "body": [61, 86],
            },
            "source_file": str(source),
            "annotation_directory": str(annotation_dir),
        }
        with (partial_dir / "preparation_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        partial_dir.replace(output_dir)
    except Exception:
        print(
            "\nPreparation failed. Partial output was kept here for diagnosis:",
            partial_dir,
            file=sys.stderr,
        )
        raise

    print("\nIsharah-1000 pose preparation completed.")
    print("Samples     :", len(annotations))
    print("Total frames:", f"{total_frames:,}")
    print("Frame range :", int(lengths.min()), "to", int(lengths.max()))
    print("Mean frames :", f"{lengths.mean():.1f}")
    print("Output      :", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
