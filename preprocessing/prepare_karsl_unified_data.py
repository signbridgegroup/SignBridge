"""Prepare KArSL-502 metadata and extend the existing SignBridge vocabulary.

This step is non-destructive: it does not copy or modify the large feature
array. It creates a compact training manifest, a KArSL-to-token mapping, and a
new vocabulary that reuses matching Isharah tokens and appends genuinely new
KArSL tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(r"C:\SignBridge_Project")
DEFAULT_KARSL = PROJECT / "data" / "processed" / "karsl_shared198"
DEFAULT_EXISTING = PROJECT / "data" / "processed" / "unified_sign_data"
DEFAULT_OUTPUT = PROJECT / "data" / "processed" / "unified_sign_data_karsl"
SPECIAL_TOKENS = ("<blank>", "<unk>")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend SignBridge unified vocabulary with KArSL-502."
    )
    parser.add_argument("--karsl", type=Path, default=DEFAULT_KARSL)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_label(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[_\-/،,;؛:()\[\]{}]+", " ", text)
    text = re.sub(r"[^\w\s&+]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def atomic_token(label: str, sign_id: str) -> str:
    cleaned = unicodedata.normalize("NFKC", str(label or "")).strip()
    if not cleaned or cleaned.upper().startswith("KARSL_"):
        return f"KARSL_{str(sign_id).zfill(4)}"
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[/،,;؛:()\[\]{}]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or f"KARSL_{str(sign_id).zfill(4)}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def unique_existing_match(
    label: str,
    normalized_existing: dict[str, list[str]],
) -> str | None:
    normalized = normalize_label(label)
    matches = normalized_existing.get(normalized, [])
    if not matches:
        return None
    if label in matches:
        return label
    return matches[0]


def main() -> int:
    args = arguments()
    karsl = args.karsl.resolve()
    existing = args.existing.resolve()
    output = args.output.resolve()

    required = [
        karsl / "features_flat.npy",
        karsl / "lengths.npy",
        karsl / "offsets.npy",
        karsl / "sample_ids.npy",
        karsl / "samples.csv",
        karsl / "preparation_summary.json",
        existing / "unified_vocabulary.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    existing_vocabulary = list(load_json(existing / "unified_vocabulary.json"))
    if tuple(existing_vocabulary[:2]) != SPECIAL_TOKENS:
        raise ValueError(
            f"Existing vocabulary must start with {SPECIAL_TOKENS}, "
            f"got {existing_vocabulary[:2]}"
        )

    rows = read_csv(karsl / "samples.csv")
    lengths = np.load(karsl / "lengths.npy", allow_pickle=False)
    offsets = np.load(karsl / "offsets.npy", allow_pickle=False)
    sample_ids = np.load(karsl / "sample_ids.npy", allow_pickle=False)
    if not (len(rows) == len(lengths) == len(offsets) == len(sample_ids)):
        raise ValueError("KArSL metadata arrays have different sample counts")
    for index, row in enumerate(rows):
        if str(row["sample_id"]) != str(sample_ids[index]):
            raise ValueError(f"Sample alignment mismatch at row {index}")
        if int(row["frames"]) != int(lengths[index]):
            raise ValueError(f"Frame-length mismatch at row {index}")

    normalized_existing: dict[str, list[str]] = defaultdict(list)
    for token in existing_vocabulary[2:]:
        normalized = normalize_label(token)
        if normalized:
            normalized_existing[normalized].append(token)

    sign_metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        sign_id = str(row["sign_id"]).zfill(4)
        sign_metadata.setdefault(
            sign_id,
            {
                "label_ar": str(row.get("label_ar", "")).strip(),
                "label_en": str(row.get("label_en", "")).strip(),
            },
        )

    vocabulary = list(existing_vocabulary)
    vocabulary_set = set(vocabulary)
    mappings: dict[str, dict[str, object]] = {}
    overlap_count = 0

    for sign_id in sorted(sign_metadata):
        label_ar = sign_metadata[sign_id]["label_ar"]
        label_en = sign_metadata[sign_id]["label_en"]
        match = unique_existing_match(label_ar, normalized_existing)
        if match is not None:
            token = match
            status = "existing_token"
            overlap_count += 1
        else:
            base = atomic_token(label_ar, sign_id)
            token = base
            suffix = 2
            # Reuse identical new semantics, but never overwrite a differently
            # normalized existing token that happens to share the same spelling.
            while token in vocabulary_set and normalize_label(token) != normalize_label(label_ar):
                token = f"{base}_KArSL{suffix}"
                suffix += 1
            if token not in vocabulary_set:
                vocabulary.append(token)
                vocabulary_set.add(token)
            status = "new_token"

        mappings[sign_id] = {
            "sign_id": sign_id,
            "label_ar": label_ar,
            "label_en": label_en,
            "unified_token": token,
            "unified_token_id": vocabulary.index(token),
            "status": status,
        }

    prepared_rows: list[dict[str, object]] = []
    excluded_train = 0
    for index, row in enumerate(rows):
        sign_id = str(row["sign_id"]).zfill(4)
        mapping = mappings[sign_id]
        left = float(row["left_hand_rate"])
        right = float(row["right_hand_rate"])
        pose = float(row["pose_rate"])
        face = float(row["face_rate"])
        best_hand = max(left, right)
        quality_score = max(0.05, min(1.0, 0.65 * best_hand + 0.25 * pose + 0.10 * face))

        # Only impossible training examples are excluded. Difficult validation
        # and signer-independent test samples remain untouched for honest scores.
        train_eligible = not (
            row["split"] == "train"
            and (best_hand == 0.0 or pose < 0.5 or int(row["frames"]) < 5)
        )
        excluded_train += int(not train_eligible)

        start = int(offsets[index])
        end = start + int(lengths[index])
        prepared_rows.append(
            {
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "split": row["split"],
                "signer": row["signer"],
                "source_split": row["source_split"],
                "sign_id": sign_id,
                "label_ar": mapping["label_ar"],
                "label_en": mapping["label_en"],
                "unified_token": mapping["unified_token"],
                "unified_token_id": mapping["unified_token_id"],
                "start": start,
                "end": end,
                "frames": int(lengths[index]),
                "face_rate": round(face, 6),
                "pose_rate": round(pose, 6),
                "left_hand_rate": round(left, 6),
                "right_hand_rate": round(right, 6),
                "best_hand_rate": round(best_hand, 6),
                "quality_score": round(quality_score, 6),
                "train_eligible": train_eligible,
            }
        )

    # Every class must remain trainable after the strict impossible-sample filter.
    eligible_signs = {
        row["sign_id"]
        for row in prepared_rows
        if row["split"] == "train" and row["train_eligible"]
    }
    missing_train_signs = sorted(set(mappings) - eligible_signs)
    if missing_train_signs:
        raise ValueError(
            "No eligible training videos for sign IDs: " + ", ".join(missing_train_signs)
        )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "karsl_samples.csv", prepared_rows)
    write_csv(output / "karsl_sign_mapping.csv", list(mappings.values()))
    with (output / "unified_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, ensure_ascii=False, indent=2)

    summary = {
        "karsl_samples": len(prepared_rows),
        "karsl_sign_ids": len(mappings),
        "existing_real_tokens_before_karsl": len(existing_vocabulary) - 2,
        "karsl_sign_ids_mapped_to_existing_tokens": overlap_count,
        "new_unique_karsl_tokens": len(vocabulary) - len(existing_vocabulary),
        "unified_real_tokens": len(vocabulary) - 2,
        "vocabulary_including_special_tokens": len(vocabulary),
        "split_counts": dict(Counter(row["split"] for row in prepared_rows)),
        "excluded_impossible_training_samples": excluded_train,
        "validation_and_test_quality_filtering": False,
        "karsl_feature_file": str(karsl / "features_flat.npy"),
        "karsl_feature_count": 198,
        "checks": {
            "metadata_arrays_aligned": True,
            "all_502_sign_ids_present": len(mappings) == 502,
            "all_sign_ids_have_eligible_training_data": not missing_train_signs,
            "source_features_unchanged": True,
        },
    }
    with (output / "preparation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 62)
    print("KArSL unified data preparation completed")
    print("KArSL samples:", f"{len(prepared_rows):,}")
    print("KArSL sign IDs:", len(mappings))
    print("Mapped to existing tokens:", overlap_count)
    print("New unique KArSL tokens:", len(vocabulary) - len(existing_vocabulary))
    print("Unified real tokens:", len(vocabulary) - 2)
    print("Vocabulary including special tokens:", len(vocabulary))
    print("Excluded impossible training samples:", excluded_train)
    print("Split counts:", dict(Counter(row["split"] for row in prepared_rows)))
    print("Output:", output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
