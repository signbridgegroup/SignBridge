"""SignBridge motion-library builder.

Commands:
  inventory          Build/refresh the full three-dataset token inventory
                      (does not generate any motion file).
  generate-token      Generate exactly one token's motion file.
  generate-dataset    Generate every pending_generation token in one dataset.
                      NOT run automatically by this tool for the full batch —
                      see the approved-tests-only note in the CLI help.
  resume              Continue a generate-dataset run, skipping tokens that
                      already have a valid ready entry.
  validate            Re-check every "ready" entry's file still exists and
                      its checksum still matches.
  rebuild-manifest     Rebuild the manifest from inventory + re-validated
                      ready entries (never silently drops a ready entry).
  coverage-report     Print counts by status/dataset.

Every write to motion_manifest.json is atomic (write to a temp file, then
os.replace). Every token is processed independently — one failure is logged
and the run continues; it never aborts the whole batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = TOOL_DIR.parent.parent
sys.path.insert(0, str(TOOL_DIR))

import extractor  # noqa: E402
import inventory  # noqa: E402
from label_quality import classify_label, label_report  # noqa: E402

GENERATOR_VERSION = "motion_library_builder-1.0"
PROJECT_ROOT = Path(r"C:\SignBridge_Project")
MANIFEST_PATH = INTEGRATION_ROOT / "data_manifests" / "motion_manifest.json"
MOTION_LIBRARY_ROOT = PROJECT_ROOT / "data" / "processed" / "avatar_motion_library"
BASELINE_MOTIONS_DIR = INTEGRATION_ROOT / "frontend" / "public" / "motions"
MEDIAPIPE_MODEL = PROJECT_ROOT / "models" / "holistic_landmarker.task"

BASELINE_FILES = {
    ("jordanian_it", "STACK"): "stack_07.motion.json",
    ("jordanian_it", "QUEUE"): "queue.motion.json",
    ("jordanian_it", "DATATYPE"): "datatype.motion.json",
    ("jordanian_it", "INITIALIZE"): "initialize.motion.json",
    ("jordanian_it", "RUN_TIME_ERROR"): "run_time_error.motion.json",
    ("jordanian_it", "BINARY_RELATIONSHIP"): "binary_relationship.motion.json",
}

# Motion files that exist under BASELINE_MOTIONS_DIR but must NOT become a
# second vocabulary entry for a token that already has a canonical entry.
# The dataset inventory is defined by the canonical dataset vocabularies
# (165 / 502 / 680 tokens) only — never by "however many motion files exist".
# stack.motion.json is the non-preferred earlier take of the same STACK
# concept as stack_07.motion.json; it is recorded as an alternate take on
# the STACK entry's quality_metadata, not as its own token.
KNOWN_ALTERNATE_TAKES = {
    ("jordanian_it", "STACK"): ["stack.motion.json"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        return {}
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {(e["dataset"], e["token"]): e for e in data.get("entries", [])}


def save_manifest(entries_by_key: dict[tuple, dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator_version": GENERATOR_VERSION,
        "entries": [entries_by_key[k] for k in sorted(entries_by_key)],
    }
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def register_baseline_motions(entries: dict[tuple, dict[str, Any]]) -> None:
    """Attach pre-existing approved motion files to their canonical
    vocabulary entry. This function must never create a manifest entry for
    a token that isn't part of the 165/502/680 canonical vocabularies —
    see KNOWN_ALTERNATE_TAKES for how a second file for the same concept
    (stack.motion.json) is recorded without inflating the token count.
    """
    for (dataset, token), filename in BASELINE_FILES.items():
        path = BASELINE_MOTIONS_DIR / filename
        key = (dataset, token)
        if key not in entries or not path.is_file():
            continue
        alternates = KNOWN_ALTERNATE_TAKES.get(key, [])
        entries[key].update(
            {
                "motion_file": filename,
                "status": "ready",
                "selection_reason": (
                    "Pre-existing approved motion file, frozen 2026-09-12; "
                    "reused unchanged, not regenerated."
                ),
                "generator_version": "extract_sign_motion.py (original, pre-integration)",
                "file_checksum": _sha256(path),
                "quality_metadata": {
                    **entries[key].get("quality_metadata", {}),
                    **({"alternate_takes_not_selected": alternates} if alternates else {}),
                },
            }
        )


def find_orphan_motion_files(entries: dict[tuple, dict[str, Any]]) -> list[str]:
    """Baseline motion files that are not the canonical motion_file for any
    entry and not a documented alternate take of one. Reported, never
    turned into a vocabulary entry.
    """
    if not BASELINE_MOTIONS_DIR.is_dir():
        return []
    referenced = {e["motion_file"] for e in entries.values() if e.get("motion_file")}
    documented_alternates = {f for files in KNOWN_ALTERNATE_TAKES.values() for f in files}
    orphans = []
    for path in BASELINE_MOTIONS_DIR.glob("*.motion.json"):
        if path.name in referenced or path.name in documented_alternates:
            continue
        orphans.append(path.name)
    return sorted(orphans)


CANONICAL_VOCABULARY_SIZES = {"jordanian_it": 165, "karsl": 502, "isharah": 680}
CANONICAL_TOTAL = sum(CANONICAL_VOCABULARY_SIZES.values())  # 1347


def _check_no_duplicate_tokens(dataset: str, fresh_rows: list[dict[str, Any]]) -> None:
    seen: dict[str, list[str]] = {}
    for row in fresh_rows:
        seen.setdefault(row["token"], []).append(row.get("label", row["token"]))
    duplicates = {token: labels for token, labels in seen.items() if len(labels) > 1}
    if duplicates:
        raise RuntimeError(
            f"{dataset}: {len(duplicates)} duplicate token(s) produced by the "
            f"inventory builder itself (before any motion-file registration): "
            f"{duplicates}. This means two distinct source rows normalized to "
            "the same token — fix the source-to-token mapping, do not silently "
            "collapse them."
        )
    if len(seen) != len(fresh_rows):
        raise RuntimeError(f"{dataset}: token count mismatch after grouping — investigate.")


def cmd_inventory(_args: argparse.Namespace) -> None:
    entries = load_manifest()
    preserved_ready = {k: v for k, v in entries.items() if v["status"] == "ready"}

    builders = {
        "jordanian_it": inventory.build_jordanian_it_inventory,
        "karsl": inventory.build_karsl_inventory,
        "isharah": inventory.build_isharah_inventory,
    }

    fresh: list[dict[str, Any]] = []
    for dataset, builder_fn in builders.items():
        rows = builder_fn()
        _check_no_duplicate_tokens(dataset, rows)
        expected = CANONICAL_VOCABULARY_SIZES[dataset]
        if len(rows) != expected:
            raise RuntimeError(
                f"{dataset}: canonical vocabulary source produced {len(rows)} tokens, "
                f"expected exactly {expected}. Stopping instead of silently accepting "
                "a mismatched count — inspect the source CSV for this dataset."
            )
        fresh.extend(rows)

    # Fields that record what generation actually produced; everything else
    # (label, selection_reason, source path, etc.) should come from the
    # fresh inventory rebuild so a source-metadata fix (e.g. the KArSL
    # numeric-placeholder label fix) also applies to already-ready entries,
    # without ever re-running generation or losing the ready status/file.
    GENERATION_RESULT_FIELDS = {"status", "motion_file", "file_checksum", "generator_version"}

    entries = {}
    for entry in fresh:
        key = (entry["dataset"], entry["token"])
        if key in preserved_ready:
            merged = dict(entry)
            old = preserved_ready[key]
            for field in GENERATION_RESULT_FIELDS:
                merged[field] = old[field]
            merged["quality_metadata"] = {**entry.get("quality_metadata", {}), **old.get("quality_metadata", {})}
            entries[key] = merged
        else:
            entries[key] = entry

    register_baseline_motions(entries)

    if len(entries) != CANONICAL_TOTAL:
        raise RuntimeError(
            f"Manifest has {len(entries)} entries after registering baseline "
            f"motions, expected exactly {CANONICAL_TOTAL} "
            f"({' + '.join(f'{v} {k}' for k, v in CANONICAL_VOCABULARY_SIZES.items())}). "
            "register_baseline_motions must only attach files to existing "
            "canonical tokens, never create a new one — investigate before saving."
        )

    mojibake = [
        f"{entry['dataset']}:{entry['token']}"
        for entry in entries.values()
        if classify_label(entry.get("label")) == "mojibake_suspected"
    ]
    if mojibake:
        raise RuntimeError(
            f"Refusing to save manifest: {len(mojibake)} label(s) contain a "
            f"mojibake marker (Ø/Ù/Ã/Â) — investigate before saving: {mojibake[:10]}"
        )

    save_manifest(entries)
    orphans = find_orphan_motion_files(entries)
    print(f"Inventory written: {len(entries)} entries -> {MANIFEST_PATH}")
    print(f"Canonical total check: {len(entries)} == {CANONICAL_TOTAL}: "
          f"{'PASS' if len(entries) == CANONICAL_TOTAL else 'FAIL'}")
    print(f"Orphan motion files (exist on disk, not a canonical entry's file): {orphans}")
    for dataset in CANONICAL_VOCABULARY_SIZES:
        labels = [e["label"] for e in entries.values() if e["dataset"] == dataset]
        print(f"Label quality [{dataset}]: {label_report(labels)}")
    _print_summary(entries)


def _entry_for(entries: dict[tuple, dict[str, Any]], dataset: str, token: str) -> dict[str, Any]:
    key = (dataset, token.upper() if dataset != "karsl" else token)
    if key not in entries:
        raise KeyError(f"{dataset}:{token} is not in the manifest. Run 'inventory' first.")
    return entries[key]


def generate_single_token(dataset: str, token: str) -> dict[str, Any]:
    entries = load_manifest()
    entry = _entry_for(entries, dataset, token)
    key = (entry["dataset"], entry["token"])

    if entry["status"] == "ready" and entry.get("motion_file"):
        candidate = (
            BASELINE_MOTIONS_DIR / entry["motion_file"]
            if (BASELINE_MOTIONS_DIR / entry["motion_file"]).is_file()
            else MOTION_LIBRARY_ROOT / dataset / entry["motion_file"]
        )
        if candidate.is_file() and (not entry.get("file_checksum") or _sha256(candidate) == entry["file_checksum"]):
            return {"token": entry["token"], "dataset": dataset, "status": "ready", "skipped": True}

    if entry["status"] == "source_unavailable":
        return {"token": entry["token"], "dataset": dataset, "status": "source_unavailable", "skipped": True}

    if entry["status"] == "validation_required":
        return {
            "token": entry["token"],
            "dataset": dataset,
            "status": "validation_required",
            "skipped": True,
            "reason": "Frame-bound alignment against the local source is unverified; "
                      "see selection_reason in the manifest. Not generated.",
        }

    output_dir = MOTION_LIBRARY_ROOT / dataset
    output_file = output_dir / f"{entry['token']}.motion.json"

    try:
        if dataset == "jordanian_it":
            frames = entry["source_frames"]
            stats = extractor.extract_from_video(
                video=Path(entry["source_path"]),
                output=output_file,
                name=entry["label"],
                model=MEDIAPIPE_MODEL,
                start_frame=frames["start"],
                end_frame=frames["end"],
            )
        elif dataset == "karsl":
            frames = entry["source_frames"]
            stats = extractor.extract_from_video(
                video=Path(entry["source_path"]),
                output=output_file,
                name=entry["label"],
                model=MEDIAPIPE_MODEL,
                start_frame=0,
                end_frame=frames.get("end"),
            )
        elif dataset == "isharah":
            frames = entry["source_frames"]
            stats = extractor.extract_from_image_sequence(
                frames_dir=Path(entry["source_path"]),
                output=output_file,
                name=entry["label"],
                model=MEDIAPIPE_MODEL,
                relative_start_frame=frames["start"],
                relative_end_frame=frames["end"],
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    except Exception as exc:  # noqa: BLE001 - one failure must not kill a batch
        entry["status"] = "extraction_failed"
        entry["selection_reason"] += f" | generation failed: {exc}"
        entries[key] = entry
        save_manifest(entries)
        return {"token": entry["token"], "dataset": dataset, "status": "extraction_failed", "error": str(exc)}

    entry["motion_file"] = output_file.name
    entry["status"] = "ready"
    entry["generator_version"] = GENERATOR_VERSION
    entry["file_checksum"] = _sha256(output_file)
    entry["quality_metadata"] = {**entry.get("quality_metadata", {}), **stats}
    entries[key] = entry
    save_manifest(entries)

    return {
        "token": entry["token"],
        "dataset": dataset,
        "status": "ready",
        "motion_file": str(output_file),
        "stats": stats,
    }


def cmd_generate_token(args: argparse.Namespace) -> None:
    result = generate_single_token(args.dataset, args.token)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_generate_dataset(args: argparse.Namespace) -> None:
    entries = load_manifest()
    pending = [
        e for (dataset, _token), e in entries.items()
        if dataset == args.dataset and e["status"] == "pending_generation"
    ]
    print(
        f"{len(pending)} pending {args.dataset} tokens. This command intentionally "
        "requires --confirm-full-batch to actually run, per the approved-tests-only "
        "gate in the integration brief."
    )
    if not args.confirm_full_batch:
        print("Refusing to run without --confirm-full-batch. Nothing generated.")
        return
    for index, entry in enumerate(pending, start=1):
        result = generate_single_token(entry["dataset"], entry["token"])
        print(f"[{index}/{len(pending)}] {entry['token']}: {result['status']}")


def cmd_resume(args: argparse.Namespace) -> None:
    cmd_generate_dataset(args)


def cmd_validate(_args: argparse.Namespace) -> None:
    entries = load_manifest()

    mojibake = [
        f"{e['dataset']}:{e['token']} -> {e['label']!r}"
        for e in entries.values()
        if classify_label(e.get("label")) == "mojibake_suspected"
    ]
    print(f"Mojibake check: {len(mojibake)} flagged" + (f" -> {mojibake}" if mojibake else " (clean)"))

    broken = 0
    for key, entry in entries.items():
        if entry["status"] != "ready" or not entry.get("motion_file"):
            continue
        candidate = BASELINE_MOTIONS_DIR / entry["motion_file"]
        if not candidate.is_file():
            candidate = MOTION_LIBRARY_ROOT / entry["dataset"] / entry["motion_file"]
        if not candidate.is_file():
            entry["status"] = "invalid_motion"
            entry["selection_reason"] += " | validation failed: file missing"
            broken += 1
            continue
        if entry.get("file_checksum") and _sha256(candidate) != entry["file_checksum"]:
            entry["status"] = "validation_required"
            entry["selection_reason"] += " | validation failed: checksum mismatch"
            broken += 1
    save_manifest(entries)
    print(f"Validated {len(entries)} entries; {broken} flagged.")


def cmd_rebuild_manifest(args: argparse.Namespace) -> None:
    cmd_inventory(args)


def _print_summary(entries: dict[tuple, dict[str, Any]]) -> None:
    by_status: dict[str, int] = {}
    by_dataset: dict[str, dict[str, int]] = {}
    tokens_by_dataset: dict[str, set[str]] = {}
    for entry in entries.values():
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
        bucket = by_dataset.setdefault(entry["dataset"], {})
        bucket[entry["status"]] = bucket.get(entry["status"], 0) + 1
        tokens_by_dataset.setdefault(entry["dataset"], set()).add(entry["token"])

    print("By status:", by_status)
    for dataset, counts in by_dataset.items():
        unique_tokens = len(tokens_by_dataset[dataset])
        dataset_total = sum(counts.values())
        duplicate_count = dataset_total - unique_tokens
        print(
            f"  {dataset}: {counts} | total {dataset_total} | "
            f"unique tokens {unique_tokens} | duplicate tokens {duplicate_count}"
        )

    status_sum = sum(by_status.values())
    print(
        f"Status-count sanity check: sum(by_status)={status_sum} == "
        f"total entries={len(entries)}: {'PASS' if status_sum == len(entries) else 'FAIL'}"
    )

    orphans = find_orphan_motion_files(entries)
    print(f"Orphan motion files: {orphans if orphans else 'none'}")


def cmd_coverage_report(_args: argparse.Namespace) -> None:
    entries = load_manifest()
    _print_summary(entries)
    print(f"Canonical total check: {len(entries)} == {CANONICAL_TOTAL}: "
          f"{'PASS' if len(entries) == CANONICAL_TOTAL else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge motion-library builder")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory").set_defaults(func=cmd_inventory)

    gen_token = sub.add_parser("generate-token")
    gen_token.add_argument("dataset", choices=["jordanian_it", "karsl", "isharah"])
    gen_token.add_argument("token")
    gen_token.set_defaults(func=cmd_generate_token)

    gen_dataset = sub.add_parser("generate-dataset")
    gen_dataset.add_argument("dataset", choices=["jordanian_it", "karsl", "isharah"])
    gen_dataset.add_argument("--confirm-full-batch", action="store_true")
    gen_dataset.set_defaults(func=cmd_generate_dataset)

    resume = sub.add_parser("resume")
    resume.add_argument("dataset", choices=["jordanian_it", "karsl", "isharah"])
    resume.add_argument("--confirm-full-batch", action="store_true")
    resume.set_defaults(func=cmd_resume)

    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("rebuild-manifest").set_defaults(func=cmd_rebuild_manifest)
    sub.add_parser("coverage-report").set_defaults(func=cmd_coverage_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
