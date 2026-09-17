"""Extract a reusable gloss-motion library from the trained Isharah CTC model.

The script performs CTC Viterbi forced alignment against each sample's known
gloss sequence.  It keeps several high-quality candidates for every training
gloss and stores their original 198-D, full-frame-rate feature segments in one
compact flat array.

By default only the train and dev splits are used.  The untouched test split is
deliberately excluded so the project's reported test WER remains valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA = Path(r"C:\SignBridge_Project\data\processed\isharah1000_shared198")
DEFAULT_CHECKPOINT = Path(
    r"C:\SignBridge_Project\data\processed\isharah_ctc_training\best_isharah_ctc.pt"
)
DEFAULT_OUTPUT = Path(
    r"C:\SignBridge_Project\data\processed\isharah_gloss_clip_library"
)

BLANK_ID = 0
UNK_ID = 1
SPECIAL_TOKENS = {"<blank>", "<unk>"}
PREFERRED_MIN_CLIP_FRAMES = 12


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all Isharah gloss candidates using CTC forced alignment."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "dev", "test"),
        default=("train", "dev"),
        help="Default excludes test to preserve untouched evaluation.",
    )
    parser.add_argument("--max-clips-per-gloss", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ContinuousSignCTC(nn.Module):
    """Exact architecture used by train_isharah_ctc.py."""

    def __init__(self, input_features: int, vocabulary_size: int) -> None:
        super().__init__()
        width = 256
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, width),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.sequence_encoder = nn.GRU(
            input_size=width,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.20,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.20), nn.Linear(512, vocabulary_size)
        )

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.input_encoder(x)
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        out_lengths = self.output_lengths(lengths)
        packed = pack_padded_sequence(
            x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed, _ = self.sequence_encoder(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        return self.classifier(x), out_lengths


class AlignmentDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        rows: list[dict[str, str]],
        splits: set[str],
        token_to_id: dict[str, int],
        stride: int,
    ) -> None:
        self.feature_path = data_dir / "features_flat.npy"
        self.rows = [row for row in rows if row["split"] in splits]
        self.token_to_id = token_to_id
        self.stride = stride
        self.features: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _array(self) -> np.ndarray:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        return self.features

    def __getitem__(self, index: int):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        base = np.asarray(
            self._array()[start:end:self.stride], dtype=np.float32
        ).copy()
        if len(base) == 0:
            raise ValueError(f"Empty sequence: {row['sample_id']}")

        velocity = np.zeros_like(base)
        velocity[1:] = base[1:] - base[:-1]
        model_features = np.concatenate([base, velocity], axis=1)
        tokens = row["gloss"].split()
        target = [self.token_to_id.get(token, UNK_ID) for token in tokens]
        return torch.from_numpy(model_features), target, tokens, row


def collate_batch(batch):
    sequences, targets, tokens, rows = zip(*batch)
    lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    feature_count = int(sequences[0].shape[1])
    padded = torch.zeros(len(batch), max_length, feature_count, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return padded, lengths, targets, tokens, rows


def greedy_decode(log_probs: np.ndarray) -> list[int]:
    result: list[int] = []
    previous: int | None = None
    for token in np.argmax(log_probs, axis=1).tolist():
        if token != BLANK_ID and token != previous:
            result.append(int(token))
        previous = int(token)
    return result


def ctc_forced_alignment(
    log_probs: np.ndarray, target: list[int]
) -> tuple[np.ndarray, float] | None:
    """Return the best CTC state path and its log score."""
    time_steps = int(log_probs.shape[0])
    if time_steps == 0 or not target:
        return None

    extended: list[int] = [BLANK_ID]
    for token in target:
        extended.extend((int(token), BLANK_ID))
    states = len(extended)

    previous = np.full(states, -np.inf, dtype=np.float64)
    backpointers = np.full((time_steps, states), -1, dtype=np.int16)
    previous[0] = float(log_probs[0, BLANK_ID])
    if states > 1:
        previous[1] = float(log_probs[0, extended[1]])

    for time_index in range(1, time_steps):
        current = np.full(states, -np.inf, dtype=np.float64)
        for state in range(states):
            best_state = state
            best_score = previous[state]

            if state >= 1 and previous[state - 1] > best_score:
                best_state = state - 1
                best_score = previous[state - 1]

            if (
                state >= 2
                and extended[state] != BLANK_ID
                and extended[state] != extended[state - 2]
                and previous[state - 2] > best_score
            ):
                best_state = state - 2
                best_score = previous[state - 2]

            if np.isfinite(best_score):
                current[state] = best_score + float(
                    log_probs[time_index, extended[state]]
                )
                backpointers[time_index, state] = best_state
        previous = current

    final_states = [states - 1]
    if states > 1:
        final_states.append(states - 2)
    final_state = max(final_states, key=lambda state: previous[state])
    final_score = float(previous[final_state])
    if not np.isfinite(final_score):
        return None

    path = np.empty(time_steps, dtype=np.int16)
    state = final_state
    for time_index in range(time_steps - 1, -1, -1):
        path[time_index] = state
        if time_index > 0:
            state = int(backpointers[time_index, state])
            if state < 0:
                return None
    return path, final_score


def frame_boundaries(centres: list[int], output_steps: int, frames: int) -> list[int]:
    """Partition a sequence around CTC peaks with a duration prior.

    CTC posteriors are intentionally peaky: a complete sign can receive a
    one-timestep label spike even when its motion spans many frames.  Midpoints
    alone can consequently create unrealistic four-frame clips.  The duration
    prior below keeps every segment as close as possible to the CTC midpoint
    while reserving a reasonable minimum duration whenever the source sequence
    is long enough.  Segments remain non-overlapping and cover every frame.
    """
    model_boundaries = [0]
    for left, right in zip(centres, centres[1:]):
        model_boundaries.append((left + right + 1) // 2)
    model_boundaries.append(output_steps)

    result = [int(round(boundary * frames / output_steps)) for boundary in model_boundaries]
    result[0], result[-1] = 0, frames

    token_count = len(centres)
    minimum = min(PREFERRED_MIN_CLIP_FRAMES, max(1, frames // token_count))

    # Clamp each internal boundary to the globally feasible interval.
    for index in range(1, token_count):
        lower = index * minimum
        upper = frames - (token_count - index) * minimum
        result[index] = min(max(result[index], lower), upper)

    # Two directional passes enforce the local minimum-duration constraint.
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1] + minimum)
    result[-1] = frames
    for index in range(len(result) - 2, -1, -1):
        result[index] = min(result[index], result[index + 1] - minimum)
    result[0] = 0
    result[-1] = frames
    return result


def align_sample(
    log_probs: np.ndarray,
    target_ids: list[int],
    tokens: list[str],
    row: dict[str, str],
    vocabulary: set[str],
) -> list[dict[str, object]]:
    aligned = ctc_forced_alignment(log_probs, target_ids)
    if aligned is None:
        return []
    state_path, path_score = aligned

    token_times: list[np.ndarray] = []
    centres: list[int] = []
    for position, token_id in enumerate(target_ids):
        times = np.flatnonzero(state_path == 2 * position + 1)
        if len(times) == 0:
            return []
        token_times.append(times)
        centres.append(int(times[np.argmax(log_probs[times, token_id])]))

    source_frames = int(row["end"]) - int(row["start"])
    boundaries = frame_boundaries(centres, len(log_probs), source_frames)
    predicted_ids = greedy_decode(log_probs)
    exact_match = predicted_ids == target_ids
    path_confidence = float(math.exp(path_score / max(len(log_probs), 1)))
    source_group = row["sample_id"].split("_", 1)[0]

    candidates: list[dict[str, object]] = []
    occurrence_counter: Counter[str] = Counter()
    for position, (token, token_id, times) in enumerate(
        zip(tokens, target_ids, token_times)
    ):
        occurrence_counter[token] += 1
        if token not in vocabulary or token in SPECIAL_TOKENS:
            continue

        probabilities = np.exp(log_probs[times, token_id])
        peak_confidence = float(np.max(probabilities))
        state_mean_confidence = float(np.mean(probabilities))
        candidate_score = 0.70 * peak_confidence + 0.30 * state_mean_confidence
        relative_start, relative_end = boundaries[position], boundaries[position + 1]
        clip_frames = relative_end - relative_start
        review_required = (
            not exact_match
            or peak_confidence < 0.40
            or clip_frames < 5
            or clip_frames > 240
        )

        candidates.append(
            {
                "gloss": token,
                "gloss_occurrence_in_sample": occurrence_counter[token],
                "gloss_position": position,
                "sample_id": row["sample_id"],
                "source_group": source_group,
                "split": row["split"],
                "reference_gloss": row["gloss"],
                "arabic_text": row.get("text", ""),
                "source_sequence_start": int(row["start"]),
                "source_sequence_end": int(row["end"]),
                "relative_start_frame": relative_start,
                "relative_end_frame": relative_end,
                "source_global_start": int(row["start"]) + relative_start,
                "source_global_end": int(row["start"]) + relative_end,
                "clip_frames": clip_frames,
                "ctc_peak_confidence": peak_confidence,
                "ctc_state_mean_confidence": state_mean_confidence,
                "ctc_path_confidence": path_confidence,
                "candidate_score": candidate_score,
                "sample_exact_match": exact_match,
                "review_required": review_required,
            }
        )
    return candidates


def candidate_sort_key(item: dict[str, object]):
    return (
        int(bool(item["sample_exact_match"])),
        float(item["candidate_score"]),
        float(item["ctc_peak_confidence"]),
        -abs(int(item["clip_frames"]) - 40),
    )


def select_diverse_candidates(
    candidates: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    ordered = sorted(candidates, key=candidate_sort_key, reverse=True)
    selected: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    used_groups: set[str] = set()

    # First pass gives each available signer/source group a chance.
    for item in ordered:
        group = str(item["source_group"])
        if group in used_groups:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        used_groups.add(group)
        if len(selected) >= limit:
            return selected

    # Second pass fills the remaining slots with the strongest candidates.
    for item in ordered:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = arguments()
    if args.max_clips_per_gloss < 1:
        raise ValueError("--max-clips-per-gloss must be at least 1")
    seed_everything(args.seed)

    data_dir = args.data.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")

    required = [data_dir / "features_flat.npy", data_dir / "samples.csv", checkpoint_path]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    id_to_token = list(checkpoint["vocabulary"])
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    vocabulary = set(id_to_token) - SPECIAL_TOKENS
    stride = int(checkpoint.get("stride", 2))
    shared_features = int(checkpoint.get("shared_features", 198))
    input_features = int(checkpoint.get("input_features", shared_features * 2))

    model = ContinuousSignCTC(input_features, len(id_to_token)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = read_rows(data_dir / "samples.csv")
    selected_splits = set(args.splits)
    dataset = AlignmentDataset(
        data_dir, rows, selected_splits, token_to_id, stride
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    print("Extracting complete Isharah gloss library")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Splits:", ", ".join(args.splits))
    print("Samples:", f"{len(dataset):,}")
    print("Training glosses:", f"{len(vocabulary):,}")
    print("Candidates kept per gloss:", args.max_clips_per_gloss)
    print("Untouched test excluded:", "test" not in selected_splits)

    started = time.perf_counter()
    candidates: list[dict[str, object]] = []
    alignment_failures: list[dict[str, object]] = []
    oov_counts: Counter[str] = Counter()
    processed = 0

    with torch.no_grad():
        for padded, lengths, batch_targets, batch_tokens, batch_rows in loader:
            logits, output_lengths = model(padded.to(device), lengths)
            batch_log_probs = logits.log_softmax(-1).cpu().numpy()

            for index, (target_ids, tokens, row) in enumerate(
                zip(batch_targets, batch_tokens, batch_rows)
            ):
                length = int(output_lengths[index])
                log_probs = batch_log_probs[index, :length]
                for token in tokens:
                    if token not in vocabulary:
                        oov_counts[token] += 1

                sample_candidates = align_sample(
                    log_probs, target_ids, tokens, row, vocabulary
                )
                if not sample_candidates:
                    alignment_failures.append(
                        {
                            "sample_id": row["sample_id"],
                            "split": row["split"],
                            "reference_gloss": row["gloss"],
                            "frames": int(row["end"]) - int(row["start"]),
                        }
                    )
                else:
                    candidates.extend(sample_candidates)
                processed += 1

            if processed % 500 < len(batch_rows) or processed == len(dataset):
                print(f"Aligned {processed:,}/{len(dataset):,} samples")

    by_gloss: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        by_gloss[str(item["gloss"])].append(item)

    selected: list[dict[str, object]] = []
    for gloss in sorted(vocabulary):
        chosen = select_diverse_candidates(
            by_gloss.get(gloss, []), args.max_clips_per_gloss
        )
        for rank, item in enumerate(chosen, 1):
            item = dict(item)
            item["rank_within_gloss"] = rank
            selected.append(item)

    selected.sort(key=lambda item: (str(item["gloss"]), int(item["rank_within_gloss"])))
    total_selected_frames = sum(int(item["clip_frames"]) for item in selected)
    source_features = np.load(data_dir / "features_flat.npy", mmap_mode="r")
    if source_features.ndim != 2 or source_features.shape[1] != shared_features:
        raise ValueError(
            f"Expected source features (*, {shared_features}), got {source_features.shape}"
        )

    library_features = np.lib.format.open_memmap(
        partial_dir / "features_flat.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_selected_frames, shared_features),
    )
    offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    clip_ids: list[str] = []
    glosses: list[str] = []
    cursor = 0
    for index, item in enumerate(selected):
        source_start = int(item["source_global_start"])
        source_end = int(item["source_global_end"])
        length = source_end - source_start
        library_features[cursor : cursor + length] = source_features[source_start:source_end]
        clip_id = f"clip_{index + 1:06d}"
        item["clip_id"] = clip_id
        item["library_start"] = cursor
        item["library_end"] = cursor + length
        offsets[index] = cursor
        clip_ids.append(clip_id)
        glosses.append(str(item["gloss"]))
        cursor += length
    offsets[-1] = cursor
    library_features.flush()
    del library_features

    np.save(partial_dir / "offsets.npy", offsets)
    np.save(partial_dir / "lengths.npy", np.diff(offsets))
    max_clip_id = max((len(value) for value in clip_ids), default=1)
    max_gloss = max((len(value) for value in glosses), default=1)
    np.save(partial_dir / "clip_ids.npy", np.asarray(clip_ids, dtype=f"<U{max_clip_id}"))
    np.save(partial_dir / "glosses.npy", np.asarray(glosses, dtype=f"<U{max_gloss}"))

    candidate_fields = [
        "gloss",
        "gloss_occurrence_in_sample",
        "gloss_position",
        "sample_id",
        "source_group",
        "split",
        "reference_gloss",
        "arabic_text",
        "source_sequence_start",
        "source_sequence_end",
        "relative_start_frame",
        "relative_end_frame",
        "source_global_start",
        "source_global_end",
        "clip_frames",
        "ctc_peak_confidence",
        "ctc_state_mean_confidence",
        "ctc_path_confidence",
        "candidate_score",
        "sample_exact_match",
        "review_required",
    ]
    selected_fields = [
        "clip_id",
        "rank_within_gloss",
        "library_start",
        "library_end",
        *candidate_fields,
    ]
    write_csv(partial_dir / "all_alignment_candidates.csv", candidates, candidate_fields)
    write_csv(partial_dir / "clips.csv", selected, selected_fields)
    write_csv(
        partial_dir / "alignment_failures.csv",
        alignment_failures,
        ["sample_id", "split", "reference_gloss", "frames"],
    )

    summary_rows: list[dict[str, object]] = []
    for gloss in sorted(vocabulary):
        available = by_gloss.get(gloss, [])
        chosen = [item for item in selected if item["gloss"] == gloss]
        summary_rows.append(
            {
                "gloss": gloss,
                "available_candidates": len(available),
                "selected_clips": len(chosen),
                "selected_exact_matches": sum(
                    bool(item["sample_exact_match"]) for item in chosen
                ),
                "selected_review_required": sum(
                    bool(item["review_required"]) for item in chosen
                ),
                "best_candidate_score": max(
                    (float(item["candidate_score"]) for item in available),
                    default=0.0,
                ),
            }
        )
    write_csv(
        partial_dir / "gloss_summary.csv",
        summary_rows,
        [
            "gloss",
            "available_candidates",
            "selected_clips",
            "selected_exact_matches",
            "selected_review_required",
            "best_candidate_score",
        ],
    )

    with (partial_dir / "ctc_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(id_to_token, handle, ensure_ascii=False, indent=2)
    with (partial_dir / "oov_gloss_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(oov_counts.items())), handle, ensure_ascii=False, indent=2)

    config = {
        "method": "CTC Viterbi forced alignment",
        "source_data": str(data_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_dev_wer": checkpoint.get("dev_wer"),
        "splits_used": list(args.splits),
        "test_split_excluded": "test" not in selected_splits,
        "samples_processed": len(dataset),
        "training_glosses": len(vocabulary),
        "glosses_with_selected_clips": sum(
            int(row["selected_clips"] > 0) for row in summary_rows
        ),
        "all_candidates": len(candidates),
        "selected_clips": len(selected),
        "max_clips_per_gloss": args.max_clips_per_gloss,
        "selected_frames": total_selected_frames,
        "features_per_frame": shared_features,
        "model_input_features": input_features,
        "training_stride": stride,
        "preferred_minimum_clip_frames": PREFERRED_MIN_CLIP_FRAMES,
        "alignment_failures": len(alignment_failures),
        "oov_gloss_occurrences": dict(sorted(oov_counts.items())),
        "review_rule": (
            "sample not exact, peak confidence below 0.40, or clip outside 5..240 frames"
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (partial_dir / "extraction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    partial_dir.replace(output_dir)
    missing_glosses = [
        row["gloss"] for row in summary_rows if int(row["selected_clips"]) == 0
    ]
    review_count = sum(bool(item["review_required"]) for item in selected)

    print("\nGloss library extraction completed.")
    print("Glosses with clips:", f"{config['glosses_with_selected_clips']:,}/{len(vocabulary):,}")
    print("Selected clips:", f"{len(selected):,}")
    print("Clips requiring review:", f"{review_count:,}")
    print("Alignment failures:", f"{len(alignment_failures):,}")
    print("OOV glosses skipped:", sorted(oov_counts))
    print("Glosses without clips:", missing_glosses)
    print("Elapsed seconds:", f"{config['elapsed_seconds']:.1f}")
    print("Output:", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
