"""Run the final SignBridge multi-task recognizer on one video.

The predictor supports three inference paths from the same checkpoint:

* Jordanian IT isolated signs (IT classification head + CTC evidence).
* KArSL isolated signs (KArSL classification head + CTC evidence).
* Continuous Isharah-style signing (CTC decoding).

Examples:
  python predict_final_signbridge_video.py --video "C:\\path\\to\\Stack_01_1.mov"
  python predict_final_signbridge_video.py --video "C:\\path\\to\\karsl.mp4" --mode karsl
  python predict_final_signbridge_video.py --video "C:\\path\\to\\question.mp4" --mode continuous
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from predict_it_video import HOLISTIC_MODEL, active_bounds, extract_video
from prepare_unified_sign_data import jordanian_572_to_shared_198
from scripts.prepare_model_sequences import prepare_sequence
from train_unified_karsl_multitask import UnifiedMultiTaskModel
from train_unified_sign_ctc import make_model_features


PROJECT_ROOT = Path(r"C:\SignBridge_Project")
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_karsl_multitask_training"
    / "best_unified_karsl_multitask.pt"
)
DEFAULT_KARSL_LABELS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unified_sign_data_karsl"
    / "karsl_sign_mapping.csv"
)
SPECIAL_TOKENS = {"<blank>", "<unk>"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize a video with the final SignBridge multi-task model."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "it", "karsl", "continuous", "single"),
        default="auto",
        help=(
            "auto detects KArSL paths and known Jordanian IT videos; "
            "single is retained as an alias for it"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--karsl-labels", type=Path, default=DEFAULT_KARSL_LABELS)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def detect_mode(video: Path, found_in_it_report: bool, requested: str) -> str:
    if requested == "single":
        return "it"
    if requested != "auto":
        return requested
    lowered_parts = {part.casefold() for part in video.parts}
    if "karsl" in lowered_parts or any("karsl" in part for part in lowered_parts):
        return "karsl"
    if found_in_it_report:
        return "it"
    return "continuous"


def resolve_bounds(
    video: Path,
    frame_count: int,
    mode: str,
    requested_start: int | None,
    requested_end: int | None,
) -> tuple[int, int, str]:
    manual = requested_start is not None or requested_end is not None
    if manual:
        start = 0 if requested_start is None else requested_start
        end = frame_count - 1 if requested_end is None else requested_end
        source = "manual"
    elif mode == "it":
        start, end, found = active_bounds(video, frame_count)
        source = "saved IT active report" if found else "full video"
    else:
        start, end = 0, frame_count - 1
        source = "full video"

    if start < 0 or end < 0:
        raise ValueError("Frame bounds cannot be negative")
    if start >= frame_count:
        raise ValueError(
            f"start-frame {start} is outside a video with {frame_count} frames"
        )
    end = min(end, frame_count - 1)
    if end < start:
        raise ValueError("end-frame must be greater than or equal to start-frame")
    return start, end, source


def greedy_decode_with_confidence(
    logits: torch.Tensor,
    output_length: int,
) -> tuple[list[int], list[float]]:
    probabilities = logits[0, :output_length].softmax(dim=-1)
    frame_scores, frame_tokens = probabilities.max(dim=-1)

    decoded: list[int] = []
    confidence: list[float] = []
    previous: int | None = None
    run_scores: list[float] = []

    def finish_run() -> None:
        nonlocal run_scores
        if previous is not None and previous != 0 and run_scores:
            decoded.append(previous)
            confidence.append(float(np.mean(run_scores)))
        run_scores = []

    for raw_token, raw_score in zip(
        frame_tokens.detach().cpu().tolist(),
        frame_scores.detach().cpu().tolist(),
    ):
        token = int(raw_token)
        score = float(raw_score)
        if token != previous:
            finish_run()
            previous = token
        if token != 0:
            run_scores.append(score)
    finish_run()
    return decoded, confidence


def temporal_token_evidence(
    logits: torch.Tensor,
    output_length: int,
    vocabulary: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    probabilities = logits[0, :output_length].softmax(dim=-1)
    evidence = probabilities.max(dim=0).values.detach().cpu()
    candidates = [
        index
        for index, token in enumerate(vocabulary)
        if index != 0 and token not in SPECIAL_TOKENS
    ]
    candidates.sort(key=lambda index: float(evidence[index]), reverse=True)
    return [
        (vocabulary[index], float(evidence[index]))
        for index in candidates[: min(max(1, top_k), len(candidates))]
    ]


def classification_top_k(
    logits: torch.Tensor,
    index_to_label: Dict[int, str],
    top_k: int,
) -> list[tuple[str, float]]:
    probabilities = logits[0].softmax(dim=-1).detach().cpu()
    count = min(max(1, top_k), probabilities.numel())
    scores, indices = torch.topk(probabilities, count)
    return [
        (index_to_label[int(index)], float(score))
        for index, score in zip(indices.tolist(), scores.tolist())
    ]


def first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def load_karsl_labels(path: Path) -> Dict[str, str]:
    """Load a human-readable KArSL word when the prepared mapping provides it."""
    if not path.is_file():
        return {}
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sign_id = first_present(row, ("sign_id", "class_id", "id"))
            label = first_present(
                row,
                (
                    "label",
                    "label_ar",
                    "arabic_label",
                    "karsl_label",
                    "word",
                    "gloss",
                    "label_en",
                    "canonical_token",
                    "mapped_token",
                    "token",
                ),
            )
            if sign_id and label:
                mapping[sign_id.zfill(4)] = label
    return mapping


def display_karsl_label(sign_id: str, labels: Dict[str, str]) -> str:
    normalized = sign_id.removeprefix("KARSL_").zfill(4)
    word = labels.get(normalized)
    return f"{word} (KArSL {normalized})" if word else f"KArSL {normalized}"


def main() -> None:
    args = arguments()
    video = args.video.resolve()
    checkpoint_path = args.checkpoint.resolve()
    for required in (video, HOLISTIC_MODEL, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    print("Final SignBridge multi-task video recognition")
    print("Video:", video)
    print("Extracting MediaPipe landmarks...")
    raw = extract_video(video)
    if len(raw) == 0:
        raise RuntimeError("No frames were extracted from the video")

    _, _, found_in_it_report = active_bounds(video, len(raw))
    mode = detect_mode(video, found_in_it_report, args.mode)
    start, end, bounds_source = resolve_bounds(
        video, len(raw), mode, args.start_frame, args.end_frame
    )

    active_frames = end - start + 1
    # Jordanian IT isolated clips were trained at exactly 200 frames.  KArSL
    # and continuous clips preserve their temporal length before stride 2.
    target_frames = 200 if mode == "it" else active_frames
    prepared_572, stats = prepare_sequence(raw, start, end, target_frames, 5)
    shared_198 = jordanian_572_to_shared_198(prepared_572)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = list(checkpoint["vocabulary"])
    karsl_class_to_index = {
        str(key).zfill(4): int(value)
        for key, value in checkpoint["karsl_class_to_index"].items()
    }
    it_label_to_class = {
        str(key): int(value) for key, value in checkpoint["it_label_to_class"].items()
    }
    index_to_karsl = {value: key for key, value in karsl_class_to_index.items()}
    index_to_it = {value: key for key, value in it_label_to_class.items()}

    stride = int(checkpoint.get("stride", 2))
    features = make_model_features(shared_198, stride)
    expected_features = int(checkpoint.get("input_features", features.shape[1]))
    if features.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: checkpoint expects {expected_features}, "
            f"but video preparation produced {features.shape[1]}"
        )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model = UnifiedMultiTaskModel(
        vocabulary_size=len(vocabulary),
        karsl_classes=len(karsl_class_to_index),
        it_classes=len(it_label_to_class),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tensor = features.unsqueeze(0).to(device)
    input_lengths = torch.tensor([len(features)], dtype=torch.long)
    with torch.inference_mode():
        logits, output_lengths, encoded = model(tensor, input_lengths)

    output_length = int(output_lengths[0])
    token_ids, token_confidences = greedy_decode_with_confidence(
        logits, output_length
    )
    predicted_tokens = [vocabulary[index] for index in token_ids]

    print("Device:", device)
    print("Mode:", mode)
    print("Frames:", len(raw))
    print("Active segment:", f"{start}:{end}", f"({bounds_source})")
    print(
        "Left/right hand observed:",
        f"{stats['resampled_left_observed_rate']:.2f}% / "
        f"{stats['resampled_right_observed_rate']:.2f}%",
    )
    print(
        "CTC predicted gloss:",
        " ".join(predicted_tokens) if predicted_tokens else "<empty>",
    )
    if predicted_tokens:
        print("Decoded CTC token confidence:")
        for token, score in zip(predicted_tokens, token_confidences):
            print(f"- {token}: {score * 100:.2f}%")

    if mode == "it":
        with torch.inference_mode():
            class_logits, _ = model.isolated_logits(encoded, output_lengths, "it")
        print(f"Top-{args.top_k} Jordanian IT classification:")
        for rank, (label, score) in enumerate(
            classification_top_k(class_logits, index_to_it, args.top_k), start=1
        ):
            print(f"{rank}. {label}: {score * 100:.2f}%")
    elif mode == "karsl":
        with torch.inference_mode():
            class_logits, _ = model.isolated_logits(encoded, output_lengths, "karsl")
        human_labels = load_karsl_labels(args.karsl_labels.resolve())
        print(f"Top-{args.top_k} KArSL classification:")
        for rank, (sign_id, score) in enumerate(
            classification_top_k(class_logits, index_to_karsl, args.top_k), start=1
        ):
            print(
                f"{rank}. {display_karsl_label(sign_id, human_labels)}: "
                f"{score * 100:.2f}%"
            )

    if mode in {"it", "karsl"}:
        print(f"Top-{args.top_k} temporal CTC token evidence:")
        for rank, (token, score) in enumerate(
            temporal_token_evidence(logits, output_length, vocabulary, args.top_k),
            start=1,
        ):
            if token.startswith("KARSL_"):
                token = display_karsl_label(
                    token, load_karsl_labels(args.karsl_labels.resolve())
                )
            print(f"{rank}. {token}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
