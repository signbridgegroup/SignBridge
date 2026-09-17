"""Train one continuous CTC recognizer for Isharah and Jordanian IT signs.

Training uses three complementary sources:
  1. Original continuous Isharah training sentences.
  2. Signer-independent Jordanian IT training clips as one-token sequences.
  3. On-the-fly mixed sequences that insert uniformly sampled IT terms into
     real Isharah gloss patterns reconstructed from aligned gloss clips.

The Isharah test split and Jordanian signer-independent test split remain
untouched until the best validation checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset


DEFAULT_ISHARAH_DATA = Path(
    r"C:\SignBridge_Project\data\processed\isharah1000_shared198"
)
DEFAULT_GLOSS_LIBRARY = Path(
    r"C:\SignBridge_Project\data\processed\isharah_gloss_clip_library"
)
DEFAULT_UNIFIED_DATA = Path(
    r"C:\SignBridge_Project\data\processed\unified_sign_data"
)
DEFAULT_PRETRAINED = Path(
    r"C:\SignBridge_Project\data\processed\isharah_ctc_training\best_isharah_ctc.pt"
)
DEFAULT_OUTPUT = Path(
    r"C:\SignBridge_Project\data\processed\unified_ctc_training"
)

BLANK_ID = 0
UNK_ID = 1
SHARED_FEATURES = 198
MODEL_FEATURES = 396


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified continuous Isharah + Jordanian IT CTC model."
    )
    parser.add_argument("--isharah-data", type=Path, default=DEFAULT_ISHARAH_DATA)
    parser.add_argument("--gloss-library", type=Path, default=DEFAULT_GLOSS_LIBRARY)
    parser.add_argument("--unified-data", type=Path, default=DEFAULT_UNIFIED_DATA)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--it-repeat", type=int, default=5)
    parser.add_argument("--synthetic-train-samples", type=int, default=6000)
    parser.add_argument("--synthetic-dev-samples", type=int, default=500)
    parser.add_argument("--max-general-glosses", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp))
            )
        previous = current
    return previous[-1]


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    predictions = logits.argmax(dim=-1).cpu()
    decoded: list[list[int]] = []
    for sequence, length in zip(predictions, lengths.cpu()):
        result: list[int] = []
        previous: int | None = None
        for raw_token in sequence[: int(length)].tolist():
            token = int(raw_token)
            if token != BLANK_ID and token != previous:
                result.append(token)
            previous = token
        decoded.append(result)
    return decoded


class ContinuousSignCTC(nn.Module):
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


def make_model_features(base: np.ndarray, stride: int) -> torch.Tensor:
    sampled = np.asarray(base[::stride], dtype=np.float32).copy()
    if len(sampled) == 0:
        raise ValueError("A prepared sequence became empty")
    velocity = np.zeros_like(sampled)
    velocity[1:] = sampled[1:] - sampled[:-1]
    return torch.from_numpy(np.concatenate([sampled, velocity], axis=1))


def temporal_resample(base: np.ndarray, target_frames: int) -> np.ndarray:
    if len(base) == target_frames:
        return base.astype(np.float32, copy=True)
    source_time = np.linspace(0.0, 1.0, len(base))
    target_time = np.linspace(0.0, 1.0, target_frames)
    output = np.empty((target_frames, SHARED_FEATURES), dtype=np.float32)
    for feature in range(SHARED_FEATURES - 2):
        output[:, feature] = np.interp(target_time, source_time, base[:, feature])
    nearest = np.rint(
        np.linspace(0, len(base) - 1, target_frames)
    ).astype(np.int64)
    output[:, -2:] = base[nearest, -2:]
    output[:, -2:] = (output[:, -2:] >= 0.5).astype(np.float32)
    return output


def augment_segment(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = float(rng.uniform(0.90, 1.10))
    target_frames = max(8, int(round(len(base) * scale)))
    output = temporal_resample(base, target_frames)
    noise = rng.normal(0.0, 0.008, size=output[:, :-2].shape).astype(np.float32)
    output[:, :-2] += noise
    return output


def concatenate_segments(
    segments: list[np.ndarray], transition_frames: int = 2
) -> np.ndarray:
    if not segments:
        raise ValueError("No segments were supplied")
    pieces = [segments[0].astype(np.float32, copy=False)]
    for right in segments[1:]:
        left = pieces[-1]
        if transition_frames > 0:
            blend = np.empty((transition_frames, SHARED_FEATURES), dtype=np.float32)
            for offset in range(transition_frames):
                alpha = (offset + 1) / (transition_frames + 1)
                blend[offset, :-2] = (
                    (1.0 - alpha) * left[-1, :-2] + alpha * right[0, :-2]
                )
                blend[offset, -2:] = (
                    left[-1, -2:] if alpha < 0.5 else right[0, -2:]
                )
            pieces.append(blend)
        pieces.append(right.astype(np.float32, copy=False))
    return np.concatenate(pieces, axis=0)


class IsharahDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        rows: list[dict[str, str]],
        split: str,
        token_to_id: dict[str, int],
        stride: int,
    ) -> None:
        self.feature_path = data_dir / "features_flat.npy"
        self.rows = [row for row in rows if row["split"] == split]
        self.token_to_id = token_to_id
        self.stride = stride
        self.features: np.ndarray | None = None

    def _array(self) -> np.ndarray:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        return self.features

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        base = self._array()[start:end]
        target = [self.token_to_id.get(token, UNK_ID) for token in row["gloss"].split()]
        return (
            make_model_features(base, self.stride),
            torch.tensor(target, dtype=torch.long),
            str(row["sample_id"]),
            str(row["gloss"]),
            "isharah",
        )


class ITDataset(Dataset):
    def __init__(
        self,
        unified_dir: Path,
        rows: list[dict[str, str]],
        split: str,
        stride: int,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.feature_path = unified_dir / "it_features_flat.npy"
        self.rows = [row for row in rows if row["split"] == split]
        self.stride = stride
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.features: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _array(self) -> np.ndarray:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        return self.features

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        base = np.asarray(self._array()[start:end], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            base = augment_segment(base, rng)
        target_id = int(row["unified_token_id"])
        return (
            make_model_features(base, self.stride),
            torch.tensor([target_id], dtype=torch.long),
            str(row["video"]),
            str(row["label"]),
            "it",
        )


class RepeatDataset(Dataset):
    def __init__(self, dataset: Dataset, repeats: int) -> None:
        if repeats < 1:
            raise ValueError("Repeat count must be at least one")
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.dataset) * self.repeats

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


class SyntheticMixedDataset(Dataset):
    def __init__(
        self,
        isharah_rows: list[dict[str, str]],
        gloss_library_dir: Path,
        gloss_rows: list[dict[str, str]],
        it_unified_dir: Path,
        it_rows: list[dict[str, str]],
        isharah_split: str,
        it_split: str,
        size: int,
        token_to_id: dict[str, int],
        stride: int,
        max_general_glosses: int,
        seed: int,
        augment: bool,
    ) -> None:
        self.patterns = [
            row["gloss"].split()
            for row in isharah_rows
            if row["split"] == isharah_split
        ]
        self.size = size
        self.token_to_id = token_to_id
        self.stride = stride
        self.max_general_glosses = max_general_glosses
        self.seed = seed
        self.augment = augment
        self.epoch = 0

        self.gloss_feature_path = gloss_library_dir / "features_flat.npy"
        self.it_feature_path = it_unified_dir / "it_features_flat.npy"
        self.gloss_features: np.ndarray | None = None
        self.it_features: np.ndarray | None = None

        self.gloss_by_token: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in gloss_rows:
            if (
                row["split"] == isharah_split
                and row["review_required"].lower() == "false"
            ):
                self.gloss_by_token[row["gloss"]].append(row)

        # Keep only patterns that can be reconstructed without crossing splits.
        self.patterns = [
            pattern
            for pattern in self.patterns
            if pattern and all(token in self.gloss_by_token for token in pattern)
        ]

        self.it_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in it_rows:
            if row["split"] == it_split:
                self.it_by_label[row["label"]].append(row)
        self.it_labels = sorted(self.it_by_label, key=str.casefold)
        if not self.patterns:
            raise ValueError(
                f"No reconstructable Isharah patterns for split={isharah_split}"
            )
        if not self.it_labels:
            raise ValueError(f"No Jordanian IT labels for split={it_split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _gloss_array(self) -> np.ndarray:
        if self.gloss_features is None:
            self.gloss_features = np.load(self.gloss_feature_path, mmap_mode="r")
        return self.gloss_features

    def _it_array(self) -> np.ndarray:
        if self.it_features is None:
            self.it_features = np.load(self.it_feature_path, mmap_mode="r")
        return self.it_features

    def __len__(self) -> int:
        return self.size

    def _pick_gloss_segment(
        self, token: str, rng: np.random.Generator
    ) -> np.ndarray:
        choices = self.gloss_by_token[token]
        row = choices[int(rng.integers(len(choices)))]
        start, end = int(row["library_start"]), int(row["library_end"])
        return np.asarray(self._gloss_array()[start:end], dtype=np.float32)

    def _pick_it_segment(
        self, label: str, rng: np.random.Generator
    ) -> np.ndarray:
        choices = self.it_by_label[label]
        row = choices[int(rng.integers(len(choices)))]
        start, end = int(row["start"]), int(row["end"])
        return np.asarray(self._it_array()[start:end], dtype=np.float32)

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        general_tokens = list(self.patterns[int(rng.integers(len(self.patterns)))])
        if len(general_tokens) > self.max_general_glosses:
            start = int(rng.integers(len(general_tokens) - self.max_general_glosses + 1))
            general_tokens = general_tokens[start : start + self.max_general_glosses]

        items: list[tuple[str, str]] = [("general", token) for token in general_tokens]
        it_count = 2 if rng.random() < 0.25 else 1
        chosen_it = [
            self.it_labels[int(rng.integers(len(self.it_labels)))]
            for _ in range(it_count)
        ]
        for label in chosen_it:
            position = int(rng.integers(len(items) + 1))
            items.insert(position, ("it", label))

        tokens: list[str] = []
        segments: list[np.ndarray] = []
        for kind, token in items:
            segment = (
                self._pick_gloss_segment(token, rng)
                if kind == "general"
                else self._pick_it_segment(token, rng)
            )
            if self.augment:
                segment = augment_segment(segment, rng)
            tokens.append(token)
            segments.append(segment)

        base = concatenate_segments(segments, transition_frames=2)
        target = [self.token_to_id[token] for token in tokens]
        sample_id = f"synthetic_{self.epoch:03d}_{index:06d}"
        return (
            make_model_features(base, self.stride),
            torch.tensor(target, dtype=torch.long),
            sample_id,
            " ".join(tokens),
            "synthetic",
        )


def collate_batch(batch):
    sequences, targets, sample_ids, glosses, sources = zip(*batch)
    lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    target_lengths = torch.tensor([len(item) for item in targets], dtype=torch.long)
    max_length = int(lengths.max())
    padded = torch.zeros(len(batch), max_length, MODEL_FEATURES, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return (
        padded,
        lengths,
        torch.cat(targets),
        target_lengths,
        sample_ids,
        glosses,
        sources,
    )


def split_targets(targets: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    result: list[list[int]] = []
    offset = 0
    for raw_length in lengths.tolist():
        length = int(raw_length)
        result.append(targets[offset : offset + length].tolist())
        offset += length
    return result


def run_training_epoch(model, loader, loss_function, device, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_errors = total_tokens = batches = 0
    for padded, lengths, targets, target_lengths, _, _, _ in loader:
        padded, targets = padded.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, output_lengths = model(padded, lengths)
            loss = loss_function(
                logits.log_softmax(-1).transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        batches += 1
        hypotheses = greedy_decode(logits.detach(), output_lengths)
        references = split_targets(targets.detach().cpu(), target_lengths)
        for reference, hypothesis in zip(references, hypotheses):
            total_errors += edit_distance(reference, hypothesis)
            total_tokens += len(reference)
    return total_loss / max(batches, 1), total_errors / max(total_tokens, 1)


def evaluate(
    model,
    loader,
    loss_function,
    device,
    id_to_token,
    prediction_path: Path | None = None,
):
    model.eval()
    total_loss = 0.0
    total_errors = total_tokens = batches = exact = samples = 0
    top5_correct = top5_samples = 0
    prediction_rows: list[list[object]] = []

    with torch.no_grad():
        for padded, lengths, targets, target_lengths, sample_ids, glosses, sources in loader:
            logits, output_lengths = model(padded.to(device), lengths)
            loss = loss_function(
                logits.log_softmax(-1).transpose(0, 1),
                targets.to(device),
                output_lengths,
                target_lengths,
            )
            total_loss += float(loss.item())
            batches += 1
            hypotheses = greedy_decode(logits, output_lengths)
            references = split_targets(targets, target_lengths)
            log_probs = logits.log_softmax(-1).cpu()

            for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
                total_errors += edit_distance(reference, hypothesis)
                total_tokens += len(reference)
                exact += int(reference == hypothesis)
                samples += 1

                top5_tokens: list[int] = []
                if len(reference) == 1:
                    length = int(output_lengths[index])
                    token_scores = log_probs[index, :length].amax(dim=0)
                    token_scores[:2] = -torch.inf
                    top5_tokens = torch.topk(token_scores, k=min(5, len(id_to_token) - 2)).indices.tolist()
                    top5_correct += int(reference[0] in top5_tokens)
                    top5_samples += 1

                if prediction_path is not None:
                    prediction_rows.append(
                        [
                            sample_ids[index],
                            sources[index],
                            glosses[index],
                            " ".join(id_to_token[token] for token in hypothesis),
                            reference == hypothesis,
                            " | ".join(id_to_token[token] for token in top5_tokens),
                        ]
                    )

    if prediction_path is not None:
        with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "sample_id",
                    "source",
                    "reference_gloss",
                    "predicted_gloss",
                    "exact_match",
                    "top5_for_single_token",
                ]
            )
            writer.writerows(prediction_rows)

    return {
        "loss": total_loss / max(batches, 1),
        "wer": total_errors / max(total_tokens, 1),
        "exact_accuracy": exact / max(samples, 1),
        "top5_single_token_accuracy": (
            top5_correct / top5_samples if top5_samples else None
        ),
        "samples": samples,
        "tokens": total_tokens,
    }


def transfer_pretrained(
    model: ContinuousSignCTC,
    checkpoint: dict,
    new_vocabulary: list[str],
) -> int:
    old_state = checkpoint["model_state"]
    old_vocabulary = list(checkpoint["vocabulary"])
    new_state = model.state_dict()
    transferred = 0

    classifier_names = {"classifier.1.weight", "classifier.1.bias"}
    for name, value in old_state.items():
        if name in classifier_names:
            continue
        if name in new_state and new_state[name].shape == value.shape:
            new_state[name] = value
            transferred += 1

    new_token_to_id = {token: index for index, token in enumerate(new_vocabulary)}
    old_weight = old_state["classifier.1.weight"]
    old_bias = old_state["classifier.1.bias"]
    for old_id, token in enumerate(old_vocabulary):
        new_id = new_token_to_id.get(token)
        if new_id is None:
            continue
        new_state["classifier.1.weight"][new_id] = old_weight[old_id]
        new_state["classifier.1.bias"][new_id] = old_bias[old_id]
    transferred += 2
    model.load_state_dict(new_state)
    return transferred


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def main() -> int:
    args = arguments()
    if args.it_repeat < 1 or args.synthetic_train_samples < 1:
        raise ValueError("IT repeat and synthetic sample counts must be positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    isharah_data = args.isharah_data.resolve()
    gloss_library = args.gloss_library.resolve()
    unified_data = args.unified_data.resolve()
    pretrained_path = args.pretrained.resolve()
    output_dir = args.output.resolve()
    required = [
        isharah_data / "features_flat.npy",
        isharah_data / "samples.csv",
        gloss_library / "features_flat.npy",
        gloss_library / "clips.csv",
        unified_data / "it_features_flat.npy",
        unified_data / "it_samples.csv",
        unified_data / "unified_vocabulary.json",
        pretrained_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    id_to_token = load_json(unified_data / "unified_vocabulary.json")
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    isharah_rows = read_csv(isharah_data / "samples.csv")
    gloss_rows = read_csv(gloss_library / "clips.csv")
    it_rows = read_csv(unified_data / "it_samples.csv")

    datasets = {
        "isharah_train": IsharahDataset(
            isharah_data, isharah_rows, "train", token_to_id, args.stride
        ),
        "isharah_dev": IsharahDataset(
            isharah_data, isharah_rows, "dev", token_to_id, args.stride
        ),
        "isharah_test": IsharahDataset(
            isharah_data, isharah_rows, "test", token_to_id, args.stride
        ),
        "it_train": ITDataset(
            unified_data, it_rows, "train", args.stride, True, args.seed
        ),
        "it_validation": ITDataset(
            unified_data, it_rows, "validation", args.stride, False, args.seed
        ),
        "it_test": ITDataset(
            unified_data, it_rows, "test", args.stride, False, args.seed
        ),
    }
    synthetic_train = SyntheticMixedDataset(
        isharah_rows,
        gloss_library,
        gloss_rows,
        unified_data,
        it_rows,
        "train",
        "train",
        args.synthetic_train_samples,
        token_to_id,
        args.stride,
        args.max_general_glosses,
        args.seed + 10_000,
        True,
    )
    synthetic_dev = SyntheticMixedDataset(
        isharah_rows,
        gloss_library,
        gloss_rows,
        unified_data,
        it_rows,
        "dev",
        "validation",
        args.synthetic_dev_samples,
        token_to_id,
        args.stride,
        args.max_general_glosses,
        args.seed + 20_000,
        False,
    )

    train_dataset = ConcatDataset(
        [
            datasets["isharah_train"],
            RepeatDataset(datasets["it_train"], args.it_repeat),
            synthetic_train,
        ]
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers
    )
    validation_loaders = {
        "isharah": make_loader(
            datasets["isharah_dev"], args.batch_size, False, args.num_workers
        ),
        "it": make_loader(
            datasets["it_validation"], args.batch_size, False, args.num_workers
        ),
        "synthetic": make_loader(
            synthetic_dev, args.batch_size, False, args.num_workers
        ),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousSignCTC(MODEL_FEATURES, len(id_to_token)).to(device)
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
    transferred = transfer_pretrained(model, checkpoint, id_to_token)

    # Fail fast on incompatible files or sequence shapes before a long run.
    probe = collate_batch(
        [
            datasets["isharah_train"][0],
            datasets["it_train"][0],
            synthetic_train[0],
            synthetic_dev[0],
        ]
    )
    probe_padded, probe_lengths, _, probe_target_lengths, _, _, _ = probe
    model.eval()
    with torch.no_grad():
        probe_logits, probe_output_lengths = model(
            probe_padded.to(device), probe_lengths
        )
    if probe_logits.shape[-1] != len(id_to_token):
        raise RuntimeError("Preflight vocabulary dimension mismatch")
    if torch.any(probe_output_lengths < probe_target_lengths):
        raise RuntimeError("A preflight sequence is too short for its CTC target")
    print("Preflight check: passed")

    classifier_parameters = list(model.classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    encoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in classifier_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.learning_rate * 0.25},
            {"params": classifier_parameters, "lr": args.learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_function = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("Unified continuous SignBridge CTC training")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Vocabulary:", len(id_to_token), "including blank and unknown")
    print("Isharah train/dev/test:", *(len(datasets[f"isharah_{s}"]) for s in ("train", "dev", "test")))
    print("IT train/validation/test:", len(datasets["it_train"]), len(datasets["it_validation"]), len(datasets["it_test"]))
    print("Synthetic train/dev:", len(synthetic_train), len(synthetic_dev))
    print("Effective training samples per epoch:", len(train_dataset))
    print("Transferred tensors:", transferred)
    print("Parameters:", f"{sum(parameter.numel() for parameter in model.parameters()):,}")
    print("Untouched tests used during training: False")

    history_path = output_dir / "training_history.csv"
    best_path = output_dir / "best_unified_ctc.pt"
    best_score = float("inf")
    stale = 0
    started = time.perf_counter()

    with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_wer",
                "isharah_dev_wer",
                "it_validation_wer",
                "it_validation_accuracy",
                "it_validation_top5",
                "synthetic_dev_wer",
                "selection_score",
                "encoder_lr",
                "classifier_lr",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            datasets["it_train"].set_epoch(epoch)
            synthetic_train.set_epoch(epoch)
            synthetic_dev.set_epoch(0)
            train_loss, train_wer = run_training_epoch(
                model, train_loader, loss_function, device, optimizer, scaler
            )
            metrics = {
                name: evaluate(
                    model, loader, loss_function, device, id_to_token
                )
                for name, loader in validation_loaders.items()
            }
            selection_score = (
                0.45 * metrics["isharah"]["wer"]
                + 0.35 * metrics["it"]["wer"]
                + 0.20 * metrics["synthetic"]["wer"]
            )
            scheduler.step(selection_score)
            encoder_lr = optimizer.param_groups[0]["lr"]
            classifier_lr = optimizer.param_groups[1]["lr"]
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_wer,
                    metrics["isharah"]["wer"],
                    metrics["it"]["wer"],
                    metrics["it"]["exact_accuracy"],
                    metrics["it"]["top5_single_token_accuracy"],
                    metrics["synthetic"]["wer"],
                    selection_score,
                    encoder_lr,
                    classifier_lr,
                ]
            )
            handle.flush()
            print(
                f"Epoch {epoch:03d} | train WER {train_wer:.4f} | "
                f"Isharah dev WER {metrics['isharah']['wer']:.4f} | "
                f"IT val WER {metrics['it']['wer']:.4f} | "
                f"IT val acc {metrics['it']['exact_accuracy']:.4f} | "
                f"mixed dev WER {metrics['synthetic']['wer']:.4f} | "
                f"score {selection_score:.4f}"
            )

            if selection_score < best_score - 1e-4:
                best_score = selection_score
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocabulary": id_to_token,
                        "input_features": MODEL_FEATURES,
                        "shared_features": SHARED_FEATURES,
                        "stride": args.stride,
                        "epoch": epoch,
                        "selection_score": selection_score,
                        "validation_metrics": metrics,
                        "pretrained_checkpoint": str(pretrained_path),
                        "test_used_during_training": False,
                    },
                    best_path,
                )
                print("Best unified model updated.")
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping activated.")
                    break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    print("\nLoading best validation checkpoint...")
    print("Evaluating untouched Isharah and Jordanian signer-independent tests...")

    test_loaders = {
        "isharah": make_loader(
            datasets["isharah_test"], args.batch_size, False, args.num_workers
        ),
        "it": make_loader(
            datasets["it_test"], args.batch_size, False, args.num_workers
        ),
    }
    test_metrics = {
        "isharah": evaluate(
            model,
            test_loaders["isharah"],
            loss_function,
            device,
            id_to_token,
            output_dir / "isharah_test_predictions.csv",
        ),
        "it": evaluate(
            model,
            test_loaders["it"],
            loss_function,
            device,
            id_to_token,
            output_dir / "it_test_predictions.csv",
        ),
    }
    summary = {
        "best_epoch": best_checkpoint["epoch"],
        "best_selection_score": best_checkpoint["selection_score"],
        "validation": best_checkpoint["validation_metrics"],
        "test": test_metrics,
        "vocabulary_including_special_tokens": len(id_to_token),
        "real_sign_tokens": len(id_to_token) - 2,
        "training_sources": {
            "isharah_real": len(datasets["isharah_train"]),
            "jordanian_it_real_before_repeat": len(datasets["it_train"]),
            "jordanian_it_repeat": args.it_repeat,
            "synthetic_mixed": len(synthetic_train),
        },
        "test_policy": (
            "Both tests remained untouched until selection of the best validation checkpoint"
        ),
        "synthetic_data_limitation": (
            "Mixed-question validation is synthetic; final real mixed signing requires Deaf-user evaluation"
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\nUnified training and evaluation completed.")
    print("Best epoch:", summary["best_epoch"])
    print("Best validation score:", f"{summary['best_selection_score']:.4f}")
    print("Isharah test WER:", f"{test_metrics['isharah']['wer']:.4f}")
    print("IT test accuracy:", f"{test_metrics['it']['exact_accuracy']:.4f}")
    print("IT test top-5:", f"{test_metrics['it']['top5_single_token_accuracy']:.4f}")
    print("IT test WER:", f"{test_metrics['it']['wer']:.4f}")
    print("Elapsed seconds:", f"{summary['elapsed_seconds']:.1f}")
    print("Results:", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
