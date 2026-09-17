"""Train a signer-independent isolated-sign encoder on KArSL-502.

This stage learns a strong ordinary-word visual encoder before it is transferred
to SignBridge's continuous multi-task model.  The untouched signer-03 split is
evaluated only after model selection is complete.

Expected inputs (created by the SignBridge preparation pipeline):
  data/processed/karsl_shared198/features_flat.npy
  data/processed/karsl_shared198/lengths.npy
  data/processed/karsl_shared198/offsets.npy
  data/processed/karsl_shared198/samples.csv
  data/processed/unified_sign_data_karsl/karsl_samples.csv

Run:
  python train_karsl_isolated_pretrain.py --preflight-only
  python train_karsl_isolated_pretrain.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(r"C:\SignBridge_Project")
FEATURE_ROOT = PROJECT_ROOT / "data" / "processed" / "karsl_shared198"
PREPARED_ROOT = PROJECT_ROOT / "data" / "processed" / "unified_sign_data_karsl"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "karsl_isolated_training"

EXPECTED_FEATURES = 198
EXPECTED_CLASSES = 502
EXPECTED_SPLITS = {"train": 42387, "validation": 8040, "test": 25087}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_column(frame: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(
            f"Could not find any of columns {list(candidates)}. Available columns: {list(frame.columns)}"
        )
    return None


def normalize_split(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "dev": "validation",
        "valid": "validation",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }
    if text not in aliases:
        raise ValueError(f"Unknown split value: {value!r}")
    return aliases[text]


def normalize_sign_id(value: object) -> str:
    text = str(value).strip()
    try:
        return f"{int(float(text)):04d}"
    except ValueError:
        return text


def build_metadata(feature_root: Path, prepared_root: Path) -> pd.DataFrame:
    source_path = feature_root / "samples.csv"
    prepared_path = prepared_root / "karsl_samples.csv"
    source = pd.read_csv(source_path, dtype=str)
    prepared = pd.read_csv(prepared_path, dtype=str)
    source["_feature_row"] = np.arange(len(source), dtype=np.int64)

    key_candidates = ["sample_id", "video_id", "relative_path", "path", "video_path", "file"]
    source_key = find_column(source, key_candidates, required=False)
    prepared_key = find_column(prepared, key_candidates, required=False)

    if source_key is not None and prepared_key is not None:
        source_keys = source[[source_key, "_feature_row"]].copy()
        source_keys[source_key] = source_keys[source_key].astype(str)
        if source_keys[source_key].duplicated().any():
            raise ValueError(f"Duplicate source key values found in {source_key}")
        metadata = prepared.merge(
            source_keys,
            left_on=prepared_key,
            right_on=source_key,
            how="left",
            validate="one_to_one",
            suffixes=("", "_source"),
        )
        if metadata["_feature_row"].isna().any():
            missing = int(metadata["_feature_row"].isna().sum())
            raise ValueError(f"Could not match {missing} prepared samples to extracted features")
        metadata["_feature_row"] = metadata["_feature_row"].astype(np.int64)
    elif len(source) == len(prepared):
        metadata = prepared.copy()
        metadata["_feature_row"] = np.arange(len(prepared), dtype=np.int64)
    else:
        # Some preparation versions remove only invalid training items.  Look for
        # an explicit source row/index before refusing an unsafe positional join.
        index_col = find_column(
            prepared,
            ["feature_row", "source_index", "source_row", "feature_index", "sample_index"],
            required=False,
        )
        if index_col is None:
            raise ValueError(
                "The prepared and feature metadata have different row counts, and no shared "
                "sample key or source index was found."
            )
        metadata = prepared.copy()
        metadata["_feature_row"] = pd.to_numeric(metadata[index_col], errors="raise").astype(np.int64)

    split_col = find_column(metadata, ["split", "model_split", "target_split", "data_split"])
    label_col = find_column(metadata, ["sign_id", "karsl_sign_id", "class_id", "label_id", "label"])
    sample_col = find_column(
        metadata,
        ["sample_id", "video_id", "relative_path", "path", "video_path", "file"],
        required=False,
    )

    result = pd.DataFrame(
        {
            "feature_row": metadata["_feature_row"].astype(np.int64),
            "split": metadata[split_col].map(normalize_split),
            "sign_id": metadata[label_col].map(normalize_sign_id),
            "sample_id": (
                metadata[sample_col].astype(str)
                if sample_col is not None
                else metadata["_feature_row"].map(lambda x: f"sample_{int(x):06d}")
            ),
        }
    )
    if result["feature_row"].duplicated().any():
        raise ValueError("A feature row is assigned more than once")
    return result.reset_index(drop=True)


def validate_inputs(feature_root: Path, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.load(feature_root / "features_flat.npy", mmap_mode="r")
    lengths = np.load(feature_root / "lengths.npy", mmap_mode="r")
    offsets = np.load(feature_root / "offsets.npy", mmap_mode="r")

    if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Expected feature shape (frames, {EXPECTED_FEATURES}), got {features.shape}")
    if len(lengths) != len(offsets):
        raise ValueError("lengths.npy and offsets.npy have different row counts")
    if metadata.empty:
        raise ValueError("No KArSL samples were found")
    if metadata["feature_row"].min() < 0 or metadata["feature_row"].max() >= len(lengths):
        raise ValueError("Metadata contains an out-of-range feature row")

    split_counts = metadata["split"].value_counts().to_dict()
    for split in ("train", "validation", "test"):
        if split_counts.get(split, 0) == 0:
            raise ValueError(f"The {split} split is empty")

    classes = sorted(metadata["sign_id"].unique().tolist())
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} sign IDs, found {len(classes)}")

    selected_lengths = np.asarray(lengths[metadata["feature_row"].to_numpy()], dtype=np.int64)
    selected_offsets = np.asarray(offsets[metadata["feature_row"].to_numpy()], dtype=np.int64)
    if np.any(selected_lengths <= 0):
        raise ValueError("At least one selected sample has zero frames")
    if np.any(selected_offsets < 0) or np.any(selected_offsets + selected_lengths > len(features)):
        raise ValueError("At least one feature slice is outside features_flat.npy")

    # The preparation step intentionally excludes four impossible training clips.
    # Report a mismatch, but do not reject a correctly regenerated dataset.
    for split, expected in EXPECTED_SPLITS.items():
        actual = split_counts.get(split, 0)
        if actual != expected:
            print(f"NOTE: {split} has {actual:,} samples (previous preparation reported {expected:,}).")

    return features, lengths, offsets


def compute_train_statistics(
    feature_root: Path,
    metadata: pd.DataFrame,
    lengths: np.ndarray,
    offsets: np.ndarray,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        mean = cached["mean"].astype(np.float32)
        std = cached["std"].astype(np.float32)
        if mean.shape == (EXPECTED_FEATURES,) and std.shape == (EXPECTED_FEATURES,):
            return mean, std

    features = np.load(feature_root / "features_flat.npy", mmap_mode="r")
    rows = metadata.loc[metadata["split"] == "train", "feature_row"].to_numpy(np.int64)
    total = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    total_sq = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    frame_count = 0
    print("Computing training-only normalization statistics...")
    for number, row in enumerate(rows, start=1):
        start = int(offsets[row])
        length = int(lengths[row])
        block = np.asarray(features[start : start + length], dtype=np.float64)
        total += block.sum(axis=0)
        total_sq += np.square(block).sum(axis=0)
        frame_count += length
        if number % 5000 == 0:
            print(f"Statistics: {number:,}/{len(rows):,} videos")

    mean = total / max(frame_count, 1)
    variance = np.maximum(total_sq / max(frame_count, 1) - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    # Observed/missing flags and nearly constant dimensions should not be amplified.
    std = np.maximum(std, 1e-3)
    np.savez(cache_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def speed_perturb(sequence: torch.Tensor, factor: float) -> torch.Tensor:
    old_length = sequence.shape[0]
    new_length = max(5, int(round(old_length * factor)))
    if new_length == old_length:
        return sequence
    values = sequence.transpose(0, 1).unsqueeze(0)
    values = F.interpolate(values, size=new_length, mode="linear", align_corners=False)
    return values.squeeze(0).transpose(0, 1)


class KArSLDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_path: Path,
        lengths: np.ndarray,
        offsets: np.ndarray,
        class_to_index: Dict[str, int],
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.feature_path = str(feature_path)
        self.lengths = np.asarray(lengths)
        self.offsets = np.asarray(offsets)
        self.class_to_index = class_to_index
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.augment = augment
        self._features: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_features"] = None
        return state

    def _open_features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        record = self.rows.iloc[index]
        row = int(record["feature_row"])
        start = int(self.offsets[row])
        length = int(self.lengths[row])
        array = np.array(self._open_features()[start : start + length], dtype=np.float32, copy=True)
        sequence = torch.from_numpy(array)
        sequence = (sequence - self.mean) / self.std

        if self.augment:
            if random.random() < 0.65:
                sequence = speed_perturb(sequence, random.uniform(0.85, 1.15))
            if len(sequence) >= 12 and random.random() < 0.35:
                trim = random.randint(0, max(1, int(len(sequence) * 0.08)))
                left = random.randint(0, trim)
                right = trim - left
                sequence = sequence[left : len(sequence) - right if right else None]
            if random.random() < 0.50:
                sequence = sequence + torch.randn_like(sequence) * random.uniform(0.002, 0.012)
            if len(sequence) >= 10 and random.random() < 0.30:
                width = random.randint(1, max(1, int(len(sequence) * 0.08)))
                first = random.randint(0, len(sequence) - width)
                sequence[first : first + width] = 0.0

        label = self.class_to_index[str(record["sign_id"])]
        return sequence, label, str(record["sample_id"])


def collate_batch(batch: Sequence[Tuple[torch.Tensor, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    sequences, labels, sample_ids = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    padded = torch.zeros(len(sequences), max_length, sequences[0].shape[1], dtype=torch.float32)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence
    return padded, lengths, torch.tensor(labels, dtype=torch.long), list(sample_ids)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dimension: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(dimension, dimension, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        x = self.conv1(values.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(F.gelu(self.norm1(x)))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(self.norm2(residual + self.dropout(x)))


class KArSLTemporalEncoder(nn.Module):
    def __init__(self, input_size: int = EXPECTED_FEATURES, model_size: int = 256, hidden_size: int = 192) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, model_size),
            nn.LayerNorm(model_size),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.temporal_blocks = nn.ModuleList(
            [ResidualTemporalBlock(model_size, 5, 0.12), ResidualTemporalBlock(model_size, 3, 0.12)]
        )
        self.recurrent = nn.GRU(
            input_size=model_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.output_norm = nn.LayerNorm(hidden_size * 2)
        self.output_size = hidden_size * 2

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(values)
        for block in self.temporal_blocks:
            x = block(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.recurrent(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=values.shape[1])
        return self.output_norm(x)


class IsolatedSignModel(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.encoder = KArSLTemporalEncoder()
        size = self.encoder.output_size
        self.attention = nn.Sequential(nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 1))
        self.embedding = nn.Sequential(
            nn.Linear(size * 2, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.classifier = nn.Linear(256, classes)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encoder(values, lengths)
        positions = torch.arange(sequence.shape[1], device=sequence.device)[None, :]
        mask = positions < lengths[:, None]
        scores = self.attention(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        attentive = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        mean = torch.sum(sequence * mask.unsqueeze(-1), dim=1) / lengths.clamp_min(1).unsqueeze(1)
        embedding = self.embedding(torch.cat([attentive, mean], dim=1))
        return self.classifier(embedding), embedding


def class_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = np.sqrt(counts.mean() / np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.5, 2.0)
    return torch.tensor(weights, dtype=torch.float32)


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, classes: int) -> Dict[str, float]:
    correct = targets == predictions
    accuracy = float(correct.mean()) if len(targets) else 0.0
    recalls = []
    precisions = []
    f1_values = []
    for label in range(classes):
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    item_count = 0
    targets_all: List[np.ndarray] = []
    predictions_all: List[np.ndarray] = []
    top5_correct = 0

    for values, lengths, targets, _ in loader:
        values = values.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(values, lengths)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

        batch_size = len(targets)
        loss_sum += float(loss.detach()) * batch_size
        item_count += batch_size
        predictions = logits.argmax(dim=1)
        top5 = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
        top5_correct += int((top5 == targets[:, None]).any(dim=1).sum())
        targets_all.append(targets.detach().cpu().numpy())
        predictions_all.append(predictions.detach().cpu().numpy())

    targets_np = np.concatenate(targets_all)
    predictions_np = np.concatenate(predictions_all)
    metrics = calculate_metrics(targets_np, predictions_np, model.classifier.out_features)
    metrics["loss"] = loss_sum / max(item_count, 1)
    metrics["top5"] = top5_correct / max(item_count, 1)
    return metrics


@torch.no_grad()
def predict_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    index_to_class: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    rows: List[dict] = []
    targets_all: List[int] = []
    predictions_all: List[int] = []
    top5_correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for values, lengths, targets, sample_ids in loader:
        values = values.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(values, lengths)
            loss = criterion(logits, targets)
        probabilities = torch.softmax(logits.float(), dim=1)
        top_values, top_indices = probabilities.topk(5, dim=1)
        predictions = top_indices[:, 0]
        loss_sum += float(loss) * len(targets)
        top5_correct += int((top_indices == targets[:, None]).any(dim=1).sum())
        targets_all.extend(targets.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())

        for i, sample_id in enumerate(sample_ids):
            row = {
                "sample_id": sample_id,
                "expected_sign_id": index_to_class[int(targets[i])],
                "predicted_sign_id": index_to_class[int(predictions[i])],
                "correct": bool(predictions[i] == targets[i]),
            }
            for rank in range(5):
                row[f"top{rank + 1}_sign_id"] = index_to_class[int(top_indices[i, rank])]
                row[f"top{rank + 1}_probability"] = float(top_values[i, rank])
            rows.append(row)

    targets_np = np.asarray(targets_all, dtype=np.int64)
    predictions_np = np.asarray(predictions_all, dtype=np.int64)
    metrics = calculate_metrics(targets_np, predictions_np, len(index_to_class))
    metrics["loss"] = loss_sum / max(len(targets_np), 1)
    metrics["top5"] = top5_correct / max(len(targets_np), 1)
    return pd.DataFrame(rows), metrics


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    feature_root = args.project_root / "data" / "processed" / "karsl_shared198"
    prepared_root = args.project_root / "data" / "processed" / "unified_sign_data_karsl"
    output_root = args.project_root / "data" / "processed" / "karsl_isolated_training"
    output_root.mkdir(parents=True, exist_ok=True)

    print("KArSL-502 isolated signer-independent pretraining")
    metadata = build_metadata(feature_root, prepared_root)
    features, lengths, offsets = validate_inputs(feature_root, metadata)
    split_counts = metadata["split"].value_counts().to_dict()
    classes = sorted(metadata["sign_id"].unique().tolist())
    class_to_index = {label: index for index, label in enumerate(classes)}

    print(f"Samples: {len(metadata):,}")
    print(f"Features: {features.shape}")
    print(f"Classes: {len(classes)}")
    print(
        "Train/validation/test: "
        f"{split_counts.get('train', 0):,} / {split_counts.get('validation', 0):,} / "
        f"{split_counts.get('test', 0):,}"
    )
    print("Untouched signer-independent test: enabled")
    if args.preflight_only:
        first = metadata.iloc[0]
        row = int(first["feature_row"])
        sample = np.asarray(features[int(offsets[row]) : int(offsets[row]) + int(lengths[row])])
        if not np.isfinite(sample).all():
            raise ValueError("The first feature sequence contains non-finite values")
        print("Preflight check: passed")
        return

    mean, std = compute_train_statistics(
        feature_root,
        metadata,
        lengths,
        offsets,
        output_root / "karsl_train_normalization.npz",
    )
    train_frame = metadata[metadata["split"] == "train"].reset_index(drop=True)
    validation_frame = metadata[metadata["split"] == "validation"].reset_index(drop=True)
    test_frame = metadata[metadata["split"] == "test"].reset_index(drop=True)

    datasets = {
        "train": KArSLDataset(
            train_frame,
            feature_root / "features_flat.npy",
            lengths,
            offsets,
            class_to_index,
            mean,
            std,
            augment=True,
        ),
        "validation": KArSLDataset(
            validation_frame,
            feature_root / "features_flat.npy",
            lengths,
            offsets,
            class_to_index,
            mean,
            std,
            augment=False,
        ),
        "test": KArSLDataset(
            test_frame,
            feature_root / "features_flat.npy",
            lengths,
            offsets,
            class_to_index,
            mean,
            std,
            augment=False,
        ),
    }
    generator = torch.Generator().manual_seed(args.seed)
    common_loader = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
    )
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=generator, **common_loader)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **common_loader)
    test_loader = DataLoader(datasets["test"], shuffle=False, **common_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = IsolatedSignModel(len(classes)).to(device)
    train_labels = train_frame["sign_id"].map(class_to_index).to_numpy(np.int64)
    weights = class_weights(train_labels, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.08)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * 0.05))

    def learning_rate(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trainable parameters: {parameter_count:,}")
    print("Starting isolated-sign pretraining...")

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[dict] = []
    checkpoint_path = output_root / "best_karsl_isolated_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler, scaler
        )
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        score = 0.5 * validation_metrics["accuracy"] + 0.5 * validation_metrics["macro_f1"]
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "validation_score": score,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_root / "training_history.csv", index=False)
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | "
            f"train acc {train_metrics['accuracy']:.4f} | val loss {validation_metrics['loss']:.4f} | "
            f"val acc {validation_metrics['accuracy']:.4f} | val top-5 {validation_metrics['top5']:.4f} | "
            f"val F1 {validation_metrics['macro_f1']:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "stage": "karsl_isolated_pretraining",
                    "epoch": epoch,
                    "validation_score": score,
                    "validation_metrics": validation_metrics,
                    "model_state": model.state_dict(),
                    "encoder_state": model.encoder.state_dict(),
                    "class_to_index": class_to_index,
                    "index_to_class": classes,
                    "feature_count": EXPECTED_FEATURES,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "architecture": {
                        "input_size": EXPECTED_FEATURES,
                        "model_size": 256,
                        "gru_hidden_size": 192,
                        "gru_layers": 2,
                        "encoder_output_size": model.encoder.output_size,
                    },
                },
                checkpoint_path,
            )
            print("Best isolated model updated.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping activated.")
                break

    print("Loading best validation checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    # The signer-independent test is touched exactly once, after model selection.
    print("Evaluating untouched signer-independent KArSL test...")
    predictions, test_metrics = predict_test(model, test_loader, device, classes)
    predictions.to_csv(output_root / "test_predictions.csv", index=False)
    torch.save(
        {
            "stage": "karsl_isolated_encoder",
            "source_checkpoint": str(checkpoint_path),
            "encoder_state": model.encoder.state_dict(),
            "feature_count": EXPECTED_FEATURES,
            "normalization_mean": mean,
            "normalization_std": std,
            "architecture": checkpoint["architecture"],
        },
        output_root / "karsl_isolated_encoder.pt",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "test": test_metrics,
        "samples": {key: int(value) for key, value in split_counts.items()},
        "classes": len(classes),
        "features_per_frame": EXPECTED_FEATURES,
        "parameters": parameter_count,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "test_protocol": "untouched signer-independent signer 03",
    }
    with open(output_root / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("KArSL isolated pretraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test top-5 accuracy: {test_metrics['top5']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Transfer encoder: {output_root / 'karsl_isolated_encoder.pt'}")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
