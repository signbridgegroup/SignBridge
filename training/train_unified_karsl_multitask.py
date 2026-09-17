"""Train the final SignBridge multi-task recognizer.

The script preserves the already-trained Isharah + Jordanian IT CTC model and
adds KArSL-502 without assuming that equally written regional glosses have the
same motion.  KArSL classes therefore receive source-aware tokens such as
KARSL_0001.  A frozen, signer-independent KArSL isolated model acts as a
representation teacher while a shared model learns:

* CTC on real Isharah sentences and a small amount of mixed synthetic data.
* CTC + isolated classification on Jordanian IT clips.
* CTC + isolated classification + embedding distillation on KArSL clips.

All three test sets remain untouched until the best validation checkpoint is
selected.

Required sibling scripts:
  train_unified_sign_ctc.py
  train_karsl_isolated_pretrain.py

Run first:
  python train_unified_karsl_multitask.py --preflight-only

Then train:
  python train_unified_karsl_multitask.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler

from train_karsl_isolated_pretrain import (
    IsolatedSignModel,
    speed_perturb as augment_karsl_tensor,
    build_metadata as build_karsl_metadata,
    validate_inputs as validate_karsl_inputs,
)
from train_unified_sign_ctc import (
    BLANK_ID,
    MODEL_FEATURES,
    SHARED_FEATURES,
    ContinuousSignCTC,
    IsharahDataset,
    SyntheticMixedDataset,
    augment_segment,
    collate_batch,
    edit_distance,
    greedy_decode,
    load_json,
    make_model_features,
    read_csv,
    split_targets,
)


PROJECT_ROOT = Path(r"C:\SignBridge_Project")
ISHARAH_DATA = PROJECT_ROOT / "data" / "processed" / "isharah1000_shared198"
GLOSS_LIBRARY = PROJECT_ROOT / "data" / "processed" / "isharah_gloss_clip_library"
UNIFIED_DATA = PROJECT_ROOT / "data" / "processed" / "unified_sign_data"
KARSL_FEATURES = PROJECT_ROOT / "data" / "processed" / "karsl_shared198"
KARSL_PREPARED = PROJECT_ROOT / "data" / "processed" / "unified_sign_data_karsl"
OLD_UNIFIED_MODEL = PROJECT_ROOT / "data" / "processed" / "unified_ctc_training" / "best_unified_ctc.pt"
KARSL_MODEL = PROJECT_ROOT / "data" / "processed" / "karsl_isolated_training" / "best_karsl_isolated_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "unified_karsl_multitask_training"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final SignBridge KArSL multi-task model")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--continuous-batch-size", type=int, default=8)
    parser.add_argument("--karsl-batch-size", type=int, default=128)
    parser.add_argument("--it-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--synthetic-train-samples", type=int, default=1500)
    parser.add_argument("--synthetic-dev-samples", type=int, default=250)
    parser.add_argument("--karsl-samples-per-epoch", type=int, default=12000)
    parser.add_argument("--it-samples-per-epoch", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def output_length(lengths: torch.Tensor) -> torch.Tensor:
    return torch.div(lengths + 1, 2, rounding_mode="floor")


class UnifiedMultiTaskModel(nn.Module):
    """The proven CTC backbone plus source-specific isolated heads."""

    def __init__(self, vocabulary_size: int, karsl_classes: int, it_classes: int) -> None:
        super().__init__()
        width = 256
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(MODEL_FEATURES),
            nn.Linear(MODEL_FEATURES, width),
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
        self.ctc_classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(512, vocabulary_size))
        self.karsl_head = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.20), nn.Linear(512, karsl_classes)
        )
        self.it_head = nn.Sequential(
            nn.Linear(1024, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.20), nn.Linear(384, it_classes)
        )
        self.karsl_embedding_projection = nn.Sequential(
            nn.Linear(1024, 384), nn.LayerNorm(384), nn.GELU(), nn.Linear(384, 256)
        )

    def encode(self, values: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_encoder(values)
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        out_lengths = output_length(lengths)
        packed = pack_padded_sequence(x, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.sequence_encoder(packed)
        encoded, _ = pad_packed_sequence(packed, batch_first=True)
        return encoded, out_lengths

    @staticmethod
    def pool(encoded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(encoded.device)
        positions = torch.arange(encoded.shape[1], device=encoded.device)[None, :]
        mask = positions < lengths[:, None]
        masked = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        mean = masked.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        return torch.cat([mean, maximum], dim=1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, out_lengths = self.encode(values, lengths)
        return self.ctc_classifier(encoded), out_lengths, encoded

    def isolated_logits(self, encoded: torch.Tensor, lengths: torch.Tensor, source: str) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(encoded, lengths)
        if source == "karsl":
            return self.karsl_head(pooled), self.karsl_embedding_projection(pooled)
        if source == "it":
            return self.it_head(pooled), pooled
        raise ValueError(f"Unknown isolated source: {source}")


def load_old_ctc_weights(
    model: UnifiedMultiTaskModel,
    checkpoint: dict,
    new_vocabulary: List[str],
) -> int:
    old_state = checkpoint["model_state"]
    new_state = model.state_dict()
    transferred = 0
    for name, value in old_state.items():
        if name.startswith("classifier."):
            continue
        if name in new_state and new_state[name].shape == value.shape:
            new_state[name] = value
            transferred += 1

    old_vocabulary = list(checkpoint["vocabulary"])
    new_token_to_id = {token: index for index, token in enumerate(new_vocabulary)}
    old_weight = old_state["classifier.1.weight"]
    old_bias = old_state["classifier.1.bias"]
    for old_id, token in enumerate(old_vocabulary):
        new_id = new_token_to_id[token]
        new_state["ctc_classifier.1.weight"][new_id] = old_weight[old_id]
        new_state["ctc_classifier.1.bias"][new_id] = old_bias[old_id]
    transferred += 2
    model.load_state_dict(new_state)
    return transferred


class ITMultiTaskDataset(Dataset):
    def __init__(
        self,
        unified_dir: Path,
        rows: List[dict[str, str]],
        split: str,
        stride: int,
        label_to_class: Dict[str, int],
        augment: bool,
        seed: int,
    ) -> None:
        self.feature_path = unified_dir / "it_features_flat.npy"
        self.rows = [row for row in rows if row["split"] == split]
        self.stride = stride
        self.label_to_class = label_to_class
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.features: Optional[np.ndarray] = None

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
        base = np.asarray(self._array()[int(row["start"]): int(row["end"])], dtype=np.float32)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            base = augment_segment(base, rng)
        return (
            make_model_features(base, self.stride),
            int(row["unified_token_id"]),
            self.label_to_class[row["label"]],
            str(row["video"]),
            str(row["label"]),
        )


def collate_it(batch):
    sequences, token_ids, class_ids, sample_ids, labels = zip(*batch)
    lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    padded = torch.zeros(len(batch), int(lengths.max()), MODEL_FEATURES, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    return (
        padded,
        lengths,
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(class_ids, dtype=torch.long),
        list(sample_ids),
        list(labels),
    )


class KArSLRawDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_root: Path,
        lengths: np.ndarray,
        offsets: np.ndarray,
        class_to_index: Dict[str, int],
        class_to_token: Dict[str, int],
        stride: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.feature_path = feature_root / "features_flat.npy"
        self.lengths = lengths
        self.offsets = offsets
        self.class_to_index = class_to_index
        self.class_to_token = class_to_token
        self.stride = stride
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.features: Optional[np.ndarray] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _array(self) -> np.ndarray:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        return self.features

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        feature_row = int(row["feature_row"])
        start = int(self.offsets[feature_row])
        length = int(self.lengths[feature_row])
        base = np.array(self._array()[start: start + length], dtype=np.float32, copy=True)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            scale = float(rng.uniform(0.88, 1.12))
            tensor = augment_karsl_tensor(torch.from_numpy(base), scale)
            base = tensor.numpy()
            if rng.random() < 0.50:
                base[:, :-2] += rng.normal(0.0, 0.007, base[:, :-2].shape).astype(np.float32)
        sign_id = str(row["sign_id"])
        return (
            make_model_features(base, self.stride),
            torch.from_numpy(base),
            self.class_to_token[sign_id],
            self.class_to_index[sign_id],
            str(row["sample_id"]),
            sign_id,
        )


def collate_karsl(batch, mean: torch.Tensor, std: torch.Tensor):
    student_sequences, raw_sequences, token_ids, class_ids, sample_ids, sign_ids = zip(*batch)
    student_lengths = torch.tensor([len(item) for item in student_sequences], dtype=torch.long)
    raw_lengths = torch.tensor([len(item) for item in raw_sequences], dtype=torch.long)
    student_padded = torch.zeros(len(batch), int(student_lengths.max()), MODEL_FEATURES, dtype=torch.float32)
    teacher_padded = torch.zeros(len(batch), int(raw_lengths.max()), SHARED_FEATURES, dtype=torch.float32)
    for index, (student, raw) in enumerate(zip(student_sequences, raw_sequences)):
        student_padded[index, : len(student)] = student
        normalized = (raw.float() - mean) / std
        teacher_padded[index, : len(raw)] = normalized
    return (
        student_padded,
        student_lengths,
        teacher_padded,
        raw_lengths,
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(class_ids, dtype=torch.long),
        list(sample_ids),
        list(sign_ids),
    )


def make_loader(dataset, batch_size, shuffle, workers, collate, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate,
    )


def ctc_loss_for_single_tokens(
    logits: torch.Tensor,
    output_lengths: torch.Tensor,
    token_ids: torch.Tensor,
    loss_function: nn.CTCLoss,
) -> torch.Tensor:
    target_lengths = torch.ones(len(token_ids), dtype=torch.long, device=token_ids.device)
    return loss_function(
        logits.log_softmax(-1).transpose(0, 1),
        token_ids,
        output_lengths,
        target_lengths,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    lengths: torch.Tensor,
    common_vocabulary: int,
    temperature: float = 2.0,
) -> torch.Tensor:
    time_steps = min(student_logits.shape[1], teacher_logits.shape[1])
    student = student_logits[:, :time_steps, :common_vocabulary]
    teacher = teacher_logits[:, :time_steps, :common_vocabulary]
    positions = torch.arange(time_steps, device=student.device)[None, :]
    lengths = lengths.to(student.device)
    mask = positions < lengths.clamp_max(time_steps)[:, None]
    values = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return (values * mask).sum() / mask.sum().clamp_min(1) * temperature * temperature


def macro_metrics(targets: np.ndarray, predictions: np.ndarray, classes: int) -> Dict[str, float]:
    precision_values: List[float] = []
    recall_values: List[float] = []
    f1_values: List[float] = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
    }


def train_continuous_batch(
    model, teacher, batch, ctc_loss, optimizer, scaler, device, common_vocabulary
) -> Tuple[float, int, int]:
    padded, lengths, targets, target_lengths, _, _, _ = batch
    padded = padded.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, out_lengths, _ = model(padded, lengths)
        primary = ctc_loss(logits.log_softmax(-1).transpose(0, 1), targets, out_lengths, target_lengths)
        with torch.no_grad():
            teacher_logits, teacher_lengths = teacher(padded, lengths)
        distill = distillation_loss(logits, teacher_logits, torch.minimum(out_lengths, teacher_lengths), common_vocabulary)
        loss = primary + 0.12 * distill
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    hypotheses = greedy_decode(logits.detach(), out_lengths)
    references = split_targets(targets.detach().cpu(), target_lengths)
    errors = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    tokens = sum(len(r) for r in references)
    return float(loss.detach()), errors, tokens


def train_it_batch(
    model, teacher, batch, ctc_loss, ce_loss, optimizer, scaler, device, common_vocabulary
) -> float:
    padded, lengths, token_ids, class_ids, _, _ = batch
    padded = padded.to(device, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)
    class_ids = class_ids.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, out_lengths, encoded = model(padded, lengths)
        isolated_logits, _ = model.isolated_logits(encoded, out_lengths, "it")
        ctc = ctc_loss_for_single_tokens(logits, out_lengths, token_ids, ctc_loss)
        classification = ce_loss(isolated_logits, class_ids)
        with torch.no_grad():
            teacher_logits, teacher_lengths = teacher(padded, lengths)
        distill = distillation_loss(logits, teacher_logits, torch.minimum(out_lengths, teacher_lengths), common_vocabulary)
        loss = ctc + 0.50 * classification + 0.10 * distill
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


def train_karsl_batch(
    model, karsl_teacher, batch, ctc_loss, ce_loss, optimizer, scaler, device
) -> float:
    student_padded, student_lengths, teacher_padded, raw_lengths, token_ids, class_ids, _, _ = batch
    student_padded = student_padded.to(device, non_blocking=True)
    teacher_padded = teacher_padded.to(device, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)
    class_ids = class_ids.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits, out_lengths, encoded = model(student_padded, student_lengths)
        isolated_logits, student_embedding = model.isolated_logits(encoded, out_lengths, "karsl")
        classification = ce_loss(isolated_logits, class_ids)
        ctc = ctc_loss_for_single_tokens(logits, out_lengths, token_ids, ctc_loss)
        with torch.no_grad():
            _, teacher_embedding = karsl_teacher(teacher_padded, raw_lengths.to(device))
        representation = 1.0 - F.cosine_similarity(student_embedding, teacher_embedding, dim=1).mean()
        loss = classification + 0.35 * ctc + 0.15 * representation
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


@torch.no_grad()
def evaluate_ctc(model, loader, loss_function, device, id_to_token, prediction_path=None):
    model.eval()
    total_loss = 0.0
    errors = tokens = exact = samples = top5_correct = top5_samples = batches = 0
    rows: List[List[object]] = []
    for padded, lengths, targets, target_lengths, sample_ids, glosses, sources in loader:
        logits, out_lengths, _ = model(padded.to(device), lengths)
        loss = loss_function(logits.log_softmax(-1).transpose(0, 1), targets.to(device), out_lengths, target_lengths)
        total_loss += float(loss)
        batches += 1
        hypotheses = greedy_decode(logits, out_lengths)
        references = split_targets(targets, target_lengths)
        log_probs = logits.log_softmax(-1).cpu()
        for index, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
            errors += edit_distance(reference, hypothesis)
            tokens += len(reference)
            exact += int(reference == hypothesis)
            samples += 1
            top5_tokens: List[int] = []
            if len(reference) == 1:
                token_scores = log_probs[index, : int(out_lengths[index])].amax(dim=0)
                token_scores[:2] = -torch.inf
                top5_tokens = torch.topk(token_scores, k=5).indices.tolist()
                top5_correct += int(reference[0] in top5_tokens)
                top5_samples += 1
            if prediction_path is not None:
                rows.append([
                    sample_ids[index], sources[index], glosses[index],
                    " ".join(id_to_token[x] for x in hypothesis), reference == hypothesis,
                    " | ".join(id_to_token[x] for x in top5_tokens),
                ])
    if prediction_path is not None:
        with Path(prediction_path).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "source", "reference_gloss", "predicted_gloss", "exact_match", "top5"])
            writer.writerows(rows)
    return {
        "loss": total_loss / max(batches, 1), "wer": errors / max(tokens, 1),
        "exact_accuracy": exact / max(samples, 1),
        "top5_single_token_accuracy": top5_correct / max(top5_samples, 1) if top5_samples else None,
        "samples": samples, "tokens": tokens,
    }


@torch.no_grad()
def evaluate_it(model, loader, ctc_loss, device):
    model.eval()
    targets: List[int] = []
    predictions: List[int] = []
    ctc_exact = ctc_top5 = samples = 0
    for padded, lengths, token_ids, class_ids, _, _ in loader:
        logits, out_lengths, encoded = model(padded.to(device), lengths)
        isolated_logits, _ = model.isolated_logits(encoded, out_lengths, "it")
        predictions.extend(isolated_logits.argmax(1).cpu().tolist())
        targets.extend(class_ids.tolist())
        log_probs = logits.log_softmax(-1).cpu()
        for index, token in enumerate(token_ids.tolist()):
            scores = log_probs[index, : int(out_lengths[index])].amax(0)
            scores[:2] = -torch.inf
            ranked = torch.topk(scores, 5).indices.tolist()
            ctc_exact += int(ranked[0] == token)
            ctc_top5 += int(token in ranked)
            samples += 1
    metrics = macro_metrics(np.asarray(targets), np.asarray(predictions), model.it_head[-1].out_features)
    metrics["ctc_exact_accuracy"] = ctc_exact / max(samples, 1)
    metrics["ctc_top5_accuracy"] = ctc_top5 / max(samples, 1)
    metrics["ctc_wer"] = 1.0 - metrics["ctc_exact_accuracy"]
    return metrics


@torch.no_grad()
def evaluate_karsl(model, loader, device, prediction_path=None):
    model.eval()
    targets: List[int] = []
    predictions: List[int] = []
    ctc_exact = ctc_top5 = samples = 0
    rows: List[List[object]] = []
    for student_padded, student_lengths, _, _, token_ids, class_ids, sample_ids, sign_ids in loader:
        logits, out_lengths, encoded = model(student_padded.to(device), student_lengths)
        class_logits, _ = model.isolated_logits(encoded, out_lengths, "karsl")
        class_predictions = class_logits.argmax(1).cpu()
        targets.extend(class_ids.tolist())
        predictions.extend(class_predictions.tolist())
        log_probs = logits.log_softmax(-1).cpu()
        for index, token in enumerate(token_ids.tolist()):
            scores = log_probs[index, : int(out_lengths[index])].amax(0)
            scores[:2] = -torch.inf
            ranked = torch.topk(scores, 5).indices.tolist()
            ctc_exact += int(ranked[0] == token)
            ctc_top5 += int(token in ranked)
            samples += 1
            if prediction_path is not None:
                rows.append([
                    sample_ids[index], sign_ids[index], int(class_predictions[index]),
                    int(class_predictions[index]) == int(class_ids[index]), ranked[0] == token, token in ranked,
                ])
    metrics = macro_metrics(np.asarray(targets), np.asarray(predictions), model.karsl_head[-1].out_features)
    metrics["ctc_exact_accuracy"] = ctc_exact / max(samples, 1)
    metrics["ctc_top5_accuracy"] = ctc_top5 / max(samples, 1)
    if prediction_path is not None:
        with Path(prediction_path).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "expected_sign_id", "predicted_class_index", "classification_correct", "ctc_correct", "ctc_top5_correct"])
            writer.writerows(rows)
    return metrics


def interleave_auxiliary(
    main_index: int,
    main_total: int,
    auxiliary_index: int,
    auxiliary_total: int,
) -> bool:
    if auxiliary_index >= auxiliary_total:
        return False
    return math.floor((main_index + 1) * auxiliary_total / main_total) > math.floor(main_index * auxiliary_total / main_total)


def main() -> int:
    args = arguments()
    seed_everything(args.seed)
    root = args.project_root.resolve()
    isharah_dir = root / "data" / "processed" / "isharah1000_shared198"
    gloss_dir = root / "data" / "processed" / "isharah_gloss_clip_library"
    unified_dir = root / "data" / "processed" / "unified_sign_data"
    karsl_dir = root / "data" / "processed" / "karsl_shared198"
    karsl_prepared = root / "data" / "processed" / "unified_sign_data_karsl"
    old_checkpoint_path = root / "data" / "processed" / "unified_ctc_training" / "best_unified_ctc.pt"
    karsl_checkpoint_path = root / "data" / "processed" / "karsl_isolated_training" / "best_karsl_isolated_model.pt"
    output_dir = root / "data" / "processed" / "unified_karsl_multitask_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        isharah_dir / "features_flat.npy", isharah_dir / "samples.csv",
        gloss_dir / "features_flat.npy", gloss_dir / "clips.csv",
        unified_dir / "it_features_flat.npy", unified_dir / "it_samples.csv",
        old_checkpoint_path, karsl_checkpoint_path,
        karsl_dir / "features_flat.npy", karsl_dir / "samples.csv",
        karsl_prepared / "karsl_samples.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_checkpoint = torch.load(old_checkpoint_path, map_location=device, weights_only=False)
    karsl_checkpoint = torch.load(karsl_checkpoint_path, map_location=device, weights_only=False)
    base_vocabulary = list(old_checkpoint["vocabulary"])

    karsl_metadata = build_karsl_metadata(karsl_dir, karsl_prepared)
    _, karsl_lengths, karsl_offsets = validate_karsl_inputs(karsl_dir, karsl_metadata)
    karsl_classes = list(karsl_checkpoint["index_to_class"])
    karsl_class_to_index = {str(key): int(value) for key, value in karsl_checkpoint["class_to_index"].items()}
    if sorted(karsl_metadata["sign_id"].unique()) != sorted(karsl_class_to_index):
        raise ValueError("KArSL metadata classes do not match the isolated checkpoint")
    karsl_tokens = [f"KARSL_{sign_id}" for sign_id in karsl_classes]
    full_vocabulary = base_vocabulary + karsl_tokens
    karsl_class_to_token = {
        sign_id: len(base_vocabulary) + index for index, sign_id in enumerate(karsl_classes)
    }

    isharah_rows = read_csv(isharah_dir / "samples.csv")
    gloss_rows = read_csv(gloss_dir / "clips.csv")
    it_rows = read_csv(unified_dir / "it_samples.csv")
    it_labels = sorted({row["label"] for row in it_rows}, key=str.casefold)
    it_label_to_class = {label: index for index, label in enumerate(it_labels)}
    token_to_id = {token: index for index, token in enumerate(full_vocabulary)}
    # IT token IDs were created with the base vocabulary and therefore remain stable.

    isharah_datasets = {
        split: IsharahDataset(isharah_dir, isharah_rows, split, token_to_id, args.stride)
        for split in ("train", "dev", "test")
    }
    it_datasets = {
        split: ITMultiTaskDataset(
            unified_dir, it_rows, split, args.stride, it_label_to_class,
            augment=split == "train", seed=args.seed + 1000,
        )
        for split in ("train", "validation", "test")
    }
    karsl_frames = {
        split: karsl_metadata[karsl_metadata["split"] == split].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }
    karsl_datasets = {
        split: KArSLRawDataset(
            karsl_frames[split], karsl_dir, karsl_lengths, karsl_offsets,
            karsl_class_to_index, karsl_class_to_token, args.stride,
            augment=split == "train", seed=args.seed + 2000,
        )
        for split in ("train", "validation", "test")
    }
    synthetic_train = SyntheticMixedDataset(
        isharah_rows, gloss_dir, gloss_rows, unified_dir, it_rows,
        "train", "train", args.synthetic_train_samples, token_to_id,
        args.stride, 8, args.seed + 3000, True,
    )
    synthetic_dev = SyntheticMixedDataset(
        isharah_rows, gloss_dir, gloss_rows, unified_dir, it_rows,
        "dev", "validation", args.synthetic_dev_samples, token_to_id,
        args.stride, 8, args.seed + 4000, False,
    )

    mean = torch.as_tensor(karsl_checkpoint["normalization_mean"], dtype=torch.float32)
    std = torch.as_tensor(karsl_checkpoint["normalization_std"], dtype=torch.float32)
    karsl_collate = partial(collate_karsl, mean=mean, std=std)

    model = UnifiedMultiTaskModel(len(full_vocabulary), len(karsl_classes), len(it_labels)).to(device)
    transferred = load_old_ctc_weights(model, old_checkpoint, full_vocabulary)
    old_teacher = ContinuousSignCTC(MODEL_FEATURES, len(base_vocabulary)).to(device)
    old_teacher.load_state_dict(old_checkpoint["model_state"])
    old_teacher.eval()
    for parameter in old_teacher.parameters():
        parameter.requires_grad = False

    karsl_teacher = IsolatedSignModel(len(karsl_classes)).to(device)
    karsl_teacher.load_state_dict(karsl_checkpoint["model_state"])
    karsl_teacher.eval()
    for parameter in karsl_teacher.parameters():
        parameter.requires_grad = False

    continuous_train = ConcatDataset([isharah_datasets["train"], synthetic_train])
    loaders = {
        "continuous_train": make_loader(continuous_train, args.continuous_batch_size, True, args.num_workers, collate_batch),
        "isharah_dev": make_loader(isharah_datasets["dev"], args.continuous_batch_size, False, args.num_workers, collate_batch),
        "isharah_test": make_loader(isharah_datasets["test"], args.continuous_batch_size, False, args.num_workers, collate_batch),
        "synthetic_dev": make_loader(synthetic_dev, args.continuous_batch_size, False, args.num_workers, collate_batch),
        "it_validation": make_loader(it_datasets["validation"], args.it_batch_size, False, args.num_workers, collate_it),
        "it_test": make_loader(it_datasets["test"], args.it_batch_size, False, args.num_workers, collate_it),
        "karsl_validation": make_loader(karsl_datasets["validation"], args.karsl_batch_size, False, args.num_workers, karsl_collate),
        "karsl_test": make_loader(karsl_datasets["test"], args.karsl_batch_size, False, args.num_workers, karsl_collate),
    }

    # Fail before a long run if any source or checkpoint is incompatible.
    continuous_probe = next(iter(loaders["continuous_train"]))
    it_probe = next(iter(loaders["it_validation"]))
    karsl_probe = next(iter(loaders["karsl_validation"]))
    model.eval()
    with torch.no_grad():
        c_logits, c_lengths, _ = model(continuous_probe[0].to(device), continuous_probe[1])
        i_logits, i_lengths, i_encoded = model(it_probe[0].to(device), it_probe[1])
        k_logits, k_lengths, k_encoded = model(karsl_probe[0].to(device), karsl_probe[1])
        i_class, _ = model.isolated_logits(i_encoded, i_lengths, "it")
        k_class, _ = model.isolated_logits(k_encoded, k_lengths, "karsl")
    if c_logits.shape[-1] != len(full_vocabulary) or i_class.shape[-1] != len(it_labels) or k_class.shape[-1] != 502:
        raise RuntimeError("Preflight output dimension mismatch")
    if torch.any(c_lengths < continuous_probe[3]):
        raise RuntimeError("A continuous preflight sequence is too short for CTC")

    print("Preflight check: passed")
    print("Final SignBridge multi-task training")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Base vocabulary:", len(base_vocabulary))
    print("Source-aware KArSL tokens:", len(karsl_tokens))
    print("Final CTC vocabulary:", len(full_vocabulary))
    print("Isharah train/dev/test:", *(len(isharah_datasets[x]) for x in ("train", "dev", "test")))
    print("IT train/validation/test:", *(len(it_datasets[x]) for x in ("train", "validation", "test")))
    print("KArSL train/validation/test:", *(len(karsl_datasets[x]) for x in ("train", "validation", "test")))
    print("Synthetic proportion:", f"{len(synthetic_train) / len(continuous_train):.1%}")
    print("Transferred tensors from old unified CTC:", transferred)
    print("Untouched tests used during training: False")
    if args.preflight_only:
        return 0

    ctc_loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.06)
    pretrained_names = ("input_encoder", "temporal_conv", "sequence_encoder")
    pretrained_parameters = []
    new_parameters = []
    for name, parameter in model.named_parameters():
        (pretrained_parameters if name.startswith(pretrained_names) else new_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": pretrained_parameters, "lr": args.learning_rate * 0.30},
            {"params": new_parameters, "lr": args.learning_rate},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history_path = output_dir / "training_history.csv"
    best_path = output_dir / "best_unified_karsl_multitask.pt"
    best_score = float("inf")
    stale = 0
    started = time.perf_counter()
    history_fields = [
        "epoch", "train_loss", "train_wer", "isharah_dev_wer", "it_val_ctc_accuracy",
        "it_val_classification_accuracy", "karsl_val_ctc_accuracy", "karsl_val_classification_accuracy",
        "karsl_val_macro_f1", "synthetic_dev_wer", "selection_score", "encoder_lr", "head_lr",
    ]
    with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            synthetic_train.set_epoch(epoch)
            it_datasets["train"].set_epoch(epoch)
            karsl_datasets["train"].set_epoch(epoch)
            karsl_sampler = RandomSampler(
                karsl_datasets["train"], replacement=False,
                num_samples=min(args.karsl_samples_per_epoch, len(karsl_datasets["train"])),
                generator=torch.Generator().manual_seed(args.seed + epoch),
            )
            it_sampler = RandomSampler(
                it_datasets["train"], replacement=True, num_samples=args.it_samples_per_epoch,
                generator=torch.Generator().manual_seed(args.seed + 10_000 + epoch),
            )
            karsl_train_loader = make_loader(
                karsl_datasets["train"], args.karsl_batch_size, False, args.num_workers, karsl_collate, karsl_sampler
            )
            it_train_loader = make_loader(
                it_datasets["train"], args.it_batch_size, False, args.num_workers, collate_it, it_sampler
            )
            karsl_iterator = iter(karsl_train_loader)
            it_iterator = iter(it_train_loader)
            karsl_done = it_done = 0
            losses: List[float] = []
            errors = tokens = 0
            model.train()
            main_total = len(loaders["continuous_train"])
            for main_index, batch in enumerate(loaders["continuous_train"]):
                loss, batch_errors, batch_tokens = train_continuous_batch(
                    model, old_teacher, batch, ctc_loss, optimizer, scaler, device, len(base_vocabulary)
                )
                losses.append(loss)
                errors += batch_errors
                tokens += batch_tokens
                if interleave_auxiliary(main_index, main_total, karsl_done, len(karsl_train_loader)):
                    losses.append(train_karsl_batch(
                        model, karsl_teacher, next(karsl_iterator), ctc_loss, ce_loss, optimizer, scaler, device
                    ))
                    karsl_done += 1
                if interleave_auxiliary(main_index, main_total, it_done, len(it_train_loader)):
                    losses.append(train_it_batch(
                        model, old_teacher, next(it_iterator), ctc_loss, ce_loss, optimizer, scaler, device, len(base_vocabulary)
                    ))
                    it_done += 1

            isharah_metrics = evaluate_ctc(model, loaders["isharah_dev"], ctc_loss, device, full_vocabulary)
            synthetic_metrics = evaluate_ctc(model, loaders["synthetic_dev"], ctc_loss, device, full_vocabulary)
            it_metrics = evaluate_it(model, loaders["it_validation"], ctc_loss, device)
            karsl_metrics = evaluate_karsl(model, loaders["karsl_validation"], device)
            selection_score = (
                0.45 * isharah_metrics["wer"]
                + 0.25 * (1.0 - it_metrics["ctc_exact_accuracy"])
                + 0.20 * (1.0 - karsl_metrics["accuracy"])
                + 0.10 * synthetic_metrics["wer"]
            )
            scheduler.step(selection_score)
            record = {
                "epoch": epoch, "train_loss": float(np.mean(losses)), "train_wer": errors / max(tokens, 1),
                "isharah_dev_wer": isharah_metrics["wer"], "it_val_ctc_accuracy": it_metrics["ctc_exact_accuracy"],
                "it_val_classification_accuracy": it_metrics["accuracy"],
                "karsl_val_ctc_accuracy": karsl_metrics["ctc_exact_accuracy"],
                "karsl_val_classification_accuracy": karsl_metrics["accuracy"],
                "karsl_val_macro_f1": karsl_metrics["macro_f1"], "synthetic_dev_wer": synthetic_metrics["wer"],
                "selection_score": selection_score, "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
            writer.writerow(record)
            handle.flush()
            print(
                f"Epoch {epoch:03d} | Isharah dev WER {isharah_metrics['wer']:.4f} | "
                f"IT val CTC acc {it_metrics['ctc_exact_accuracy']:.4f} | "
                f"KArSL val CTC/class {karsl_metrics['ctc_exact_accuracy']:.4f}/{karsl_metrics['accuracy']:.4f} | "
                f"mixed dev WER {synthetic_metrics['wer']:.4f} | score {selection_score:.4f}"
            )
            if selection_score < best_score - 1e-4:
                best_score = selection_score
                stale = 0
                torch.save({
                    "model_state": model.state_dict(), "vocabulary": full_vocabulary,
                    "base_vocabulary_size": len(base_vocabulary), "karsl_tokens": karsl_tokens,
                    "karsl_class_to_index": karsl_class_to_index, "it_label_to_class": it_label_to_class,
                    "epoch": epoch, "selection_score": selection_score,
                    "validation": {"isharah": isharah_metrics, "it": it_metrics, "karsl": karsl_metrics, "synthetic": synthetic_metrics},
                    "stride": args.stride, "input_features": MODEL_FEATURES, "test_used_during_training": False,
                }, best_path)
                print("Best multi-task model updated.")
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping activated.")
                    break

    print("Loading best validation checkpoint...")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    print("Evaluating all untouched signer-independent tests...")
    tests = {
        "isharah": evaluate_ctc(
            model, loaders["isharah_test"], ctc_loss, device, full_vocabulary,
            output_dir / "isharah_test_predictions.csv",
        ),
        "it": evaluate_it(model, loaders["it_test"], ctc_loss, device),
        "karsl": evaluate_karsl(
            model, loaders["karsl_test"], device, output_dir / "karsl_test_predictions.csv"
        ),
    }
    summary = {
        "best_epoch": best_checkpoint["epoch"], "best_selection_score": best_checkpoint["selection_score"],
        "validation": best_checkpoint["validation"], "test": tests,
        "base_vocabulary": len(base_vocabulary), "source_aware_karsl_tokens": 502,
        "final_vocabulary": len(full_vocabulary), "synthetic_train_samples": len(synthetic_train),
        "test_policy": "Isharah, Jordanian IT, and signer-03 KArSL tests were untouched until final model selection",
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "unified_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(full_vocabulary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "karsl_token_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump({token: sign_id for token, sign_id in zip(karsl_tokens, karsl_classes)}, handle, ensure_ascii=False, indent=2)

    print("Multi-task training completed.")
    print("Best epoch:", best_checkpoint["epoch"])
    print("Isharah test WER:", f"{tests['isharah']['wer']:.4f}")
    print("IT test CTC accuracy:", f"{tests['it']['ctc_exact_accuracy']:.4f}")
    print("KArSL signer-03 classification accuracy:", f"{tests['karsl']['accuracy']:.4f}")
    print("KArSL signer-03 CTC accuracy:", f"{tests['karsl']['ctc_exact_accuracy']:.4f}")
    print("Results:", output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
