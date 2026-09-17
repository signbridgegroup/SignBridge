from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(r"C:\SignBridge_Project\data\processed\model_data")
DEFAULT_PRETRAINED = Path(
    r"C:\SignBridge_Project\data\processed\isharah_ctc_training\best_isharah_ctc.pt"
)
DEFAULT_OUTPUT = ROOT / "training_results_transfer"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Isharah encoder on Jordanian IT signs")
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_indices(split_dir: Path, name: str) -> np.ndarray:
    path = split_dir / f"{name}_indices.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return np.load(path).astype(np.int64)


def to_shared198(sequence: np.ndarray) -> np.ndarray:
    """Select the exact 2D blocks used during Isharah pretraining."""
    pose = sequence[:, 0:42].reshape(len(sequence), 14, 3)[:, :, :2].reshape(len(sequence), -1)
    left_global = sequence[:, 42:105].reshape(len(sequence), 21, 3)[:, :, :2].reshape(len(sequence), -1)
    right_global = sequence[:, 105:168].reshape(len(sequence), 21, 3)[:, :, :2].reshape(len(sequence), -1)
    left_local = sequence[:, 444:507].reshape(len(sequence), 21, 3)[:, :, :2].reshape(len(sequence), -1)
    right_local = sequence[:, 507:570].reshape(len(sequence), 21, 3)[:, :, :2].reshape(len(sequence), -1)
    observed = sequence[:, 570:572]
    shared = np.concatenate(
        [pose, left_global, right_global, left_local, right_local, observed], axis=1
    ).astype(np.float32)
    if shared.shape[1] != 198:
        raise AssertionError(f"Expected 198 features, got {shared.shape[1]}")
    return shared


def valid_length(sequence: np.ndarray) -> int:
    # Prepared sequences are padded with all-zero frames at the end.
    valid = np.any(np.abs(sequence) > 1e-8, axis=1)
    locations = np.flatnonzero(valid)
    return int(locations[-1] + 1) if len(locations) else len(sequence)


class ITDataset(Dataset):
    def __init__(self, x, y, indices, training=False):
        self.x = x
        self.y = y
        self.indices = indices
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        original = np.asarray(self.x[index], dtype=np.float32)
        length = valid_length(original)
        shared = to_shared198(original[:length])

        if self.training:
            # Gentle augmentation preserves sign meaning while reducing overfitting.
            if random.random() < 0.5:
                shared[:, :196] += np.random.normal(0, 0.008, shared[:, :196].shape).astype(np.float32)
            if random.random() < 0.3 and len(shared) > 20:
                mask_start = random.randrange(1, len(shared) - 5)
                mask_size = random.randint(1, min(5, len(shared) - mask_start))
                shared[mask_start : mask_start + mask_size, :196] = 0.0

        velocity = np.zeros_like(shared)
        velocity[1:] = shared[1:] - shared[:-1]
        features = np.concatenate([shared, velocity], axis=1)
        return torch.from_numpy(features), int(self.y[index]), index


def collate(batch):
    sequences, labels, indices = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    padded = torch.zeros(len(batch), int(lengths.max()), 396, dtype=torch.float32)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence
    return padded, lengths, torch.tensor(labels, dtype=torch.long), indices


class TransferClassifier(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        width = 256
        self.input_encoder = nn.Sequential(
            nn.LayerNorm(396), nn.Linear(396, width), nn.GELU(), nn.Dropout(0.15)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.15)
        )
        self.sequence_encoder = nn.GRU(
            width, 256, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.20
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024), nn.Dropout(0.25),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(512, classes),
        )

    @staticmethod
    def output_lengths(lengths):
        return torch.div(lengths + 1, 2, rounding_mode="floor")

    def forward(self, x, lengths):
        x = self.input_encoder(x)
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        out_lengths = self.output_lengths(lengths)
        packed = pack_padded_sequence(x, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.sequence_encoder(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)

        time = torch.arange(x.shape[1], device=x.device)[None, :]
        mask = time < out_lengths.to(x.device)[:, None]
        mean_pool = (x * mask[:, :, None]).sum(1) / out_lengths.to(x.device)[:, None]
        max_pool = x.masked_fill(~mask[:, :, None], float("-inf")).max(1).values
        return self.classifier(torch.cat([mean_pool, max_pool], dim=1))

    def set_encoder_trainable(self, trainable: bool):
        for module in (self.input_encoder, self.temporal_conv, self.sequence_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = trainable


def macro_metrics(confusion: np.ndarray):
    tp = np.diag(confusion).astype(float)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros_like(tp), where=confusion.sum(0) != 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros_like(tp), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return float(precision.mean()), float(recall.mean()), float(f1.mean())


def run(model, loader, criterion, device, optimizer=None, scaler=None, classes=165):
    training = optimizer is not None
    model.train(training)
    total_loss = total = correct = top5_correct = 0
    confusion = np.zeros((classes, classes), dtype=np.int64)

    for padded, lengths, labels, _ in loader:
        padded, labels = padded.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(padded, lengths)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

        predictions = logits.argmax(1)
        top5 = logits.topk(min(5, classes), dim=1).indices
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
        correct += int((predictions == labels).sum())
        top5_correct += int((top5 == labels[:, None]).any(1).sum())
        for true, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            confusion[true, predicted] += 1

    precision, recall, f1 = macro_metrics(confusion)
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "top5": top5_correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": confusion,
    }


def load_labels(path: Path, classes: int):
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and "label_to_id" in raw:
        raw = raw["label_to_id"]
    elif isinstance(raw, dict) and "id_to_label" in raw:
        raw = raw["id_to_label"]
    if isinstance(raw, list):
        labels = raw
    elif all(str(key).isdigit() for key in raw):
        labels = [raw[str(i)] for i in range(classes)]
    else:
        labels = [None] * classes
        for label, index in raw.items():
            labels[int(index)] = label
    if len(labels) != classes or any(label is None for label in labels):
        raise ValueError("label_map.json does not align with y.npy")
    return labels


def main():
    args = parse_args()
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    split_dir = args.data / "splits_signer_independent"
    x = np.load(args.data / "X.npy", mmap_mode="r")
    y = np.load(args.data / "y.npy").astype(np.int64)
    classes = int(y.max()) + 1
    labels = load_labels(args.data / "label_map.json", classes)
    indices = {name: load_indices(split_dir, name) for name in ("train", "validation", "test")}

    datasets = {
        name: ITDataset(x, y, value, training=(name == "train"))
        for name, value in indices.items()
    }
    train_counts = np.bincount(y[indices["train"]], minlength=classes)
    sample_weights = 1.0 / train_counts[y[indices["train"]]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferClassifier(classes)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    source = checkpoint["model_state"]
    transferable = {
        key: value for key, value in source.items()
        if not key.startswith("classifier.") and key in model.state_dict()
        and model.state_dict()[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    encoder_keys = [key for key in model.state_dict() if not key.startswith("classifier.")]
    if any(key in missing for key in encoder_keys):
        raise RuntimeError("The pretrained encoder is not fully compatible")
    model.to(device)
    model.set_encoder_trainable(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=8e-4, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1, stale = -1.0, 0
    best_path = args.output / "best_transfer_model.pt"
    history_path = args.output / "training_history.csv"
    start_time = time.time()

    print("Jordanian IT transfer learning")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    print("X shape:", x.shape)
    print("Train/validation/test:", *(len(datasets[n]) for n in ("train", "validation", "test")))
    print("Classes:", classes)
    print("Transferred tensors:", len(transferable))

    with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "train_f1",
                         "validation_loss", "validation_accuracy", "validation_top5", "validation_f1"])
        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_epochs + 1:
                model.set_encoder_trainable(True)
                optimizer = torch.optim.AdamW([
                    {"params": model.input_encoder.parameters(), "lr": 3e-5},
                    {"params": model.temporal_conv.parameters(), "lr": 3e-5},
                    {"params": model.sequence_encoder.parameters(), "lr": 3e-5},
                    {"params": model.classifier.parameters(), "lr": 2e-4},
                ], weight_decay=1e-4)
                print("Encoder unfrozen for careful fine-tuning.")

            train = run(model, loaders["train"], criterion, device, optimizer, scaler, classes)
            validation = run(model, loaders["validation"], criterion, device, classes=classes)
            writer.writerow([epoch, train["loss"], train["accuracy"], train["f1"],
                             validation["loss"], validation["accuracy"], validation["top5"], validation["f1"]])
            handle.flush()
            print(f"Epoch {epoch:03d} | train loss {train['loss']:.4f} | train acc {train['accuracy']:.4f} | "
                  f"val loss {validation['loss']:.4f} | val acc {validation['accuracy']:.4f} | "
                  f"val top-5 {validation['top5']:.4f} | val F1 {validation['f1']:.4f}")

            if validation["f1"] > best_f1 + 1e-4:
                best_f1, stale = validation["f1"], 0
                torch.save({"model_state": model.state_dict(), "labels": labels,
                            "epoch": epoch, "validation_f1": best_f1,
                            "pretrained_checkpoint": str(args.pretrained)}, best_path)
                print("Best model updated.")
            else:
                stale += 1
                if epoch > args.freeze_epochs and stale >= args.patience:
                    print("Early stopping activated.")
                    break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test = run(model, loaders["test"], criterion, device, classes=classes)
    np.save(args.output / "test_confusion_matrix.npy", test.pop("confusion"))
    summary = {"best_epoch": best["epoch"], "best_validation_f1": best["validation_f1"],
               "test": test, "elapsed_seconds": round(time.time() - start_time, 1),
               "pretrained_dev_wer": checkpoint.get("dev_wer")}
    with (args.output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("Transfer training completed.")
    print("Best validation F1:", f"{best['validation_f1']:.4f}")
    print("Test accuracy:", f"{test['accuracy']:.4f}")
    print("Test top-5 accuracy:", f"{test['top5']:.4f}")
    print("Test macro F1:", f"{test['f1']:.4f}")
    print("Results:", args.output)


if __name__ == "__main__":
    main()
