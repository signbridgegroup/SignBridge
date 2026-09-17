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
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA = Path(r"C:\SignBridge_Project\data\processed\isharah1000_shared198")
DEFAULT_OUTPUT = Path(r"C:\SignBridge_Project\data\processed\isharah_ctc_training")
BLANK = "<blank>"
UNK = "<unk>"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a continuous Isharah CTC model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
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


def create_vocabulary(rows: list[dict[str, str]]) -> tuple[list[str], dict[str, int]]:
    # Only training glosses define the vocabulary; dev/test OOV maps to <unk>.
    tokens = sorted(
        {token for row in rows if row["split"] == "train" for token in row["gloss"].split()}
    )
    id_to_token = [BLANK, UNK, *tokens]
    return id_to_token, {token: index for index, token in enumerate(id_to_token)}


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

    def __len__(self) -> int:
        return len(self.rows)

    def _array(self) -> np.ndarray:
        if self.features is None:
            self.features = np.load(self.feature_path, mmap_mode="r")
        return self.features

    def __getitem__(self, index: int):
        row = self.rows[index]
        start, end = int(row["start"]), int(row["end"])
        base = np.asarray(self._array()[start:end:self.stride], dtype=np.float32)
        if len(base) == 0:
            raise ValueError(f"Empty sequence: {row['sample_id']}")

        # Motion is useful for distinguishing signs with similar hand shapes.
        velocity = np.zeros_like(base)
        velocity[1:] = base[1:] - base[:-1]
        features = np.concatenate([base, velocity], axis=1)

        target = [self.token_to_id.get(token, 1) for token in row["gloss"].split()]
        return (
            torch.from_numpy(features),
            torch.tensor(target, dtype=torch.long),
            row["sample_id"],
            row["gloss"],
        )


def collate_batch(batch):
    sequences, targets, sample_ids, glosses = zip(*batch)
    lengths = torch.tensor([len(item) for item in sequences], dtype=torch.long)
    target_lengths = torch.tensor([len(item) for item in targets], dtype=torch.long)
    max_length = int(lengths.max())
    feature_count = sequences[0].shape[1]
    padded = torch.zeros(len(batch), max_length, feature_count, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = sequence
    concatenated_targets = torch.cat(targets)
    return padded, lengths, concatenated_targets, target_lengths, sample_ids, glosses


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
        self.classifier = nn.Sequential(nn.Dropout(0.20), nn.Linear(512, vocabulary_size))

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        # Conv1d(k=5, stride=2, padding=2) output length.
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
    decoded = []
    for sequence, length in zip(predictions, lengths.cpu()):
        result, previous = [], None
        for token in sequence[: int(length)].tolist():
            if token != 0 and token != previous:
                result.append(token)
            previous = token
        decoded.append(result)
    return decoded


def split_targets(targets: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    result, offset = [], 0
    for length in lengths.tolist():
        result.append(targets[offset : offset + length].tolist())
        offset += length
    return result


def run_epoch(model, loader, loss_function, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_errors = total_tokens = 0
    batches = 0

    for padded, lengths, targets, target_lengths, _, _ in loader:
        padded, targets = padded.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, output_lengths = model(padded, lengths)
                log_probs = logits.log_softmax(-1).transpose(0, 1)
                loss = loss_function(log_probs, targets, output_lengths, target_lengths)

            if training:
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


def evaluate_and_write(model, loader, device, id_to_token, output_path):
    model.eval()
    errors = tokens = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "reference_gloss", "predicted_gloss"])
        with torch.no_grad():
            for padded, lengths, targets, target_lengths, sample_ids, glosses in loader:
                logits, output_lengths = model(padded.to(device), lengths)
                hypotheses = greedy_decode(logits, output_lengths)
                references = split_targets(targets, target_lengths)
                for sample_id, gloss, reference, hypothesis in zip(
                    sample_ids, glosses, references, hypotheses
                ):
                    errors += edit_distance(reference, hypothesis)
                    tokens += len(reference)
                    predicted = " ".join(id_to_token[token] for token in hypothesis)
                    writer.writerow([sample_id, gloss, predicted])
    return errors / max(tokens, 1)


def main() -> int:
    args = arguments()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data / "samples.csv")
    id_to_token, token_to_id = create_vocabulary(rows)

    with (args.output / "ctc_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(id_to_token, handle, ensure_ascii=False, indent=2)

    datasets = {
        split: IsharahDataset(args.data, rows, split, token_to_id, args.stride)
        for split in ("train", "dev", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousSignCTC(396, len(id_to_token)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print("Continuous Isharah CTC training")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Train/dev/test:", *(len(datasets[s]) for s in ("train", "dev", "test")))
    print("Vocabulary size:", len(id_to_token), "(including blank and unknown)")
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

    history_path = args.output / "training_history.csv"
    best_path = args.output / "best_isharah_ctc.pt"
    best_wer, stale = float("inf"), 0
    start_time = time.time()

    with history_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_wer", "dev_loss", "dev_wer", "lr"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_wer = run_epoch(
                model, loaders["train"], loss_function, device, optimizer, scaler
            )
            dev_loss, dev_wer = run_epoch(model, loaders["dev"], loss_function, device)
            scheduler.step(dev_wer)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, train_wer, dev_loss, dev_wer, lr])
            handle.flush()
            print(
                f"Epoch {epoch:03d} | train loss {train_loss:.4f} | "
                f"train WER {train_wer:.4f} | dev loss {dev_loss:.4f} | "
                f"dev WER {dev_wer:.4f} | LR {lr:.2e}"
            )

            if dev_wer < best_wer - 1e-4:
                best_wer, stale = dev_wer, 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocabulary": id_to_token,
                        "input_features": 396,
                        "shared_features": 198,
                        "stride": args.stride,
                        "epoch": epoch,
                        "dev_wer": dev_wer,
                    },
                    best_path,
                )
                print("Best model updated.")
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping activated.")
                    break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_wer = evaluate_and_write(
        model, loaders["test"], device, id_to_token, args.output / "test_predictions.csv"
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_dev_wer": checkpoint["dev_wer"],
        "test_wer": test_wer,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "train_samples": len(datasets["train"]),
        "dev_samples": len(datasets["dev"]),
        "test_samples": len(datasets["test"]),
        "vocabulary_size_including_special_tokens": len(id_to_token),
    }
    with (args.output / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Training completed.")
    print("Best dev WER:", f"{checkpoint['dev_wer']:.4f}")
    print("Test WER:", f"{test_wer:.4f}")
    print("Results:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
