#!/usr/bin/env python3
"""GPU experiment for automatic modulation classification.

The script generates synthetic baseband IQ fragments for eight modulation
classes, trains a compact 1D-CNN on raw I/Q samples, and saves the results
needed for the article:

- accuracy_by_snr.csv;
- accuracy_by_snr.png;
- training_curve.png;
- confusion_matrix.png;
- gpu_run_log.txt.

The default settings are intended for NVIDIA RTX 4060 Mobile 8 GB.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "4PAM", "2FSK", "GFSK"]
SNR_LEVELS = [-10, -6, -2, 2, 6, 10, 14, 18]
IQ_LENGTH = 128
SAMPLES_PER_SYMBOL = 4


@dataclass
class ExperimentConfig:
    seed: int = 20260508
    epochs: int = 30
    batch_size: int = 512
    train_per_class_snr: int = 900
    test_per_class_snr: int = 240
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    output_dir: str = "results"
    require_cuda: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def normalize_power(signal: np.ndarray) -> np.ndarray:
    return signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-12)


def generate_symbols(modulation: str, rng: np.random.Generator) -> np.ndarray:
    n_symbols = IQ_LENGTH // SAMPLES_PER_SYMBOL

    if modulation == "BPSK":
        symbols = rng.choice([-1, 1], size=n_symbols).astype(np.complex128)
        return np.repeat(symbols, SAMPLES_PER_SYMBOL)

    if modulation == "QPSK":
        phases = np.pi / 4 + rng.integers(0, 4, size=n_symbols) * np.pi / 2
        return np.repeat(np.exp(1j * phases), SAMPLES_PER_SYMBOL)

    if modulation == "8PSK":
        phases = rng.integers(0, 8, size=n_symbols) * 2 * np.pi / 8
        return np.repeat(np.exp(1j * phases), SAMPLES_PER_SYMBOL)

    if modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3])
        symbols = rng.choice(levels, size=n_symbols) + 1j * rng.choice(levels, size=n_symbols)
        symbols = normalize_power(symbols)
        return np.repeat(symbols, SAMPLES_PER_SYMBOL)

    if modulation == "64QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        symbols = rng.choice(levels, size=n_symbols) + 1j * rng.choice(levels, size=n_symbols)
        symbols = normalize_power(symbols)
        return np.repeat(symbols, SAMPLES_PER_SYMBOL)

    if modulation == "4PAM":
        levels = np.array([-3, -1, 1, 3])
        symbols = rng.choice(levels, size=n_symbols).astype(np.float64)
        symbols = symbols / (np.sqrt(np.mean(symbols**2)) + 1e-12)
        return np.repeat(symbols, SAMPLES_PER_SYMBOL).astype(np.complex128)

    if modulation == "2FSK":
        bits = rng.choice([-1, 1], size=n_symbols)
        inst_freq = np.repeat(bits, SAMPLES_PER_SYMBOL) * 0.070
        phase = 2 * np.pi * np.cumsum(inst_freq) + rng.uniform(0, 2 * np.pi)
        return np.exp(1j * phase)

    if modulation == "GFSK":
        bits = rng.choice([-1, 1], size=n_symbols)
        shaped = moving_average(np.repeat(bits, SAMPLES_PER_SYMBOL), width=7)
        inst_freq = shaped * 0.075
        phase = 2 * np.pi * np.cumsum(inst_freq) + rng.uniform(0, 2 * np.pi)
        return np.exp(1j * phase)

    raise ValueError(f"Unknown modulation: {modulation}")


def generate_iq(modulation: str, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Generate one noisy 2 x IQ_LENGTH I/Q example."""
    signal = generate_symbols(modulation, rng)
    signal = normalize_power(signal)

    time_index = np.arange(IQ_LENGTH)
    phase_offset = rng.uniform(0, 2 * np.pi)
    carrier_frequency_offset = rng.normal(loc=0.0, scale=0.004)
    signal = signal * np.exp(1j * (phase_offset + 2 * np.pi * carrier_frequency_offset * time_index))
    signal = normalize_power(signal)

    noise_power = 1.0 / (10 ** (snr_db / 10))
    noise_std = math.sqrt(noise_power / 2)
    noise = noise_std * (rng.normal(size=IQ_LENGTH) + 1j * rng.normal(size=IQ_LENGTH))
    noisy = normalize_power(signal + noise)

    iq = np.stack([noisy.real, noisy.imag]).astype(np.float32)
    return iq


def build_dataset(samples_per_class_snr: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    snr_rows: list[int] = []

    for snr in SNR_LEVELS:
        for label, modulation in enumerate(MODULATIONS):
            for _ in range(samples_per_class_snr):
                x_rows.append(generate_iq(modulation, snr, rng))
                y_rows.append(label)
                snr_rows.append(snr)

    x = torch.tensor(np.stack(x_rows), dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    snr_tensor = torch.tensor(snr_rows, dtype=torch.long)
    return x, y, snr_tensor


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class AMCNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            ResidualDepthwiseBlock(64),
            ResidualDepthwiseBlock(64),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            ResidualDepthwiseBlock(96),
            ResidualDepthwiseBlock(96),
            nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            ResidualDepthwiseBlock(128),
            ResidualDepthwiseBlock(128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[int, float], np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    by_snr: dict[int, list[int]] = {snr: [0, 0] for snr in SNR_LEVELS}
    confusion = np.zeros((len(MODULATIONS), len(MODULATIONS)), dtype=np.int64)

    for x_batch, y_batch, snr_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        predictions = logits.argmax(dim=1).cpu()
        y_cpu = y_batch.cpu()

        correct += int((predictions == y_cpu).sum())
        total += len(y_cpu)

        for true_label, predicted_label, snr in zip(y_cpu.tolist(), predictions.tolist(), snr_batch.tolist()):
            by_snr[int(snr)][0] += int(true_label == predicted_label)
            by_snr[int(snr)][1] += 1
            confusion[true_label, predicted_label] += 1

    accuracy = correct / total
    snr_accuracy = {snr: hits / count for snr, (hits, count) in by_snr.items() if count > 0}
    return accuracy, snr_accuracy, confusion


def save_accuracy_csv(path: Path, snr_accuracy: dict[int, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["snr_db", "accuracy"])
        for snr in SNR_LEVELS:
            writer.writerow([snr, f"{snr_accuracy[snr]:.6f}"])


def save_training_curve(path: Path, history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_accuracy = [row["test_accuracy"] for row in history]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax1.plot(epochs, train_loss, color="#2563eb", marker="o", linewidth=2, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, test_accuracy, color="#0f766e", marker="s", linewidth=2, label="Test accuracy")
    ax2.set_ylabel("Test accuracy")
    ax2.set_ylim(0, 1)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_accuracy_plot(path: Path, snr_accuracy: dict[int, float]) -> None:
    x = SNR_LEVELS
    y = [snr_accuracy[snr] for snr in x]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax.plot(x, y, color="#0f766e", marker="o", linewidth=2.5)
    ax.set_title("CNN modulation classification accuracy vs SNR")
    ax.set_xlabel("SNR, dB")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    for snr, acc in zip(x, y):
        ax.text(snr, min(acc + 0.035, 0.98), f"{acc:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix(path: Path, confusion: np.ndarray) -> None:
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = confusion / np.maximum(row_sums, 1)
    fig, ax = plt.subplots(figsize=(7.6, 6.7), dpi=160)
    image = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(MODULATIONS)))
    ax.set_yticks(range(len(MODULATIONS)))
    ax.set_xticklabels(MODULATIONS, rotation=45, ha="right")
    ax.set_yticklabels(MODULATIONS)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(MODULATIONS)):
        for j in range(len(MODULATIONS)):
            value = normalized[i, j]
            if value >= 0.15:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train 1D-CNN for synthetic AMC on GPU.")
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--train-per-class-snr", type=int, default=ExperimentConfig.train_per_class_snr)
    parser.add_argument("--test-per-class-snr", type=int, default=ExperimentConfig.test_per_class_snr)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ExperimentConfig.weight_decay)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--output-dir", default=ExperimentConfig.output_dir)
    parser.add_argument("--allow-cpu", action="store_true", help="Run on CPU if CUDA is unavailable.")
    args = parser.parse_args()
    return ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_per_class_snr=args.train_per_class_snr,
        test_per_class_snr=args.test_per_class_snr,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        require_cuda=not args.allow_cpu,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    if config.require_cuda and not cuda_available:
        raise SystemExit(
            "CUDA is not available. Check NVIDIA driver and CUDA PyTorch installation, "
            "or rerun with --allow-cpu only for debugging."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    log_lines: list[str] = []
    log_lines.append("Automatic modulation classification GPU experiment")
    log_lines.append(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Python platform: {platform.platform()}")
    log_lines.append(f"PyTorch version: {torch.__version__}")
    log_lines.append(f"CUDA available: {cuda_available}")
    if cuda_available:
        log_lines.append(f"CUDA version reported by torch: {torch.version.cuda}")
        log_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        log_lines.append(f"GPU memory: {props.total_memory / (1024 ** 3):.2f} GB")
    log_lines.append(f"Config: {json.dumps(asdict(config), ensure_ascii=False)}")
    log_lines.append(f"Classes: {', '.join(MODULATIONS)}")
    log_lines.append(f"SNR levels: {SNR_LEVELS}")

    print("\n".join(log_lines))
    print("Generating datasets...")
    train_x, train_y, train_snr = build_dataset(config.train_per_class_snr, config.seed)
    test_x, test_y, test_snr = build_dataset(config.test_per_class_snr, config.seed + 1)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y, train_snr),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y, test_snr),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = AMCNet(len(MODULATIONS)).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    log_lines.append(f"Train samples: {len(train_y)}")
    log_lines.append(f"Test samples: {len(test_y)}")
    log_lines.append(f"Trainable parameters: {n_params}")
    print(f"Train samples: {len(train_y)}")
    print(f"Test samples: {len(test_y)}")
    print(f"Trainable parameters: {n_params:,}")

    history: list[dict[str, float]] = []
    start_time = time.time()
    best_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x_batch, y_batch, _ in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu()) * len(y_batch)
            seen += len(y_batch)

        scheduler.step()
        train_loss = running_loss / seen
        test_accuracy, snr_accuracy, _ = evaluate(model, test_loader, device)
        best_accuracy = max(best_accuracy, test_accuracy)
        row = {"epoch": epoch, "train_loss": train_loss, "test_accuracy": test_accuracy}
        history.append(row)
        message = (
            f"epoch {epoch:02d}/{config.epochs} "
            f"loss={train_loss:.4f} test_accuracy={test_accuracy:.4f} best={best_accuracy:.4f}"
        )
        log_lines.append(message)
        print(message)

    total_time = time.time() - start_time
    final_accuracy, final_snr_accuracy, confusion = evaluate(model, test_loader, device)
    log_lines.append(f"Final accuracy: {final_accuracy:.6f}")
    log_lines.append(f"Best epoch accuracy: {best_accuracy:.6f}")
    log_lines.append(f"Elapsed seconds: {total_time:.2f}")
    for snr in SNR_LEVELS:
        log_lines.append(f"SNR {snr:>3} dB accuracy: {final_snr_accuracy[snr]:.6f}")

    save_accuracy_csv(output_dir / "accuracy_by_snr.csv", final_snr_accuracy)
    save_accuracy_plot(output_dir / "accuracy_by_snr.png", final_snr_accuracy)
    save_training_curve(output_dir / "training_curve.png", history)
    save_confusion_matrix(output_dir / "confusion_matrix.png", confusion)
    torch.save(model.state_dict(), output_dir / "amcnet_state_dict.pt")
    write_log(output_dir / "gpu_run_log.txt", log_lines)

    print("\nFinal accuracy:", f"{final_accuracy:.4f}")
    print("Accuracy by SNR:")
    for snr in SNR_LEVELS:
        print(f"  {snr:>3} dB: {final_snr_accuracy[snr]:.4f}")
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
