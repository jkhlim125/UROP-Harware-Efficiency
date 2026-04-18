"""
RadioML CNN trained with instantaneous frequency features.

Minimal change from baseline:
  - Input channels: 2 (I, Q) → 3 (I, Q, phase_derivative)
  - First Conv1d layer: in_channels 2 → 3
  - All other architecture identical to baseline
"""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from radio_dataloader_ifreq import get_radioml2016a_dataloaders_ifreq


class RadioMLCNNIFreq(nn.Module):
    """1D CNN for RadioML with instantaneous frequency feature (3 channels)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),  # 3 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 128 -> 64
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 64 -> 32
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(8192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> int:
    pred = logits.argmax(dim=1)
    return pred.eq(target).sum().item()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    log_interval: int,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    dry_run: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(data)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = target.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(logits, target)
        total += batch_size

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * batch_size,
                    len(loader.dataset),
                    100.0 * batch_idx / max(1, len(loader)),
                    loss.item(),
                )
            )

        if dry_run:
            break

    avg_loss = running_loss / max(1, total)
    avg_acc = running_correct / max(1, total)
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
    dry_run: bool,
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(data)
                loss = criterion(logits, target)

            batch_size = target.size(0)
            val_loss += loss.item() * batch_size
            val_correct += accuracy_from_logits(logits, target)
            total += batch_size

            if dry_run:
                break

    avg_loss = val_loss / max(1, total)
    avg_acc = val_correct / max(1, total)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train RadioML CNN with IF features")
    parser.add_argument("--data-path", type=str, default="RML2016.10a_dict.pkl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_radioml2016a_dataloaders_ifreq(
        args.data_path,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = RadioMLCNNIFreq(num_classes=11)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    amp_enabled = str(device) == "cuda"

    best_val_acc = 0.0
    best_epoch = 0

    print("Starting training...")
    with open("temp_ifreq.log", "w") as log_file:
        log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                log_interval=50,
                scaler=scaler,
                amp_enabled=amp_enabled,
                dry_run=args.dry_run,
            )

            val_loss, val_acc = validate(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                dry_run=args.dry_run,
            )

            log_file.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f}\n")
            log_file.flush()

            if epoch % 10 == 0 or val_acc > best_val_acc:
                print(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), "radioml_cnn_ifreq_best.pt")
                print(f"  -> Best model saved (epoch {best_epoch})")

            if args.dry_run:
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
