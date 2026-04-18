"""
RadioML refined two-branch CNN (v3).

Refinements over v2:
1. Keep the successful two-branch late-fusion structure
2. Keep the smoothed IF idea
3. Expand IF input to a medium-size interpretable feature set
4. Remove v2 class weighting and return to standard cross-entropy

Architecture:
- IQ branch: same as v2
- IF branch: same lightweight v2 stack, now with 8-channel input
- Fusion: concatenate + classify
"""

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from radio_dataloader_branch_v3 import get_radioml2016a_dataloaders_branch_v3


class IQFeatureExtractorV3(nn.Module):
    """Feature extractor for raw IQ branch (same as v2)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class IFFeatureExtractorV3(nn.Module):
    """IF feature extractor with the same v2 depth and an 8-channel input."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class RadioMLBranchCNNV3(nn.Module):
    """Refined two-branch CNN with minimal v3 IF-feature expansion."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.iq_branch = IQFeatureExtractorV3()
        self.if_branch = IFFeatureExtractorV3()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(8192 + 4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, iq: torch.Tensor, if_feat: torch.Tensor) -> torch.Tensor:
        iq_features = self.iq_branch(iq)
        if_features = self.if_branch(if_feat)
        fused = torch.cat([iq_features, if_features], dim=1)
        logits = self.classifier(fused)
        return logits


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
        if len(batch) == 4:
            iq, if_feat, target, _ = batch
        else:
            iq, if_feat, target = batch

        iq = iq.to(device, non_blocking=True)
        if_feat = if_feat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(iq, if_feat)
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
            if len(batch) == 4:
                iq, if_feat, target, _ = batch
            else:
                iq, if_feat, target = batch

            iq = iq.to(device, non_blocking=True)
            if_feat = if_feat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(iq, if_feat)
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
    parser = argparse.ArgumentParser(description="Train refined two-branch RadioML CNN (v3)")
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

    print("Loading data...")
    train_loader, val_loader, test_loader = get_radioml2016a_dataloaders_branch_v3(
        args.data_path,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    model = RadioMLBranchCNNV3(num_classes=11)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    print("Using standard cross-entropy loss (no class weighting).")
    print(
        "IF branch channels: IF, IF_smoothed, IF_abs, IF_energy, "
        "IF_variance, amplitude_envelope, amplitude_mean, amplitude_variance"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_acc = 0.0
    best_epoch = 0

    print("Starting training...")

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

        if epoch % 10 == 0 or val_acc > best_val_acc:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "radioml_cnn_branch_v3_best.pt")
            print(f"  -> Best model saved (epoch {best_epoch})")

        if args.dry_run:
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
