import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from radio_dataloader import get_radioml2016a_dataloaders


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adapt_radioml_input(data: torch.Tensor) -> torch.Tensor:
    """
    Convert possible shapes into [B, 2, 128] for Conv1d.
    Supported examples:
      [B, 2, 128]
      [B, 1, 2, 128]
      [B, 2, 1, 128]
      [B, 128, 2]
    """
    if data.dim() == 4 and data.size(1) == 1 and data.size(2) == 2:
        data = data.squeeze(1)
    elif data.dim() == 4 and data.size(1) == 2 and data.size(2) == 1:
        data = data.squeeze(2)

    if data.dim() != 3:
        raise ValueError(f"Unexpected input shape: {tuple(data.shape)}")

    if data.size(1) != 2:
        if data.size(2) == 2:
            data = data.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Expected channel dim=2, got shape {tuple(data.shape)}")

    return data.float()


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

        data = adapt_radioml_input(data).to(device, non_blocking=True)
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
    dry_run: bool,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            data, target, _ = batch
        else:
            data, target = batch

        data = adapt_radioml_input(data).to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(data)
            loss = criterion(logits, target)

        batch_size = target.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(logits, target)
        total += batch_size

        if dry_run and batch_idx >= 0:
            break

    avg_loss = running_loss / max(1, total)
    avg_acc = running_correct / max(1, total)
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch Traditional CNN for RadioML 2016.10a")
    parser.add_argument("--radioml-path", type=str, default="../data/RML2016.10a_dict.pkl")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-path", type=str, default="radioml_cnn_best.pt")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_loader, val_loader, test_loader, _, meta = get_radioml2016a_dataloaders(
        pkl_path=args.radioml_path,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        as_2d=False,
        normalize=True,
        distributed=False,
        pin_memory=(device.type == "cuda"),
    )

    model = TraditionalRadioMLCNN(num_classes=meta["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_acc = -1.0

    print(f"Device: {device}")
    print(f"Num classes: {meta['num_classes']}")
    print(f"Train/Val/Test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            scaler=scaler,
            amp_enabled=amp_enabled,
            dry_run=args.dry_run,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            dry_run=args.dry_run,
        )

        scheduler.step()

        print(
            "Epoch {:03d}: train_loss={:.4f} train_acc={:.2f}% | val_loss={:.4f} val_acc={:.2f}%".format(
                epoch,
                train_loss,
                train_acc * 100.0,
                val_loss,
                val_acc * 100.0,
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "num_classes": meta["num_classes"],
                },
                args.save_path,
            )
            print(f"Saved best checkpoint to {args.save_path} (val_acc={val_acc * 100:.2f}%)")

        if args.dry_run:
            break

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        dry_run=args.dry_run,
    )
    print("Final Test: loss={:.4f} acc={:.2f}%".format(test_loss, test_acc * 100.0))


if __name__ == "__main__":
    main()
