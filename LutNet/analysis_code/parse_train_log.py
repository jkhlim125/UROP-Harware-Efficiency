import argparse
import os
import re

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def avg(values):
    if not values:
        return None
    return sum(values) / len(values)


def parse_log(log_path):
    train_loss_by_epoch = {}
    bib_loss_by_epoch = {}
    test_acc_by_epoch = {}
    test_loss_by_epoch = {}
    event_hits = {
        "warmup": False,
        "pairing": False,
        "pruning": False,
        "fpga_report": False,
    }
    event_lines = []
    seen_event_lines = set()

    current_epoch = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            if "Warmup finished" in line:
                event_hits["warmup"] = True
                if (line_num, line) not in seen_event_lines:
                    event_lines.append((line_num, line))
                    seen_event_lines.add((line_num, line))
            if "Computing Pairing Maps" in line or "[Pairing]" in line:
                event_hits["pairing"] = True
                if (line_num, line) not in seen_event_lines:
                    event_lines.append((line_num, line))
                    seen_event_lines.add((line_num, line))
            if "Pruning Surgery Completed" in line:
                event_hits["pruning"] = True
                if (line_num, line) not in seen_event_lines:
                    event_lines.append((line_num, line))
                    seen_event_lines.add((line_num, line))
            if "FPGA PHYSICAL COMPILATION REPORT" in line:
                event_hits["fpga_report"] = True
                if (line_num, line) not in seen_event_lines:
                    event_lines.append((line_num, line))
                    seen_event_lines.add((line_num, line))

            m = re.search(
                r"Train Epoch:\s*(\d+).*?Loss:\s*([0-9]*\.?[0-9]+)(?:\s+BIB_Loss:\s*([0-9]*\.?[0-9]+))?",
                line,
            )
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                bib = m.group(3)
                current_epoch = epoch

                train_loss_by_epoch.setdefault(epoch, []).append(loss)
                if bib is not None:
                    bib_loss_by_epoch.setdefault(epoch, []).append(float(bib))
                continue

            m = re.search(
                r"Test set:\s*Average loss:\s*([0-9]*\.?[0-9]+),\s*Accuracy:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)",
                line,
            )
            if m and current_epoch is not None:
                test_loss = float(m.group(1))
                correct = int(m.group(2))
                total = int(m.group(3))
                test_acc = 100.0 * correct / total if total else None

                # DDP logs can print the same test line multiple times.
                if current_epoch not in test_acc_by_epoch:
                    test_loss_by_epoch[current_epoch] = test_loss
                    test_acc_by_epoch[current_epoch] = test_acc
                continue

    train_loss_by_epoch = {
        epoch: avg(values) for epoch, values in train_loss_by_epoch.items()
    }
    bib_loss_by_epoch = {
        epoch: avg(values) for epoch, values in bib_loss_by_epoch.items()
    }

    return {
        "train_loss_by_epoch": train_loss_by_epoch,
        "bib_loss_by_epoch": bib_loss_by_epoch,
        "test_acc_by_epoch": test_acc_by_epoch,
        "test_loss_by_epoch": test_loss_by_epoch,
        "event_hits": event_hits,
        "event_lines": event_lines,
    }


def first_epoch_above(acc_by_epoch, threshold):
    for epoch in sorted(acc_by_epoch):
        acc = acc_by_epoch[epoch]
        if acc is not None and acc >= threshold:
            return epoch
    return None


def print_summary(log_path, parsed, threshold):
    train_loss_by_epoch = parsed["train_loss_by_epoch"]
    bib_loss_by_epoch = parsed["bib_loss_by_epoch"]
    test_acc_by_epoch = parsed["test_acc_by_epoch"]
    event_hits = parsed["event_hits"]
    event_lines = parsed["event_lines"]

    print("=" * 70)
    print("Quick Log Summary")
    print("=" * 70)
    print("file:", log_path)
    print("epochs with train lines:", len(train_loss_by_epoch))
    print("epochs with test accuracy:", len(test_acc_by_epoch))

    if test_acc_by_epoch:
        best_epoch = max(test_acc_by_epoch, key=lambda e: test_acc_by_epoch[e])
        best_acc = test_acc_by_epoch[best_epoch]
        final_epoch = sorted(test_acc_by_epoch)[-1]
        final_acc = test_acc_by_epoch[final_epoch]
        hit_epoch = first_epoch_above(test_acc_by_epoch, threshold)

        print()
        print(f"best test accuracy: {best_acc:.2f}% at epoch {best_epoch}")
        print(f"final test accuracy: {final_acc:.2f}% at epoch {final_epoch}")
        if hit_epoch is None:
            print(f"first epoch >= {threshold:.1f}%: not found")
        else:
            print(f"first epoch >= {threshold:.1f}%: {hit_epoch}")
    else:
        print()
        print("no test accuracy lines found")

    if train_loss_by_epoch:
        first_train_epoch = sorted(train_loss_by_epoch)[0]
        last_train_epoch = sorted(train_loss_by_epoch)[-1]
        print()
        print(
            f"train loss: epoch {first_train_epoch} = {train_loss_by_epoch[first_train_epoch]:.4f}, "
            f"epoch {last_train_epoch} = {train_loss_by_epoch[last_train_epoch]:.4f}"
        )

    if bib_loss_by_epoch:
        first_bib_epoch = sorted(bib_loss_by_epoch)[0]
        last_bib_epoch = sorted(bib_loss_by_epoch)[-1]
        print(
            f"BIB loss seen: yes (epoch {first_bib_epoch} = {bib_loss_by_epoch[first_bib_epoch]:.4f}, "
            f"epoch {last_bib_epoch} = {bib_loss_by_epoch[last_bib_epoch]:.4f})"
        )
    else:
        print("BIB loss seen: no")

    print()
    print("events:")
    print("  warmup finished:", event_hits["warmup"])
    print("  pairing map step:", event_hits["pairing"])
    print("  pruning surgery:", event_hits["pruning"])
    print("  fpga report:", event_hits["fpga_report"])

    if event_lines:
        print()
        print("some event lines:")
        for line_num, text in event_lines[:8]:
            print(f"  line {line_num}: {text}")


def maybe_plot(parsed, log_path, save_path=None, show_plot=False):
    if not HAVE_MPL:
        print("matplotlib not installed, skipping plot")
        return

    test_acc_by_epoch = parsed["test_acc_by_epoch"]
    if not test_acc_by_epoch:
        print("no test accuracy found, skipping plot")
        return

    epochs = sorted(test_acc_by_epoch)
    accs = [test_acc_by_epoch[e] for e in epochs]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Accuracy vs Epoch\n{os.path.basename(log_path)}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print("saved plot to:", save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quick parser for LUT pruning training logs")
    parser.add_argument("log_file", help="raw log text file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="accuracy threshold for first-hit summary",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show a simple accuracy-vs-epoch plot",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="optional path to save the accuracy plot",
    )
    args = parser.parse_args()

    parsed = parse_log(args.log_file)
    print_summary(args.log_file, parsed, args.threshold)

    if args.plot or args.save_plot:
        maybe_plot(parsed, args.log_file, save_path=args.save_plot or None, show_plot=args.plot)


if __name__ == "__main__":
    main()
