import argparse
import csv
import os
import re

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def parse_one_log(log_path):
    test_acc_by_epoch = {}
    current_epoch = None

    pruning_surgery_happened = False
    pack_ratio = None
    global_sparsity = None
    slice_reduction = None
    pin_reduction_rate = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if "Pruning Surgery Completed" in line:
                pruning_surgery_happened = True

            m = re.search(
                r"Train Epoch:\s*(\d+).*?Loss:\s*([0-9]*\.?[0-9]+)",
                line,
            )
            if m:
                current_epoch = int(m.group(1))
                continue

            m = re.search(
                r"Test set:\s*Average loss:\s*([0-9]*\.?[0-9]+),\s*Accuracy:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)",
                line,
            )
            if m and current_epoch is not None:
                correct = int(m.group(2))
                total = int(m.group(3))
                acc = 100.0 * correct / total if total else None

                if current_epoch not in test_acc_by_epoch:
                    test_acc_by_epoch[current_epoch] = acc
                continue

            m = re.search(
                r"Pack Ratio:\s*([0-9]*\.?[0-9]+),\s*Global Sparsity:\s*([0-9]*\.?[0-9]+)",
                line,
            )
            if m:
                pack_ratio = float(m.group(1))
                global_sparsity = float(m.group(2))
                continue

            m = re.search(r"Slice Reduction\s*:\s*([0-9]*\.?[0-9]+)%", line)
            if m:
                slice_reduction = float(m.group(1))
                continue

            m = re.search(r"Pin Reduction Rate\s*:\s*([0-9]*\.?[0-9]+)%", line)
            if m:
                pin_reduction_rate = float(m.group(1))
                continue

    best_test_accuracy = None
    final_test_accuracy = None

    if test_acc_by_epoch:
        best_test_accuracy = max(test_acc_by_epoch.values())
        final_epoch = sorted(test_acc_by_epoch)[-1]
        final_test_accuracy = test_acc_by_epoch[final_epoch]

    return {
        "run": os.path.basename(log_path),
        "best_test_accuracy": best_test_accuracy,
        "final_test_accuracy": final_test_accuracy,
        "pruning_surgery_happened": pruning_surgery_happened,
        "pack_ratio": pack_ratio,
        "global_sparsity": global_sparsity,
        "slice_reduction": slice_reduction,
        "pin_reduction_rate": pin_reduction_rate,
    }


def make_plots(df):
    if not HAVE_MPL:
        print("matplotlib not installed, skipping plots")
        return

    if df.empty:
        print("empty dataframe, skipping plots")
        return

    plt.figure(figsize=(9, 4.5))
    plt.bar(df["run"], df["best_test_accuracy"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Best Test Accuracy (%)")
    plt.title("Best Accuracy vs Run")
    plt.tight_layout()
    plt.show()

    sub = df.dropna(subset=["slice_reduction", "pin_reduction_rate"])
    if len(sub) > 0:
        plt.figure(figsize=(6, 5))
        plt.scatter(sub["slice_reduction"], sub["pin_reduction_rate"])
        for _, row in sub.iterrows():
            plt.text(row["slice_reduction"], row["pin_reduction_rate"], row["run"], fontsize=8)
        plt.xlabel("Slice Reduction (%)")
        plt.ylabel("Pin Reduction Rate (%)")
        plt.title("Slice Reduction vs Pin Reduction")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("no slice/pin reduction pairs found, skipping scatter plot")


def print_simple_table(rows):
    cols = [
        "run",
        "best_test_accuracy",
        "final_test_accuracy",
        "pruning_surgery_happened",
        "pack_ratio",
        "global_sparsity",
        "slice_reduction",
        "pin_reduction_rate",
    ]

    widths = {}
    for col in cols:
        widths[col] = len(col)
        for row in rows:
            value = row.get(col, "")
            text = "" if value is None else str(value)
            widths[col] = max(widths[col], len(text))

    header = "  ".join(col.ljust(widths[col]) for col in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(
            ("" if row.get(col) is None else str(row.get(col))).ljust(widths[col])
            for col in cols
        )
        print(line)


def save_simple_csv(rows, out_path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Quick comparison for LUT pruning log files")
    parser.add_argument("log_files", nargs="+", help="list of raw log files")
    parser.add_argument("--csv", type=str, default="", help="optional output csv path")
    parser.add_argument("--plot", action="store_true", help="show a couple of quick plots")
    args = parser.parse_args()

    rows = []
    for log_file in args.log_files:
        print("parsing:", log_file)
        rows.append(parse_one_log(log_file))

    print()
    if HAVE_PANDAS:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    else:
        df = None
        print("pandas not installed, printing a plain table instead")
        print_simple_table(rows)

    if args.csv:
        if HAVE_PANDAS:
            df.to_csv(args.csv, index=False)
        else:
            save_simple_csv(rows, args.csv)
        print()
        print("saved csv to:", args.csv)

    if args.plot:
        if HAVE_PANDAS:
            make_plots(df)
        else:
            print("plots skipped because pandas is not installed in this environment")


if __name__ == "__main__":
    main()
