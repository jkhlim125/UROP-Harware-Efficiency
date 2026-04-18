import os
import re
import csv
import glob
import matplotlib.pyplot as plt
from collections import defaultdict


# =========================================================
# Options
# =========================================================
SAVE_SUMMARY_CSV = True
OUTPUT_ROOT = "figures"


# =========================================================
# Basic utilities
# =========================================================
def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def config_to_name(cfg):
    if cfg == "baseline":
        return "baseline"
    return f"pr{cfg[0]}_gs{cfg[1]}"


def sort_configs(configs):
    baseline = [c for c in configs if c == "baseline"]
    others = [c for c in configs if c != "baseline"]
    others = sorted(others, key=lambda x: (x[0], x[1]))
    return baseline + others


def find_model_tags():
    l1_files = glob.glob("log*_l1.txt")
    l2_files = glob.glob("log*_l2.txt")

    l1_tags = set()
    l2_tags = set()

    for f in l1_files:
        m = re.match(r"log(.+)_l1\.txt$", os.path.basename(f))
        if m:
            l1_tags.add(m.group(1))

    for f in l2_files:
        m = re.match(r"log(.+)_l2\.txt$", os.path.basename(f))
        if m:
            l2_tags.add(m.group(1))

    return sorted(l1_tags & l2_tags)


# =========================================================
# Parse one log file
# =========================================================
def parse_log(log_file: str):
    data = {}
    current_cfg = "baseline"
    current_epoch = None

    data[current_cfg] = {
        "train_loss": defaultdict(list),
        "test_loss": {},
        "test_acc": {},
        "hardware": {}
    }

    hw_tmp = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            # -------------------------------------------------
            # Config marker
            # Example:
            # [Surgery] Rank 0 starting calculation (Pack Ratio: 0.8, Global Sparsity: 0.7)...
            # -------------------------------------------------
            cfg_match = re.search(
                r"Pack Ratio:\s*([\d.]+),\s*Global Sparsity:\s*([\d.]+)",
                line
            )
            if cfg_match:
                cfg = (float(cfg_match.group(1)), float(cfg_match.group(2)))
                current_cfg = cfg
                if current_cfg not in data:
                    data[current_cfg] = {
                        "train_loss": defaultdict(list),
                        "test_loss": {},
                        "test_acc": {},
                        "hardware": {}
                    }
                current_epoch = None
                hw_tmp = None
                continue

            # -------------------------------------------------
            # Train line
            # Example:
            # Train Epoch: 81 ... Loss: 0.8721
            # -------------------------------------------------
            train_match = re.search(
                r"Train Epoch:\s*(\d+).*?Loss:\s*([0-9.]+)",
                line
            )
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                current_epoch = epoch
                data[current_cfg]["train_loss"][epoch].append(loss)
                continue

            # -------------------------------------------------
            # Test line
            # Example:
            # Test set: Average loss: 1.0523, Accuracy: 6057/10000
            # -------------------------------------------------
            test_match = re.search(
                r"Test set:\s*Average loss:\s*([0-9.]+),\s*Accuracy:\s*(\d+)/(\d+)",
                line
            )
            if test_match and current_epoch is not None:
                test_loss = float(test_match.group(1))
                correct = int(test_match.group(2))
                total = int(test_match.group(3))
                acc = 100.0 * correct / total

                # Remove duplicated DDP test prints
                if current_epoch not in data[current_cfg]["test_loss"]:
                    data[current_cfg]["test_loss"][current_epoch] = test_loss
                    data[current_cfg]["test_acc"][current_epoch] = acc
                continue

            # -------------------------------------------------
            # Hardware summary block start
            # -------------------------------------------------
            if "Final Summary (Physics-Aware):" in line:
                hw_tmp = {}
                continue

            # -------------------------------------------------
            # Parse hardware summary lines
            # -------------------------------------------------
            if hw_tmp is not None:
                m = re.search(r"Total Sub-LUTs\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["total_sub_luts"] = int(m.group(1))
                    continue

                m = re.search(r"Total Dead\s*:\s*(\d+)\s*\(([\d.]+)%\)", line)
                if m:
                    hw_tmp["total_dead"] = int(m.group(1))
                    hw_tmp["dead_ratio"] = float(m.group(2))
                    continue

                m = re.search(r"Successful Packs\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["successful_packs"] = int(m.group(1))
                    continue

                m = re.search(r"Failed Packs\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["failed_packs"] = int(m.group(1))
                    continue

                m = re.search(r"Total Slices Used\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["total_slices_used"] = int(m.group(1))
                    continue

                m = re.search(r"Slice Reduction\s*:\s*([\d.]+)%", line)
                if m:
                    hw_tmp["slice_reduction"] = float(m.group(1))
                    continue

                m = re.search(r"Total Pins \(Theory\)\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["pins_theory"] = int(m.group(1))
                    continue

                m = re.search(r"Total Pins \(Phys\.?\)\s*:\s*(\d+)", line)
                if m:
                    hw_tmp["pins_phys"] = int(m.group(1))
                    continue

                m = re.search(r"Pin Reduction Rate\s*:\s*([\d.]+)%", line)
                if m:
                    hw_tmp["pin_reduction_rate"] = float(m.group(1))
                    continue

                m = re.search(r"Avg Fan-in / LUT\s*:\s*([\d.]+)", line)
                if m:
                    hw_tmp["avg_fanin"] = float(m.group(1))
                    data[current_cfg]["hardware"] = hw_tmp
                    hw_tmp = None
                    continue

    # Average repeated train loss values
    for cfg in data:
        for epoch in data[cfg]["train_loss"]:
            vals = data[cfg]["train_loss"][epoch]
            data[cfg]["train_loss"][epoch] = sum(vals) / len(vals)

    return data


# =========================================================
# Plot: single config (L1 or L2 only)
# =========================================================
def plot_single(data, cfg, label, out_dir, model_tag):
    if cfg not in data:
        return

    d = data[cfg]
    train_epochs = sorted(d["train_loss"].keys())
    test_epochs = sorted(d["test_loss"].keys())

    if not train_epochs and not test_epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Train loss
    if train_epochs:
        y = [d["train_loss"][e] for e in train_epochs]
        axes[0].plot(train_epochs, y, linewidth=2)
    axes[0].set_title(f"{label} Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Test loss
    if test_epochs:
        y = [d["test_loss"][e] for e in test_epochs]
        axes[1].plot(test_epochs, y, linewidth=2)
    axes[1].set_title(f"{label} Test Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    # Test accuracy
    if test_epochs:
        y = [d["test_acc"][e] for e in test_epochs]
        axes[2].plot(test_epochs, y, linewidth=2)
    axes[2].set_title(f"{label} Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{model_tag}_{label}_{config_to_name(cfg)}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()


# =========================================================
# Plot: compare one config (L1 vs L2)
# =========================================================
def plot_compare(l1_data, l2_data, cfg, out_dir, model_tag):
    if cfg not in l1_data or cfg not in l2_data:
        return

    d1 = l1_data[cfg]
    d2 = l2_data[cfg]

    train_epochs = sorted(set(d1["train_loss"].keys()) & set(d2["train_loss"].keys()))
    test_epochs = sorted(set(d1["test_loss"].keys()) & set(d2["test_loss"].keys()))

    if not train_epochs and not test_epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Train loss
    if train_epochs:
        y1 = [d1["train_loss"][e] for e in train_epochs]
        y2 = [d2["train_loss"][e] for e in train_epochs]
        axes[0].plot(train_epochs, y1, label="L1", linewidth=2)
        axes[0].plot(train_epochs, y2, label="L2", linewidth=2, linestyle="--")
        axes[0].legend()
    axes[0].set_title(f"Compare Train Loss ({cfg})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Test loss
    if test_epochs:
        y1 = [d1["test_loss"][e] for e in test_epochs]
        y2 = [d2["test_loss"][e] for e in test_epochs]
        axes[1].plot(test_epochs, y1, label="L1", linewidth=2)
        axes[1].plot(test_epochs, y2, label="L2", linewidth=2, linestyle="--")
        axes[1].legend()
    axes[1].set_title(f"Compare Test Loss ({cfg})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    # Test accuracy
    if test_epochs:
        y1 = [d1["test_acc"][e] for e in test_epochs]
        y2 = [d2["test_acc"][e] for e in test_epochs]
        axes[2].plot(test_epochs, y1, label="L1", linewidth=2)
        axes[2].plot(test_epochs, y2, label="L2", linewidth=2, linestyle="--")
        axes[2].legend()
    axes[2].set_title(f"Compare Test Accuracy ({cfg})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"{model_tag}_compare_{config_to_name(cfg)}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close()


# =========================================================
# Summary rows
# =========================================================
def get_final_rows(data, norm_name):
    rows = []

    for cfg in sort_configs(list(data.keys())):
        d = data[cfg]
        if not d["test_acc"]:
            continue

        last_epoch = max(d["test_acc"].keys())

        rows.append({
            "norm": norm_name,
            "config": cfg,
            "pack_ratio": None if cfg == "baseline" else cfg[0],
            "global_sparsity": None if cfg == "baseline" else cfg[1],
            "final_epoch": last_epoch,
            "final_train_loss": d["train_loss"].get(last_epoch, None),
            "final_test_loss": d["test_loss"][last_epoch],
            "final_test_acc": d["test_acc"][last_epoch],
        })

    return rows


def get_hardware_rows(data, norm_name):
    rows = []

    baseline_acc = None
    if "baseline" in data and data["baseline"]["test_acc"]:
        e = max(data["baseline"]["test_acc"].keys())
        baseline_acc = data["baseline"]["test_acc"][e]

    for cfg in sort_configs(list(data.keys())):
        if cfg == "baseline":
            continue

        d = data[cfg]
        if not d["test_acc"]:
            continue
        if not d["hardware"]:
            continue

        last_epoch = max(d["test_acc"].keys())
        acc = d["test_acc"][last_epoch]
        loss = d["test_loss"][last_epoch]
        hw = d["hardware"]

        rows.append({
            "norm": norm_name,
            "config": cfg,
            "pack_ratio": cfg[0],
            "global_sparsity": cfg[1],
            "final_test_acc": acc,
            "final_test_loss": loss,
            "acc_drop": None if baseline_acc is None else baseline_acc - acc,
            "total_sub_luts": hw.get("total_sub_luts"),
            "total_dead": hw.get("total_dead"),
            "dead_ratio": hw.get("dead_ratio"),
            "successful_packs": hw.get("successful_packs"),
            "failed_packs": hw.get("failed_packs"),
            "total_slices_used": hw.get("total_slices_used"),
            "slice_reduction": hw.get("slice_reduction"),
            "pins_theory": hw.get("pins_theory"),
            "pins_phys": hw.get("pins_phys"),
            "pin_reduction_rate": hw.get("pin_reduction_rate"),
            "avg_fanin": hw.get("avg_fanin"),
        })

    return rows


def save_csv(l1_rows, l2_rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "norm", "config", "pack_ratio", "global_sparsity",
            "final_epoch", "final_train_loss", "final_test_loss", "final_test_acc"
        ])

        for row in l1_rows + l2_rows:
            writer.writerow([
                row["norm"],
                row["config"],
                row["pack_ratio"],
                row["global_sparsity"],
                row["final_epoch"],
                row["final_train_loss"],
                row["final_test_loss"],
                row["final_test_acc"],
            ])


def save_hardware_csv(l1_rows, l2_rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "norm", "config", "pack_ratio", "global_sparsity",
            "final_test_acc", "final_test_loss", "acc_drop",
            "total_sub_luts", "total_dead", "dead_ratio",
            "successful_packs", "failed_packs", "total_slices_used",
            "slice_reduction", "pins_theory", "pins_phys",
            "pin_reduction_rate", "avg_fanin"
        ])

        for row in l1_rows + l2_rows:
            writer.writerow([
                row["norm"], row["config"], row["pack_ratio"], row["global_sparsity"],
                row["final_test_acc"], row["final_test_loss"], row["acc_drop"],
                row["total_sub_luts"], row["total_dead"], row["dead_ratio"],
                row["successful_packs"], row["failed_packs"], row["total_slices_used"],
                row["slice_reduction"], row["pins_theory"], row["pins_phys"],
                row["pin_reduction_rate"], row["avg_fanin"]
            ])


def split_baseline(rows):
    baseline = None
    others = []
    for r in rows:
        if r["config"] == "baseline":
            baseline = r
        else:
            others.append(r)
    return baseline, others


# =========================================================
# Summary plots
# =========================================================
def plot_acc_vs_sparsity(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        xs, ys = [], []
        for r in rows:
            if r["config"] == "baseline":
                continue
            xs.append(r["global_sparsity"])
            ys.append(r["final_test_acc"])
        plt.scatter(xs, ys, label=label, marker=marker)

    plt.xlabel("Global Sparsity")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title(f"{model_tag}: Accuracy vs Global Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_overlay_acc_vs_sparsity.png"), dpi=300)
    plt.close()


def plot_acc_vs_pack(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        xs, ys = [], []
        for r in rows:
            if r["config"] == "baseline":
                continue
            xs.append(r["pack_ratio"])
            ys.append(r["final_test_acc"])
        plt.scatter(xs, ys, label=label, marker=marker)

    plt.xlabel("Pack Ratio")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title(f"{model_tag}: Accuracy vs Pack Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_overlay_acc_vs_pack_ratio.png"), dpi=300)
    plt.close()


def plot_testloss_vs_sparsity(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        xs, ys = [], []
        for r in rows:
            if r["config"] == "baseline":
                continue
            xs.append(r["global_sparsity"])
            ys.append(r["final_test_loss"])
        plt.scatter(xs, ys, label=label, marker=marker)

    plt.xlabel("Global Sparsity")
    plt.ylabel("Final Test Loss")
    plt.title(f"{model_tag}: Test Loss vs Global Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_overlay_testloss_vs_sparsity.png"), dpi=300)
    plt.close()


def plot_accdrop_vs_sparsity(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        baseline, others = split_baseline(rows)
        if baseline is None:
            continue

        base_acc = baseline["final_test_acc"]
        xs, ys = [], []

        for r in others:
            xs.append(r["global_sparsity"])
            ys.append(base_acc - r["final_test_acc"])

        plt.scatter(xs, ys, label=label, marker=marker)

    plt.xlabel("Global Sparsity")
    plt.ylabel("Accuracy Drop from Baseline (%)")
    plt.title(f"{model_tag}: Accuracy Drop vs Global Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_overlay_accdrop_vs_sparsity.png"), dpi=300)
    plt.close()


def plot_l2_minus_l1(common_cfgs, l1_data, l2_data, out_dir, model_tag):
    xs, ys = [], []
    idx = 0

    for cfg in common_cfgs:
        if cfg == "baseline":
            continue

        d1 = l1_data[cfg]
        d2 = l2_data[cfg]

        if not d1["test_acc"] or not d2["test_acc"]:
            continue

        e1 = max(d1["test_acc"].keys())
        e2 = max(d2["test_acc"].keys())

        acc1 = d1["test_acc"][e1]
        acc2 = d2["test_acc"][e2]

        xs.append(idx)
        ys.append(acc2 - acc1)
        idx += 1

    if not xs:
        return

    plt.figure(figsize=(10, 5))
    plt.scatter(xs, ys)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Configuration Index")
    plt.ylabel("L2 - L1 Final Accuracy (%)")
    plt.title(f"{model_tag}: Accuracy Difference (L2 - L1)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_delta_l2_minus_l1_accuracy.png"), dpi=300)
    plt.close()


def plot_line_by_pack_ratio(l1_rows, l2_rows, pack_ratio, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    def build_curve(rows):
        xs, ys = [], []
        for r in rows:
            if r["config"] == "baseline":
                continue
            if r["pack_ratio"] == pack_ratio:
                xs.append(r["global_sparsity"])
                ys.append(r["final_test_acc"])
        if not xs:
            return [], []
        pairs = sorted(zip(xs, ys))
        return [p[0] for p in pairs], [p[1] for p in pairs]

    x1, y1 = build_curve(l1_rows)
    x2, y2 = build_curve(l2_rows)

    if x1:
        plt.plot(x1, y1, marker="o", label="L1")
    if x2:
        plt.plot(x2, y2, marker="x", label="L2")

    plt.xlabel("Global Sparsity")
    plt.ylabel("Final Accuracy (%)")
    plt.title(f"{model_tag}: Acc vs Sparsity (pack_ratio={pack_ratio})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_line_packratio_{pack_ratio}.png"), dpi=300)
    plt.close()


def plot_best_acc_vs_pack_ratio(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    def best_per_pack(rows):
        best = {}
        for r in rows:
            if r["config"] == "baseline":
                continue
            pr = r["pack_ratio"]
            if pr not in best or r["final_test_acc"] > best[pr]["final_test_acc"]:
                best[pr] = r
        xs = sorted(best.keys())
        ys = [best[x]["final_test_acc"] for x in xs]
        return xs, ys, best

    x1, y1, best1 = best_per_pack(l1_rows)
    x2, y2, best2 = best_per_pack(l2_rows)

    if x1:
        plt.plot(x1, y1, marker="o", label="L1")
    if x2:
        plt.plot(x2, y2, marker="x", label="L2")

    plt.xlabel("Pack Ratio")
    plt.ylabel("Best Final Accuracy (%)")
    plt.title(f"{model_tag}: Best Accuracy vs Pack Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_best_acc_vs_pack_ratio.png"), dpi=300)
    plt.close()

    print(f"\n[{model_tag}] Best config per pack ratio")
    for label, best_dict in [("L1", best1), ("L2", best2)]:
        print(f"  {label}:")
        for pr in sorted(best_dict.keys()):
            r = best_dict[pr]
            print(
                f"    pr={pr}: gs={r['global_sparsity']}, "
                f"acc={r['final_test_acc']:.3f}, loss={r['final_test_loss']:.4f}"
            )


def plot_pareto(l1_rows, l2_rows, out_dir, model_tag):
    plt.figure(figsize=(7, 5))

    def collect_points(rows):
        pts = []
        for r in rows:
            if r["config"] == "baseline":
                continue
            pts.append({
                "x": r["global_sparsity"],
                "y": r["final_test_acc"],
            })
        return pts

    def pareto_front(points):
        front = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if (q["x"] >= p["x"] and q["y"] >= p["y"]) and (q["x"] > p["x"] or q["y"] > p["y"]):
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        front = sorted(front, key=lambda d: (d["x"], d["y"]))
        return front

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        pts = collect_points(rows)
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        plt.scatter(xs, ys, label=f"{label} all", marker=marker, alpha=0.5)

        front = pareto_front(pts)
        fx = [p["x"] for p in front]
        fy = [p["y"] for p in front]
        if fx:
            plt.plot(fx, fy, linewidth=2, label=f"{label} Pareto")

    plt.xlabel("Global Sparsity")
    plt.ylabel("Final Accuracy (%)")
    plt.title(f"{model_tag}: Pareto Accuracy vs Sparsity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_tag}_pareto_acc_vs_sparsity.png"), dpi=300)
    plt.close()


def print_best_results(l1_rows, l2_rows, model_tag):
    def print_one(rows, norm_name):
        baseline = next((r for r in rows if r["config"] == "baseline"), None)
        valid = [r for r in rows if r["config"] != "baseline"]

        if baseline is not None:
            print(
                f"\n[{model_tag} {norm_name}] Baseline"
                f"\n  acc={baseline['final_test_acc']:.3f}, "
                f"test_loss={baseline['final_test_loss']:.4f}"
            )

        if not valid:
            print(f"[{model_tag} {norm_name}] No pruning configs")
            return

        best_acc = max(valid, key=lambda r: r["final_test_acc"])
        print(
            f"[{model_tag} {norm_name}] Best Accuracy Config"
            f"\n  pack_ratio={best_acc['pack_ratio']}, "
            f"global_sparsity={best_acc['global_sparsity']}, "
            f"final_acc={best_acc['final_test_acc']:.3f}, "
            f"final_test_loss={best_acc['final_test_loss']:.4f}"
        )

    print_one(l1_rows, "L1")
    print_one(l2_rows, "L2")


# =========================================================
# Hardware summary plots
# =========================================================
def plot_hw_scatter(l1_rows, l2_rows, x_key, x_label, y_key, y_label, title, out_file):
    plt.figure(figsize=(7, 5))

    for rows, label, marker in [(l1_rows, "L1", "o"), (l2_rows, "L2", "x")]:
        xs, ys = [], []
        for r in rows:
            if r.get(x_key) is None or r.get(y_key) is None:
                continue
            xs.append(r[x_key])
            ys.append(r[y_key])
        plt.scatter(xs, ys, label=label, marker=marker)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# =========================================================
# Process one model
# =========================================================
def process_model(model_tag):
    log_l1 = f"log{model_tag}_l1.txt"
    log_l2 = f"log{model_tag}_l2.txt"

    model_root = os.path.join(OUTPUT_ROOT, model_tag)
    l1_dir = os.path.join(model_root, "l1_only")
    l2_dir = os.path.join(model_root, "l2_only")
    cmp_per_dir = os.path.join(model_root, "compare", "per_config")
    cmp_sum_dir = os.path.join(model_root, "compare", "summary")
    summary_dir = os.path.join(model_root, "summary")
    hardware_dir = os.path.join(model_root, "hardware")

    make_dir(model_root)
    make_dir(l1_dir)
    make_dir(l2_dir)
    make_dir(cmp_per_dir)
    make_dir(cmp_sum_dir)
    make_dir(summary_dir)
    make_dir(hardware_dir)

    l1_data = parse_log(log_l1)
    l2_data = parse_log(log_l2)

    l1_cfgs = sort_configs(list(l1_data.keys()))
    l2_cfgs = sort_configs(list(l2_data.keys()))
    common_cfgs = sort_configs(list(set(l1_cfgs) & set(l2_cfgs)))

    print(f"\n===== {model_tag} =====")
    print("L1 configs:")
    for c in l1_cfgs:
        print(" ", c)

    print("\nL2 configs:")
    for c in l2_cfgs:
        print(" ", c)

    print("\nCommon configs:")
    for c in common_cfgs:
        print(" ", c)

    # -----------------------------------------------------
    # Single figures
    # -----------------------------------------------------
    for cfg in l1_cfgs:
        plot_single(l1_data, cfg, "L1", l1_dir, model_tag)

    for cfg in l2_cfgs:
        plot_single(l2_data, cfg, "L2", l2_dir, model_tag)

    # -----------------------------------------------------
    # Compare per-config
    # -----------------------------------------------------
    for cfg in common_cfgs:
        plot_compare(l1_data, l2_data, cfg, cmp_per_dir, model_tag)

    # -----------------------------------------------------
    # Summary rows
    # -----------------------------------------------------
    l1_rows = get_final_rows(l1_data, "L1")
    l2_rows = get_final_rows(l2_data, "L2")

    if SAVE_SUMMARY_CSV:
        save_csv(l1_rows, l2_rows, os.path.join(summary_dir, f"{model_tag}_summary.csv"))

    # -----------------------------------------------------
    # Summary plots
    # -----------------------------------------------------
    plot_acc_vs_sparsity(l1_rows, l2_rows, cmp_sum_dir, model_tag)
    plot_acc_vs_pack(l1_rows, l2_rows, cmp_sum_dir, model_tag)
    plot_testloss_vs_sparsity(l1_rows, l2_rows, cmp_sum_dir, model_tag)
    plot_accdrop_vs_sparsity(l1_rows, l2_rows, cmp_sum_dir, model_tag)
    plot_l2_minus_l1(common_cfgs, l1_data, l2_data, cmp_sum_dir, model_tag)
    plot_best_acc_vs_pack_ratio(l1_rows, l2_rows, cmp_sum_dir, model_tag)
    plot_pareto(l1_rows, l2_rows, cmp_sum_dir, model_tag)

    pack_ratios = sorted(set(
        r["pack_ratio"] for r in (l1_rows + l2_rows)
        if r["pack_ratio"] is not None
    ))
    for pr in pack_ratios:
        plot_line_by_pack_ratio(l1_rows, l2_rows, pr, cmp_sum_dir, model_tag)

    print_best_results(l1_rows, l2_rows, model_tag)

    # -----------------------------------------------------
    # Hardware rows
    # -----------------------------------------------------
    l1_hw_rows = get_hardware_rows(l1_data, "L1")
    l2_hw_rows = get_hardware_rows(l2_data, "L2")

    if SAVE_SUMMARY_CSV:
        save_hardware_csv(
            l1_hw_rows,
            l2_hw_rows,
            os.path.join(hardware_dir, f"{model_tag}_hardware_summary.csv")
        )

    # -----------------------------------------------------
    # Hardware summary plots
    # -----------------------------------------------------
    plot_hw_scatter(
        l1_hw_rows, l2_hw_rows,
        x_key="slice_reduction",
        x_label="Slice Reduction (%)",
        y_key="acc_drop",
        y_label="Accuracy Drop from Baseline (%)",
        title=f"{model_tag}: Slice Reduction vs Accuracy Drop",
        out_file=os.path.join(hardware_dir, f"{model_tag}_hw_slice_reduction_vs_acc_drop.png")
    )

    plot_hw_scatter(
        l1_hw_rows, l2_hw_rows,
        x_key="pin_reduction_rate",
        x_label="Pin Reduction Rate (%)",
        y_key="acc_drop",
        y_label="Accuracy Drop from Baseline (%)",
        title=f"{model_tag}: Pin Reduction vs Accuracy Drop",
        out_file=os.path.join(hardware_dir, f"{model_tag}_hw_pin_reduction_vs_acc_drop.png")
    )

    plot_hw_scatter(
        l1_hw_rows, l2_hw_rows,
        x_key="total_dead",
        x_label="Total Dead",
        y_key="acc_drop",
        y_label="Accuracy Drop from Baseline (%)",
        title=f"{model_tag}: Total Dead vs Accuracy Drop",
        out_file=os.path.join(hardware_dir, f"{model_tag}_hw_total_dead_vs_acc_drop.png")
    )

    plot_hw_scatter(
        l1_hw_rows, l2_hw_rows,
        x_key="pack_ratio",
        x_label="Pack Ratio",
        y_key="slice_reduction",
        y_label="Slice Reduction (%)",
        title=f"{model_tag}: Pack Ratio vs Slice Reduction",
        out_file=os.path.join(hardware_dir, f"{model_tag}_hw_pack_ratio_vs_slice_reduction.png")
    )

    plot_hw_scatter(
        l1_hw_rows, l2_hw_rows,
        x_key="global_sparsity",
        x_label="Global Sparsity",
        y_key="pin_reduction_rate",
        y_label="Pin Reduction Rate (%)",
        title=f"{model_tag}: Sparsity vs Pin Reduction",
        out_file=os.path.join(hardware_dir, f"{model_tag}_hw_sparsity_vs_pin_reduction.png")
    )


# =========================================================
# Main
# =========================================================
def main():
    make_dir(OUTPUT_ROOT)

    model_tags = find_model_tags()
    if not model_tags:
        print("No valid model log pairs found.")
        print("Expected files like log1x_l1.txt / log1x_l2.txt")
        return

    print("Detected model tags:", model_tags)

    for model_tag in model_tags:
        process_model(model_tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
