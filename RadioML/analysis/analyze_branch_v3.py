"""
Unified analysis script for refined two-branch architecture (v3).

Evaluates radioml_cnn_branch_v3_best.pt and generates:
  1. analysis_summary_branch_v3.txt
  2. analysis_table_branch_v3.csv
  3. analysis_heatmap_branch_v3.png
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from radio_dataloader_branch_v3 import (
    _load_rml2016a,
    _pack_arrays,
    _split_by_group,
    RadioML2016aDatasetBranchV3,
)
from train_cnn_branch_v3 import RadioMLBranchCNNV3


def load_model_and_data(data_path: str, device):
    """Load trained model and test dataset."""

    model = RadioMLBranchCNNV3(num_classes=11)
    checkpoint = torch.load("radioml_cnn_branch_v3_best.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    data_dict = _load_rml2016a(data_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(data_dict)

    test_ds = RadioML2016aDatasetBranchV3(X, y, snr, test_idx, normalize=True, return_snr=True)

    return model, test_ds, mods


def collect_predictions(model, test_ds, device):
    """Collect predictions on test set."""

    predictions = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            raw_iq, if_feat, y, s = test_ds[i]
            raw_iq = raw_iq.unsqueeze(0).to(device)
            if_feat = if_feat.unsqueeze(0).to(device)

            logits = model(raw_iq, if_feat)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()

            predictions.append(
                {
                    "true_label": y.item(),
                    "pred_label": pred,
                    "confidence": probs[0, pred].item(),
                    "snr": s.item(),
                    "correct": pred == y.item(),
                }
            )

    return pd.DataFrame(predictions)


def compute_class_accuracy(df, mods):
    """Compute per-class accuracy."""

    class_acc = {}
    for i, mod in enumerate(mods):
        subset = df[df["true_label"] == i]
        if len(subset) > 0:
            class_acc[mod] = {"accuracy": subset["correct"].mean() * 100, "count": len(subset)}
    return class_acc


def compute_class_snr_accuracy(df, mods):
    """Compute class x SNR accuracy matrix."""

    snrs = sorted(df["snr"].unique())
    matrix = np.zeros((len(mods), len(snrs)))

    for i, mod in enumerate(mods):
        for j, s in enumerate(snrs):
            subset = df[(df["true_label"] == i) & (df["snr"] == s)]
            if len(subset) > 0:
                matrix[i, j] = subset["correct"].mean() * 100

    return matrix, snrs


def compute_confusion_matrix(df, mods):
    """Compute raw and row-normalized confusion matrices."""

    counts = np.zeros((len(mods), len(mods)), dtype=np.int64)
    for _, row in df.iterrows():
        counts[int(row["true_label"]), int(row["pred_label"])] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts * 100.0,
        np.maximum(row_sums, 1),
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums > 0,
    )
    return counts, normalized


def analyze_wbfm(df, mods):
    """Analyze WBFM-specific failure patterns."""

    wbfm_idx = mods.index("WBFM")
    wbfm_df = df[df["true_label"] == wbfm_idx]

    wbfm_acc = wbfm_df["correct"].mean() * 100

    high_snr = wbfm_df[wbfm_df["snr"] >= 10]
    low_snr = wbfm_df[wbfm_df["snr"] <= -10]
    high_snr_acc = high_snr["correct"].mean() * 100 if len(high_snr) > 0 else 0.0
    low_snr_acc = low_snr["correct"].mean() * 100 if len(low_snr) > 0 else 0.0

    incorrect_wbfm = wbfm_df[~wbfm_df["correct"]]

    misclass = []
    for pred_label in incorrect_wbfm["pred_label"].unique():
        subset = incorrect_wbfm[incorrect_wbfm["pred_label"] == pred_label]
        misclass.append(
            {
                "target_class": mods[pred_label],
                "count": len(subset),
                "pct": len(subset) / len(incorrect_wbfm) * 100 if len(incorrect_wbfm) > 0 else 0.0,
                "avg_confidence": subset["confidence"].mean(),
            }
        )
    misclass = sorted(misclass, key=lambda x: x["count"], reverse=True)

    return {
        "overall_acc": wbfm_acc,
        "high_snr_acc": high_snr_acc,
        "low_snr_acc": low_snr_acc,
        "misclass": misclass,
    }


def analyze_am_ssb_sink(df, mods):
    """Analyze AM-SSB sink behavior."""

    am_ssb_idx = mods.index("AM-SSB")
    am_ssb_pred = df[df["pred_label"] == am_ssb_idx]

    correct = am_ssb_pred[am_ssb_pred["pred_label"] == am_ssb_pred["true_label"]]
    sink = am_ssb_pred[am_ssb_pred["pred_label"] != am_ssb_pred["true_label"]]

    sink_rate = len(sink) / len(am_ssb_pred) * 100 if len(am_ssb_pred) > 0 else 0.0

    sources = []
    for true_label in sink["true_label"].unique():
        subset = sink[sink["true_label"] == true_label]
        sources.append(
            {
                "source_class": mods[true_label],
                "count": len(subset),
                "pct": len(subset) / len(sink) * 100 if len(sink) > 0 else 0.0,
            }
        )
    sources = sorted(sources, key=lambda x: x["count"], reverse=True)

    return {
        "correct_count": len(correct),
        "sink_count": len(sink),
        "sink_rate": sink_rate,
        "sources": sources,
    }


def format_confusion_matrix(normalized_matrix, mods):
    """Format the normalized confusion matrix for the summary file."""

    header = "true\\pred".ljust(10) + " " + " ".join(f"{mod:>8s}" for mod in mods)
    lines = [header]
    for i, mod in enumerate(mods):
        row = " ".join(f"{normalized_matrix[i, j]:8.1f}" for j in range(len(mods)))
        lines.append(f"{mod:10s} {row}")
    return lines


def generate_summary(
    predictions_df,
    class_acc,
    class_snr_matrix,
    snrs,
    wbfm_analysis,
    am_ssb_analysis,
    confusion_counts,
    confusion_normalized,
    mods,
):
    """Generate human-readable summary."""

    overall_acc = predictions_df["correct"].mean() * 100

    summary = []
    summary.append("=" * 80)
    summary.append("TWO-BRANCH ARCHITECTURE REFINED (V3) - ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")

    summary.append("[OVERALL PERFORMANCE]")
    summary.append(f"Test Accuracy: {overall_acc:.2f}%")
    summary.append("")

    summary.append("[CLASS-WISE ACCURACY]")
    for mod in mods:
        if mod in class_acc:
            summary.append(f"  {mod:10s}: {class_acc[mod]['accuracy']:6.2f}%")
    summary.append("")

    summary.append("[WBFM ANALYSIS]")
    summary.append(f"Overall WBFM Accuracy: {wbfm_analysis['overall_acc']:.2f}%")
    summary.append(f"  High SNR (>=10 dB): {wbfm_analysis['high_snr_acc']:.2f}%")
    summary.append(f"  Low SNR (<=-10 dB): {wbfm_analysis['low_snr_acc']:.2f}%")
    summary.append("WBFM Top Confusion Targets:")
    for item in wbfm_analysis["misclass"][:5]:
        summary.append(
            f"  {item['target_class']:10s}: {item['count']:4d} ({item['pct']:5.1f}%)"
        )
    summary.append("")

    summary.append("[AM-SSB SINK ANALYSIS]")
    summary.append(
        f"AM-SSB Predictions: {am_ssb_analysis['correct_count']} correct, "
        f"{am_ssb_analysis['sink_count']} sink"
    )
    summary.append(f"Sink Rate: {am_ssb_analysis['sink_rate']:.2f}%")
    summary.append("Top Sink Sources:")
    for item in am_ssb_analysis["sources"][:5]:
        summary.append(
            f"  {item['source_class']:10s}: {item['count']:4d} ({item['pct']:5.1f}%)"
        )
    summary.append("")

    summary.append("[CLASS X SNR ACCURACY]")
    summary.append("Rows are classes, columns are SNR levels.")
    snr_header = " " * 12 + " ".join(f"{snr:>6d}" for snr in snrs)
    summary.append(snr_header)
    for i, mod in enumerate(mods):
        row = " ".join(f"{class_snr_matrix[i, j]:6.1f}" for j in range(len(snrs)))
        summary.append(f"{mod:12s}{row}")
    summary.append("")

    summary.append("[CONFUSION MATRIX - ROW NORMALIZED %]")
    summary.extend(format_confusion_matrix(confusion_normalized, mods))
    summary.append("")

    summary.append("[NOTES]")
    summary.append("1. V3 keeps the v2 two-branch structure and removes class weighting.")
    summary.append("2. IF branch input now uses a medium-size interpretable feature set.")
    summary.append("3. Features include IF, smoothed IF, IF magnitude/energy/variance, and amplitude descriptors.")
    summary.append("")
    summary.append("=" * 80)

    return "\n".join(summary)


def build_output_table(
    predictions_df,
    class_acc,
    class_snr_matrix,
    snrs,
    wbfm_analysis,
    am_ssb_analysis,
    confusion_counts,
    confusion_normalized,
    mods,
):
    """Build compact CSV table."""

    rows = []

    rows.append(
        {
            "metric": "overall_accuracy",
            "value": predictions_df["correct"].mean() * 100,
            "class": None,
            "snr": None,
            "target_class": None,
            "count": None,
        }
    )

    for mod in mods:
        if mod in class_acc:
            rows.append(
                {
                    "metric": "class_accuracy",
                    "value": class_acc[mod]["accuracy"],
                    "class": mod,
                    "snr": None,
                    "target_class": None,
                    "count": class_acc[mod]["count"],
                }
            )

    for i, mod in enumerate(mods):
        for j, snr in enumerate(snrs):
            rows.append(
                {
                    "metric": "class_snr_accuracy",
                    "value": class_snr_matrix[i, j],
                    "class": mod,
                    "snr": snr,
                    "target_class": None,
                    "count": None,
                }
            )

    rows.append(
        {
            "metric": "wbfm_overall_acc",
            "value": wbfm_analysis["overall_acc"],
            "class": "WBFM",
            "snr": None,
            "target_class": None,
            "count": None,
        }
    )
    rows.append(
        {
            "metric": "wbfm_high_snr_acc",
            "value": wbfm_analysis["high_snr_acc"],
            "class": "WBFM",
            "snr": None,
            "target_class": None,
            "count": None,
        }
    )
    rows.append(
        {
            "metric": "wbfm_low_snr_acc",
            "value": wbfm_analysis["low_snr_acc"],
            "class": "WBFM",
            "snr": None,
            "target_class": None,
            "count": None,
        }
    )

    for item in wbfm_analysis["misclass"][:5]:
        rows.append(
            {
                "metric": "wbfm_top_confusion",
                "value": item["pct"],
                "class": "WBFM",
                "snr": None,
                "target_class": item["target_class"],
                "count": item["count"],
            }
        )

    rows.append(
        {
            "metric": "am_ssb_sink_rate",
            "value": am_ssb_analysis["sink_rate"],
            "class": "AM-SSB",
            "snr": None,
            "target_class": None,
            "count": am_ssb_analysis["sink_count"],
        }
    )

    for i, true_mod in enumerate(mods):
        for j, pred_mod in enumerate(mods):
            rows.append(
                {
                    "metric": "confusion_matrix_pct",
                    "value": confusion_normalized[i, j],
                    "class": true_mod,
                    "snr": None,
                    "target_class": pred_mod,
                    "count": confusion_counts[i, j],
                }
            )

    return pd.DataFrame(rows)


def plot_heatmaps(predictions_df, mods):
    """Plot class x SNR accuracy and row-normalized confusion matrix."""

    class_snr_matrix, snr_sorted = compute_class_snr_accuracy(predictions_df, mods)
    _, confusion_normalized = compute_confusion_matrix(predictions_df, mods)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        class_snr_matrix,
        xticklabels=snr_sorted,
        yticklabels=mods,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        ax=axes[0],
        cbar_kws={"label": "Accuracy (%)"},
        annot=True,
        fmt=".0f",
        linewidths=0.5,
    )
    axes[0].set_title("Class x SNR Accuracy (%) - V3")
    axes[0].set_xlabel("SNR Level")
    axes[0].set_ylabel("Modulation")

    sns.heatmap(
        confusion_normalized,
        xticklabels=mods,
        yticklabels=mods,
        cmap="Blues",
        vmin=0,
        vmax=100,
        ax=axes[1],
        cbar_kws={"label": "Row-normalized (%)"},
        annot=True,
        fmt=".0f",
        linewidths=0.5,
    )
    axes[1].set_title("Confusion Matrix (%) - V3")
    axes[1].set_xlabel("Predicted Modulation")
    axes[1].set_ylabel("True Modulation")

    plt.tight_layout()
    plt.savefig("analysis_heatmap_branch_v3.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: analysis_heatmap_branch_v3.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "RML2016.10a_dict.pkl"

    print("Loading model and data...")
    model, test_ds, mods = load_model_and_data(data_path, device)

    print("Collecting predictions...")
    predictions_df = collect_predictions(model, test_ds, device)

    print("Computing class accuracy...")
    class_acc = compute_class_accuracy(predictions_df, mods)
    class_snr_matrix, snrs = compute_class_snr_accuracy(predictions_df, mods)

    print("Analyzing WBFM...")
    wbfm_analysis = analyze_wbfm(predictions_df, mods)

    print("Analyzing AM-SSB sink...")
    am_ssb_analysis = analyze_am_ssb_sink(predictions_df, mods)

    print("Computing confusion matrix...")
    confusion_counts, confusion_normalized = compute_confusion_matrix(predictions_df, mods)

    summary_text = generate_summary(
        predictions_df,
        class_acc,
        class_snr_matrix,
        snrs,
        wbfm_analysis,
        am_ssb_analysis,
        confusion_counts,
        confusion_normalized,
        mods,
    )
    print("\n" + summary_text)

    with open("analysis_summary_branch_v3.txt", "w") as f:
        f.write(summary_text)
    print("Saved: analysis_summary_branch_v3.txt")

    table_df = build_output_table(
        predictions_df,
        class_acc,
        class_snr_matrix,
        snrs,
        wbfm_analysis,
        am_ssb_analysis,
        confusion_counts,
        confusion_normalized,
        mods,
    )
    table_df.to_csv("analysis_table_branch_v3.csv", index=False)
    print("Saved: analysis_table_branch_v3.csv")

    plot_heatmaps(predictions_df, mods)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
