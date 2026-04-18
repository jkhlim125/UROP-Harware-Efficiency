"""
Deep-dive analysis for RadioML CNN model:
- Class x SNR accuracy matrix
- WBFM-specific behavior
- AM-SSB sink analysis
- Heatmap visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import os

from train_cnn import TraditionalRadioMLCNN, adapt_radioml_input
from radio_dataloader import _load_rml2016a, _pack_arrays, _split_by_group, RadioML2016aDataset
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("[WARN] seaborn not available, skipping heatmaps")


def collect_predictions_by_snr(model, loader, device, num_classes, mod_names, snr_values):
    """
    Collect model predictions organized by (class, SNR) combinations.
    Returns: dict[(class_idx, snr_val)] -> {'pred': [...], 'correct': [...], 'counts': ...}
    """
    model.eval()
    
    # Structure: per_class_snr[(class_idx, snr)] = {'pred': list, 'correct': list}
    per_class_snr = defaultdict(lambda: {'pred': [], 'correct': [], 'total': 0, 'correct_count': 0})
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, snr = batch
            else:
                raise ValueError("Expected SNR values in batch (return_snr=True)")
            
            data = adapt_radioml_input(data).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            snr = snr.cpu().numpy()
            
            logits = model(data)
            pred = logits.argmax(dim=1).cpu().numpy()
            target_cpu = target.cpu().numpy()
            
            # Group by (true_class, snr)
            for i in range(len(target_cpu)):
                true_class = target_cpu[i]
                snr_val = int(snr[i])
                pred_class = pred[i]
                is_correct = (true_class == pred_class)
                
                key = (true_class, snr_val)
                per_class_snr[key]['pred'].append(pred_class)
                per_class_snr[key]['correct'].append(is_correct)
                per_class_snr[key]['total'] += 1
                if is_correct:
                    per_class_snr[key]['correct_count'] += 1
    
    return per_class_snr


def compute_class_snr_accuracy(per_class_snr, mod_names, snr_values):
    """
    Build class x SNR accuracy matrix.
    rows = classes, columns = SNR values
    """
    num_classes = len(mod_names)
    accuracy_matrix = np.zeros((num_classes, len(snr_values)))
    sample_counts = np.zeros((num_classes, len(snr_values)), dtype=np.int32)
    
    for class_idx in range(num_classes):
        for snr_idx, snr_val in enumerate(snr_values):
            key = (class_idx, snr_val)
            if key in per_class_snr:
                data = per_class_snr[key]
                acc = data['correct_count'] / max(1, data['total'])
                accuracy_matrix[class_idx, snr_idx] = acc * 100
                sample_counts[class_idx, snr_idx] = data['total']
            else:
                accuracy_matrix[class_idx, snr_idx] = np.nan
                sample_counts[class_idx, snr_idx] = 0
    
    return accuracy_matrix, sample_counts


def print_class_snr_table(accuracy_matrix, mod_names, snr_values):
    """Pretty print class x SNR accuracy table."""
    print("\n" + "="*140)
    print("CLASS x SNR ACCURACY MATRIX (% correct)")
    print("="*140)
    print(f"{'Class':<12}", end="")
    for snr in snr_values:
        print(f"{snr:>6d}dB", end=" ")
    print(f"{'| Mean':<10}")
    print("-" * 140)
    
    for i, mod in enumerate(mod_names):
        print(f"{mod:<12}", end="")
        row = accuracy_matrix[i, :]
        for acc in row:
            if np.isnan(acc):
                print(f"{'N/A':>6}", end=" ")
            else:
                print(f"{acc:>6.1f}", end=" ")
        row_mean = np.nanmean(row)
        print(f"| {row_mean:>6.1f}%")
    
    print("-" * 140)
    print(f"{'Mean':<12}", end="")
    for j in range(len(snr_values)):
        col = accuracy_matrix[:, j]
        col_mean = np.nanmean(col)
        print(f"{col_mean:>6.1f}", end=" ")
    overall_mean = np.nanmean(accuracy_matrix)
    print(f"| {overall_mean:>6.1f}%")


def analyze_wbfm(per_class_snr, mod_names, snr_values, num_classes):
    """Detailed analysis of WBFM performance."""
    print("\n" + "="*100)
    print("WBFM DETAILED ANALYSIS")
    print("="*100)
    
    wbfm_idx = mod_names.index('WBFM')
    
    # WBFM accuracy per SNR
    print(f"\nWBFM Accuracy by SNR:")
    print(f"{'SNR (dB)':<12} {'Accuracy':<12} {'Samples':<12}")
    print("-" * 36)
    
    wbfm_by_snr = {}
    for snr_val in sorted(snr_values):
        key = (wbfm_idx, snr_val)
        if key in per_class_snr:
            data = per_class_snr[key]
            acc = data['correct_count'] / max(1, data['total']) * 100
            wbfm_by_snr[snr_val] = {'acc': acc, 'total': data['total']}
            print(f"{snr_val:<12d} {acc:<12.2f} {data['total']:<12d}")
    
    # Find high and low SNR performance
    high_snr_vals = [snr for snr in sorted(snr_values) if snr >= 10]
    low_snr_vals = [snr for snr in sorted(snr_values) if snr <= -10]
    
    if high_snr_vals:
        high_acc = np.mean([wbfm_by_snr[snr]['acc'] for snr in high_snr_vals if snr in wbfm_by_snr])
        print(f"\n  At high SNR (>= 10 dB): {high_acc:.2f}%")
    
    if low_snr_vals:
        low_acc = np.mean([wbfm_by_snr[snr]['acc'] for snr in low_snr_vals if snr in wbfm_by_snr])
        print(f"  At low SNR (<= -10 dB): {low_acc:.2f}%")
    
    # Where is WBFM misclassified?
    print(f"\nWBFM Misclassification Analysis:")
    misclass_by_target = defaultdict(int)
    total_samples = 0
    
    for snr_val in snr_values:
        key = (wbfm_idx, snr_val)
        if key in per_class_snr:
            preds = per_class_snr[key]['pred']
            corrects = per_class_snr[key]['correct']
            for pred, correct in zip(preds, corrects):
                total_samples += 1
                if not correct:
                    misclass_by_target[pred] += 1
    
    top_confusions = sorted(misclass_by_target.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"{'Predicted As':<15} {'Count':<10} {'% of errors':<15}")
    print("-" * 40)
    total_errors = sum(misclass_by_target.values())
    for pred_idx, count in top_confusions:
        pct = count / max(1, total_errors) * 100
        print(f"{mod_names[pred_idx]:<15} {count:<10d} {pct:<14.1f}%")
    
    print(f"\nTotal WBFM samples: {total_samples}")
    print(f"Total errors: {total_errors} ({total_errors/max(1,total_samples)*100:.1f}%)")


def compute_am_ssb_sink(per_class_snr, mod_names, snr_values, num_classes):
    """Analyze how often each class is confused with AM-SSB, by SNR."""
    print("\n" + "="*100)
    print("AM-SSB SINK ANALYSIS: % of class X samples predicted as AM-SSB")
    print("="*100)
    
    am_ssb_idx = mod_names.index('AM-SSB')
    
    # Build matrix: rows = true class, columns = SNR, values = % predicted as AM-SSB
    am_ssb_sink_matrix = np.zeros((num_classes, len(snr_values)))
    sample_counts = np.zeros((num_classes, len(snr_values)), dtype=np.int32)
    
    for class_idx in range(num_classes):
        for snr_idx, snr_val in enumerate(snr_values):
            key = (class_idx, snr_val)
            if key in per_class_snr:
                preds = per_class_snr[key]['pred']
                total = len(preds)
                am_ssb_count = sum(1 for p in preds if p == am_ssb_idx)
                am_ssb_sink_matrix[class_idx, snr_idx] = (am_ssb_count / max(1, total)) * 100
                sample_counts[class_idx, snr_idx] = total
    
    # Print AM-SSB sink table
    print(f"\n{'Class':<12}", end="")
    for snr in snr_values:
        print(f"{snr:>6d}dB", end=" ")
    print(f"{'| Mean':<10}")
    print("-" * 140)
    
    for i, mod in enumerate(mod_names):
        print(f"{mod:<12}", end="")
        row = am_ssb_sink_matrix[i, :]
        for val in row:
            print(f"{val:>6.1f}", end=" ")
        row_mean = np.nanmean(row)
        print(f"| {row_mean:>6.1f}%")
    
    print("-" * 140)
    print(f"{'Mean':<12}", end="")
    for j in range(len(snr_values)):
        col = am_ssb_sink_matrix[:, j]
        col_mean = np.nanmean(col)
        print(f"{col_mean:>6.1f}", end=" ")
    overall_mean = np.nanmean(am_ssb_sink_matrix)
    print(f"| {overall_mean:>6.1f}%")
    
    return am_ssb_sink_matrix


def save_dataframes(accuracy_matrix, am_ssb_sink_matrix, mod_names, snr_values):
    """Save analysis results as CSV files."""
    print("\n" + "="*100)
    print("SAVING ANALYSIS RESULTS")
    print("="*100)
    
    # Class x SNR accuracy
    df_acc = pd.DataFrame(accuracy_matrix, index=mod_names, columns=[f'{snr}dB' for snr in snr_values])
    csv_path = "class_snr_accuracy.csv"
    df_acc.to_csv(csv_path)
    print(f"\nSaved class x SNR accuracy to: {csv_path}")
    
    # AM-SSB sink matrix
    df_sink = pd.DataFrame(am_ssb_sink_matrix, index=mod_names, columns=[f'{snr}dB' for snr in snr_values])
    csv_sink_path = "am_ssb_sink_by_snr.csv"
    df_sink.to_csv(csv_sink_path)
    print(f"Saved AM-SSB sink matrix to: {csv_sink_path}")
    
    return df_acc, df_sink


def plot_heatmaps(df_acc, df_sink, mod_names):
    """Create heatmap visualizations."""
    if not SEABORN_AVAILABLE:
        print("\n[WARN] seaborn/matplotlib not available, skipping heatmaps")
        return
    
    print("\nGenerating heatmaps...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Class x SNR Accuracy heatmap
    sns.heatmap(df_acc, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100, 
                ax=axes[0], cbar_kws={'label': 'Accuracy (%)'})
    axes[0].set_title('Class x SNR Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('SNR Level')
    axes[0].set_ylabel('Modulation')
    
    # AM-SSB sink heatmap
    sns.heatmap(df_sink, annot=True, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=100, 
                ax=axes[1], cbar_kws={'label': '% Confused as AM-SSB'})
    axes[1].set_title('AM-SSB Sink Effect by SNR', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('SNR Level')
    axes[1].set_ylabel('True Modulation')
    
    plt.tight_layout()
    heatmap_path = "analysis_heatmaps.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmaps to: {heatmap_path}")
    plt.close()


def main():
    # Load model and data
    checkpoint_path = "radioml_cnn_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = checkpoint["num_classes"]
    
    model = TraditionalRadioMLCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded model from {checkpoint_path}")
    
    # Find and load data
    data_path = None
    candidates = [
        "/home/jliangbr/workspace/LutNet/no_param/quant_0_1/data/RML2016.10a_dict.pkl",
        "../data/RML2016.10a_dict.pkl",
        "./data/RML2016.10a_dict.pkl",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            data_path = candidate
            break
    
    if not data_path:
        print("[ERROR] Could not find RML2016.10a_dict.pkl")
        sys.exit(1)
    
    print(f"[OK] Using data file: {data_path}")
    data_dict = _load_rml2016a(data_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(data_dict, seed=1)
    
    # Create test loader with SNR
    test_ds = RadioML2016aDataset(X, y, snr, test_idx, normalize=True, as_2d=False, return_snr=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    
    mod_names = mods
    snr_values = snrs
    
    print(f"\nDataset: {len(mod_names)} modulations, {len(snr_values)} SNR levels")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Collect predictions grouped by (class, SNR)
    print("\nCollecting predictions by (class, SNR)...")
    per_class_snr = collect_predictions_by_snr(model, test_loader, device, num_classes, mod_names, snr_values)
    
    # Compute class x SNR accuracy matrix
    print("Computing class x SNR accuracy matrix...")
    accuracy_matrix, sample_counts = compute_class_snr_accuracy(per_class_snr, mod_names, snr_values)
    print_class_snr_table(accuracy_matrix, mod_names, snr_values)
    
    # WBFM analysis
    analyze_wbfm(per_class_snr, mod_names, snr_values, num_classes)
    
    # AM-SSB sink analysis
    am_ssb_sink_matrix = compute_am_ssb_sink(per_class_snr, mod_names, snr_values, num_classes)
    
    # Save to CSV
    df_acc, df_sink = save_dataframes(accuracy_matrix, am_ssb_sink_matrix, mod_names, snr_values)
    
    # Plot heatmaps
    plot_heatmaps(df_acc, df_sink, mod_names)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Generated files:")
    print(f"  - class_snr_accuracy.csv: Class x SNR accuracy matrix")
    print(f"  - am_ssb_sink_by_snr.csv: % of each class confused as AM-SSB, by SNR")
    print(f"  - analysis_heatmaps.png: Visualization (if matplotlib available)")


if __name__ == "__main__":
    main()
