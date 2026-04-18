"""
Unified analysis script for instantaneous frequency feature experiment.

Evaluates radioml_cnn_ifreq_best.pt on test set and generates:
  1. analysis_summary_ifreq.txt - Human-readable diagnostics
  2. analysis_table_ifreq.csv - Long-format data table
  3. analysis_heatmap_ifreq.png - Class × SNR heatmaps
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from radio_dataloader_ifreq import (
    _load_rml2016a,
    _pack_arrays,
    _split_by_group,
    RadioML2016aDatasetIFreq,
)
from train_cnn_ifreq import RadioMLCNNIFreq


def load_model_and_data(data_path: str, device):
    """Load trained model and test dataset."""
    
    # Load model
    model = RadioMLCNNIFreq(num_classes=11)
    checkpoint = torch.load("radioml_cnn_ifreq_best.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load data
    data_dict = _load_rml2016a(data_path)
    X, y, snr, mods, snrs = _pack_arrays(data_dict)
    train_idx, val_idx, test_idx = _split_by_group(data_dict)
    
    test_ds = RadioML2016aDatasetIFreq(X, y, snr, test_idx, normalize=True, return_snr=True)
    
    return model, test_ds, mods


def collect_predictions(model, test_ds, device):
    """Collect predictions on test set."""
    
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y, s = test_ds[i]
            x = x.unsqueeze(0).to(device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            
            predictions.append({
                'true_label': y.item(),
                'pred_label': pred,
                'confidence': probs[0, pred].item(),
                'snr': s.item(),
                'correct': (pred == y.item())
            })
    
    return pd.DataFrame(predictions)


def compute_class_accuracy(df, mods):
    """Compute per-class accuracy."""
    class_acc = {}
    for i, mod in enumerate(mods):
        subset = df[df['true_label'] == i]
        if len(subset) > 0:
            class_acc[mod] = {
                'accuracy': subset['correct'].mean() * 100,
                'count': len(subset)
            }
    return class_acc


def compute_class_snr_accuracy(df, mods):
    """Compute class × SNR accuracy matrix."""
    snrs = sorted(df['snr'].unique())
    matrix = np.zeros((len(mods), len(snrs)))
    
    for i, mod in enumerate(mods):
        for j, s in enumerate(snrs):
            subset = df[(df['true_label'] == i) & (df['snr'] == s)]
            if len(subset) > 0:
                matrix[i, j] = subset['correct'].mean() * 100
    
    return matrix, snrs


def analyze_wbfm(df, mods):
    """Analyze WBFM-specific failure patterns."""
    wbfm_idx = mods.index('WBFM')
    wbfm_df = df[df['true_label'] == wbfm_idx]
    
    # Overall accuracy
    wbfm_acc = wbfm_df['correct'].mean() * 100
    
    # High/Low SNR accuracy
    high_snr = wbfm_df[wbfm_df['snr'] >= 10]
    low_snr = wbfm_df[wbfm_df['snr'] <= -10]
    high_snr_acc = high_snr['correct'].mean() * 100 if len(high_snr) > 0 else 0
    low_snr_acc = low_snr['correct'].mean() * 100 if len(low_snr) > 0 else 0
    
    # Correct vs incorrect analysis
    correct_wbfm = wbfm_df[wbfm_df['correct']]
    incorrect_wbfm = wbfm_df[~wbfm_df['correct']]
    
    correct_conf = correct_wbfm['confidence'].mean()
    incorrect_conf = incorrect_wbfm['confidence'].mean()
    
    # Misclassification targets
    misclass = []
    for pred_label in incorrect_wbfm['pred_label'].unique():
        subset = incorrect_wbfm[incorrect_wbfm['pred_label'] == pred_label]
        misclass.append({
            'target_class': mods[pred_label],
            'count': len(subset),
            'pct': len(subset) / len(incorrect_wbfm) * 100,
            'avg_confidence': subset['confidence'].mean()
        })
    misclass = sorted(misclass, key=lambda x: x['count'], reverse=True)
    
    return {
        'overall_acc': wbfm_acc,
        'high_snr_acc': high_snr_acc,
        'low_snr_acc': low_snr_acc,
        'correct_conf': correct_conf,
        'incorrect_conf': incorrect_conf,
        'misclass': misclass
    }


def analyze_am_ssb_sink(df, mods):
    """Analyze AM-SSB sink behavior."""
    am_ssb_idx = mods.index('AM-SSB')
    am_ssb_pred = df[df['pred_label'] == am_ssb_idx]
    
    # Correct vs sink
    correct = am_ssb_pred[am_ssb_pred['pred_label'] == am_ssb_pred['true_label']]
    sink = am_ssb_pred[am_ssb_pred['pred_label'] != am_ssb_pred['true_label']]
    
    sink_rate = len(sink) / len(am_ssb_pred) * 100 if len(am_ssb_pred) > 0 else 0
    
    correct_conf = correct['confidence'].mean() if len(correct) > 0 else 0
    sink_conf = sink['confidence'].mean() if len(sink) > 0 else 0
    
    # Sink sources
    sources = []
    for true_label in sink['true_label'].unique():
        subset = sink[sink['true_label'] == true_label]
        sources.append({
            'source_class': mods[true_label],
            'count': len(subset),
            'pct': len(subset) / len(sink) * 100,
            'avg_confidence': subset['confidence'].mean()
        })
    sources = sorted(sources, key=lambda x: x['count'], reverse=True)
    
    return {
        'correct_count': len(correct),
        'sink_count': len(sink),
        'sink_rate': sink_rate,
        'correct_conf': correct_conf,
        'sink_conf': sink_conf,
        'sources': sources
    }


def generate_summary(predictions_df, class_acc, wbfm_analysis, am_ssb_analysis, mods):
    """Generate human-readable summary."""
    
    overall_acc = predictions_df['correct'].mean() * 100
    
    summary = []
    summary.append("=" * 80)
    summary.append("INSTANTANEOUS FREQUENCY FEATURE - ANALYSIS SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Overall stats
    summary.append("[OVERALL PERFORMANCE]")
    summary.append(f"Test Accuracy: {overall_acc:.2f}%")
    summary.append("")
    
    # Per-class accuracy
    summary.append("[CLASS-WISE ACCURACY]")
    for mod in mods:
        if mod in class_acc:
            acc = class_acc[mod]['accuracy']
            summary.append(f"  {mod:10s}: {acc:6.2f}%")
    summary.append("")
    
    # WBFM analysis
    summary.append("[WBFM ANALYSIS]")
    summary.append(f"Overall WBFM Accuracy: {wbfm_analysis['overall_acc']:.2f}%")
    summary.append(f"  High SNR (≥10 dB): {wbfm_analysis['high_snr_acc']:.2f}%")
    summary.append(f"  Low SNR (≤-10 dB): {wbfm_analysis['low_snr_acc']:.2f}%")
    summary.append("")
    summary.append("WBFM Correct vs Incorrect:")
    summary.append(f"  Correct confidence: {wbfm_analysis['correct_conf']:.4f}")
    summary.append(f"  Incorrect confidence: {wbfm_analysis['incorrect_conf']:.4f}")
    summary.append(f"  Gap: {wbfm_analysis['correct_conf'] - wbfm_analysis['incorrect_conf']:.4f}")
    summary.append("")
    
    summary.append("WBFM Misclassification Targets (top 5):")
    for i, mc in enumerate(wbfm_analysis['misclass'][:5]):
        summary.append(f"  {mc['target_class']:10s}: {mc['count']:4d} ({mc['pct']:5.1f}%) - conf: {mc['avg_confidence']:.4f}")
    summary.append("")
    
    # AM-SSB sink
    summary.append("[AM-SSB SINK ANALYSIS]")
    summary.append(f"AM-SSB Predictions: {am_ssb_analysis['correct_count']} correct, {am_ssb_analysis['sink_count']} sink")
    summary.append(f"Sink Rate: {am_ssb_analysis['sink_rate']:.2f}%")
    summary.append("")
    summary.append("Confidence Comparison:")
    summary.append(f"  Correct AM-SSB: {am_ssb_analysis['correct_conf']:.4f}")
    summary.append(f"  Sink (wrong) AM-SSB: {am_ssb_analysis['sink_conf']:.4f}")
    summary.append("")
    summary.append("Top Sink Sources:")
    for i, src in enumerate(am_ssb_analysis['sources'][:5]):
        summary.append(f"  {src['source_class']:10s}: {src['count']:4d} ({src['pct']:5.1f}%) - conf: {src['avg_confidence']:.4f}")
    summary.append("")
    summary.append("=" * 80)
    
    return "\n".join(summary)


def build_output_table(predictions_df, class_acc, wbfm_analysis, am_ssb_analysis, mods, snrs):
    """Build long-format CSV table."""
    
    rows = []
    
    # Global stats
    rows.append({
        'table_type': 'global_stats',
        'metric': 'overall_accuracy',
        'value': predictions_df['correct'].mean() * 100,
        'snr': None,
        'class': None
    })
    
    # Per-class accuracy
    for mod in mods:
        if mod in class_acc:
            rows.append({
                'table_type': 'class_accuracy',
                'metric': 'accuracy',
                'value': class_acc[mod]['accuracy'],
                'snr': None,
                'class': mod
            })
    
    # WBFM stats
    rows.append({
        'table_type': 'wbfm_stats',
        'metric': 'overall_acc',
        'value': wbfm_analysis['overall_acc'],
        'snr': None,
        'class': 'WBFM'
    })
    rows.append({
        'table_type': 'wbfm_stats',
        'metric': 'high_snr_acc',
        'value': wbfm_analysis['high_snr_acc'],
        'snr': None,
        'class': 'WBFM'
    })
    rows.append({
        'table_type': 'wbfm_stats',
        'metric': 'low_snr_acc',
        'value': wbfm_analysis['low_snr_acc'],
        'snr': None,
        'class': 'WBFM'
    })
    
    # AM-SSB stats
    rows.append({
        'table_type': 'am_ssb_stats',
        'metric': 'sink_rate',
        'value': am_ssb_analysis['sink_rate'],
        'snr': None,
        'class': 'AM-SSB'
    })
    
    df_out = pd.DataFrame(rows)
    return df_out


def plot_heatmaps(predictions_df, mods, snrs):
    """Plot class × SNR heatmaps."""
    
    matrix, snr_sorted = compute_class_snr_accuracy(predictions_df, mods)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Class × SNR accuracy
    sns.heatmap(
        matrix,
        xticklabels=snr_sorted,
        yticklabels=mods,
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
        ax=axes[0],
        cbar_kws={'label': 'Accuracy (%)'},
        annot=True,
        fmt='.0f',
        linewidths=0.5
    )
    axes[0].set_title('Class × SNR Accuracy (%)')
    axes[0].set_xlabel('SNR Level')
    axes[0].set_ylabel('Modulation')
    
    # AM-SSB sink by SNR
    am_ssb_idx = mods.index('AM-SSB')
    sink_matrix = np.zeros((len(mods), len(snr_sorted)))
    
    for i, mod in enumerate(mods):
        for j, s in enumerate(snr_sorted):
            subset = predictions_df[(predictions_df['pred_label'] == am_ssb_idx) & (predictions_df['snr'] == s)]
            if len(subset) > 0:
                sink_count = ((subset['true_label'] != am_ssb_idx).sum())
                sink_matrix[i, j] = 100.0 * sink_count / len(subset) if len(subset) > 0 else 0
    
    sns.heatmap(
        sink_matrix,
        xticklabels=snr_sorted,
        yticklabels=mods,
        cmap='RdYlGn_r',
        vmin=0,
        vmax=100,
        ax=axes[1],
        cbar_kws={'label': '% Predicted as AM-SSB'},
        annot=True,
        fmt='.0f',
        linewidths=0.5
    )
    axes[1].set_title('AM-SSB Sink Effect by SNR (%)')
    axes[1].set_xlabel('SNR Level')
    axes[1].set_ylabel('True Modulation')
    
    plt.tight_layout()
    plt.savefig('analysis_heatmap_ifreq.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved: analysis_heatmap_ifreq.png")


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "RML2016.10a_dict.pkl"
    
    print("Loading model and data...")
    model, test_ds, mods = load_model_and_data(data_path, device)
    
    print("Collecting predictions...")
    predictions_df = collect_predictions(model, test_ds, device)
    
    print("Computing class accuracy...")
    class_acc = compute_class_accuracy(predictions_df, mods)
    
    print("Analyzing WBFM...")
    wbfm_analysis = analyze_wbfm(predictions_df, mods)
    
    print("Analyzing AM-SSB sink...")
    am_ssb_analysis = analyze_am_ssb_sink(predictions_df, mods)
    
    # Get unique SNRs
    snrs = sorted(predictions_df['snr'].unique())
    
    # Generate summary
    summary_text = generate_summary(predictions_df, class_acc, wbfm_analysis, am_ssb_analysis, mods)
    print("\n" + summary_text)
    
    # Save summary
    with open('analysis_summary_ifreq.txt', 'w') as f:
        f.write(summary_text)
    print("Saved: analysis_summary_ifreq.txt")
    
    # Save table
    table_df = build_output_table(predictions_df, class_acc, wbfm_analysis, am_ssb_analysis, mods, snrs)
    table_df.to_csv('analysis_table_ifreq.csv', index=False)
    print("Saved: analysis_table_ifreq.csv")
    
    # Plot heatmaps
    plot_heatmaps(predictions_df, mods, snrs)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
