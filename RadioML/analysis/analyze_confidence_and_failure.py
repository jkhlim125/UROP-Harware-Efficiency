"""
Confidence and Failure Analysis for RadioML CNN

Goal: Diagnose WHY the model fails using confidence-based analysis
Focus: WBFM and AM-SSB "sink" behavior

Input: Use existing checkpoint, dataloader, test split
Output: 
  - analysis_summary.txt (human-readable interpretation)
  - analysis_tables.csv (unified long-format table)
  - analysis_heatmaps.png (visualizations)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
import sys
import os

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib/seaborn not available, skipping heatmaps")


def setup_imports():
    """Set up dynamic imports for model and dataloader."""
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    
    # Add paths
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Determine if baseline or STFT based on checkpoint file
    checkpoint_path = "radioml_cnn_best.pt"
    is_stft = False
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "radioml_cnn_stft_best.pt"
        is_stft = True
    
    if not os.path.exists(checkpoint_path):
        print("[ERROR] No checkpoint found (radioml_cnn_best.pt or radioml_cnn_stft_best.pt)")
        sys.exit(1)
    
    # Import baseline components
    try:
        from train_cnn import (
            TraditionalRadioMLCNN,
            adapt_radioml_input
        )
        from radio_dataloader import (
            _load_rml2016a, 
            _pack_arrays, 
            _split_by_group, 
            RadioML2016aDataset,
            get_radioml2016a_dataloaders as get_baseline_dataloaders
        )
        print(f"[OK] Loaded baseline imports (train_cnn, radio_dataloader)")
    except ImportError as e:
        print(f"[WARN] Baseline import failed: {e}")
        TraditionalRadioMLCNN = None
        adapt_radioml_input = None
        _load_rml2016a = None
        _pack_arrays = None
        _split_by_group = None
        RadioML2016aDataset = None
        get_baseline_dataloaders = None
    
    # Import STFT components
    try:
        from train_cnn_stft import RadioMLConv2D
        from radio_dataloader_stft import get_radioml2016a_dataloaders as get_stft_dataloaders
        print(f"[OK] Loaded STFT imports (train_cnn_stft, radio_dataloader_stft)")
    except ImportError as e:
        print(f"[WARN] STFT import failed: {e}")
        RadioMLConv2D = None
        get_stft_dataloaders = None
    
    if is_stft:
        if RadioMLConv2D is None:
            print("[ERROR] Cannot load STFT model")
            sys.exit(1)
        return {
            'model_class': RadioMLConv2D,
            'is_stft': True,
            'checkpoint_path': checkpoint_path,
            'get_dataloaders': get_stft_dataloaders,
            'baseline_utils': None,
            'adapt_input': None,
        }
    else:
        if TraditionalRadioMLCNN is None:
            print("[ERROR] Cannot load baseline model")
            sys.exit(1)
        return {
            'model_class': TraditionalRadioMLCNN,
            'is_stft': False,
            'checkpoint_path': checkpoint_path,
            'get_dataloaders': get_baseline_dataloaders,
            'baseline_utils': {
                '_load_rml2016a': _load_rml2016a,
                '_pack_arrays': _pack_arrays,
                '_split_by_group': _split_by_group,
                'RadioML2016aDataset': RadioML2016aDataset,
            },
            'adapt_input': adapt_radioml_input,
        }


# Perform imports
IMPORT_CONFIG = setup_imports()
RadioMLCNN = IMPORT_CONFIG['model_class']
IS_STFT = IMPORT_CONFIG['is_stft']
GET_DATALOADERS = IMPORT_CONFIG['get_dataloaders']
BASELINE_UTILS = IMPORT_CONFIG['baseline_utils']


def load_checkpoint():
    """Load trained model checkpoint."""
    checkpoint_path = IMPORT_CONFIG['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def build_model(checkpoint):
    """Build and load model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = checkpoint["num_classes"]
    
    if IS_STFT:
        # STFT model
        input_shape = checkpoint.get("input_shape", [2, 33, 8])
        freq_bins, time_frames = input_shape[1], input_shape[2]
        model = RadioMLCNN(num_classes=num_classes, freq_bins=freq_bins, time_frames=time_frames).to(device)
    else:
        # Baseline model
        model = RadioMLCNN(num_classes=num_classes).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, device


def get_dataloader(checkpoint):
    """Load test dataloader with SNR info."""
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
    
    if IS_STFT:
        # STFT model - use standard STFT dataloader
        stft_params = checkpoint.get("stft_params", {"n_fft": 64, "hop_length": 16})
        _, _, test_loader, _, meta = GET_DATALOADERS(
            pkl_path=data_path,
            batch_size=512,
            test_batch_size=512,
            num_workers=4,
            normalize=True,
            distributed=False,
            pin_memory=True,
            n_fft=stft_params.get("n_fft", 64),
            hop_length=stft_params.get("hop_length", 16),
            return_snr=True,
        )
    else:
        # Baseline model - use low-level functions to create dataset with SNR support
        _load_rml2016a = BASELINE_UTILS['_load_rml2016a']
        _pack_arrays = BASELINE_UTILS['_pack_arrays']
        _split_by_group = BASELINE_UTILS['_split_by_group']
        RadioML2016aDataset = BASELINE_UTILS['RadioML2016aDataset']
        
        # Load data
        data_dict = _load_rml2016a(data_path)
        X, y, snr, mods, snrs = _pack_arrays(data_dict)
        train_idx, val_idx, test_idx = _split_by_group(data_dict)
        
        # Create test dataset with return_snr=True
        test_ds = RadioML2016aDataset(
            X, y, snr, test_idx, 
            normalize=True, 
            as_2d=False, 
            return_snr=True,  # KEY: Enable SNR return for analysis
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_ds,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        meta = {
            "mods": mods,
            "snrs": snrs,
            "num_classes": len(mods),
            "input_shape": [2, 128],
        }
    
    return test_loader, meta


def collect_predictions(model, loader, device, num_classes, mod_names, snr_values):
    """
    Run inference and collect predictions with confidence scores.
    
    Returns:
        DataFrame with columns: true_label, pred_label, conf_top1, conf_top2, margin, snr, is_correct
    """
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                data, target, snr = batch
            else:
                print("[ERROR] Expected SNR in batch")
                sys.exit(1)
            
            # Apply adapt_radioml_input for baseline, otherwise use data as-is
            if not IS_STFT and IMPORT_CONFIG['adapt_input'] is not None:
                data = IMPORT_CONFIG['adapt_input'](data)
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            snr = snr.cpu().numpy()
            
            logits = model(data)
            probs = torch.softmax(logits, dim=1)
            
            # Get top-2 predictions
            top2_probs, top2_indices = torch.topk(probs, k=2, dim=1)
            
            pred_label = top2_indices[:, 0].cpu().numpy()
            true_label = target.cpu().numpy()
            conf_top1 = top2_probs[:, 0].cpu().numpy()
            conf_top2 = top2_probs[:, 1].cpu().numpy()
            margin = conf_top1 - conf_top2
            is_correct = (pred_label == true_label).astype(int)
            
            for i in range(len(true_label)):
                results.append({
                    'true_label': true_label[i],
                    'true_class': mod_names[true_label[i]],
                    'pred_label': pred_label[i],
                    'pred_class': mod_names[pred_label[i]],
                    'conf_top1': conf_top1[i],
                    'conf_top2': conf_top2[i],
                    'margin': margin[i],
                    'snr': snr[i],
                    'is_correct': is_correct[i],
                })
    
    return pd.DataFrame(results)


def analyze_global_confidence(df):
    """Compute global confidence statistics."""
    correct = df[df['is_correct'] == 1]
    incorrect = df[df['is_correct'] == 0]
    
    stats = {
        'overall_accuracy': (df['is_correct'].sum() / len(df)) * 100,
        'conf_correct_mean': correct['conf_top1'].mean(),
        'conf_incorrect_mean': incorrect['conf_top1'].mean(),
        'margin_correct_mean': correct['margin'].mean(),
        'margin_incorrect_mean': incorrect['margin'].mean(),
        'num_correct': len(correct),
        'num_incorrect': len(incorrect),
    }
    
    return stats


def analyze_wbfm(df, mod_names):
    """Analyze WBFM failure modes."""
    wbfm_idx = mod_names.index('WBFM')
    wbfm = df[df['true_label'] == wbfm_idx]
    
    wbfm_correct = wbfm[wbfm['is_correct'] == 1]
    wbfm_incorrect = wbfm[wbfm['is_correct'] == 0]
    
    results = {
        'wbfm_total': len(wbfm),
        'wbfm_accuracy': (wbfm_correct.shape[0] / len(wbfm) * 100) if len(wbfm) > 0 else 0,
        'wbfm_conf_correct': wbfm_correct['conf_top1'].mean() if len(wbfm_correct) > 0 else 0,
        'wbfm_conf_incorrect': wbfm_incorrect['conf_top1'].mean() if len(wbfm_incorrect) > 0 else 0,
        'wbfm_margin_correct': wbfm_correct['margin'].mean() if len(wbfm_correct) > 0 else 0,
        'wbfm_margin_incorrect': wbfm_incorrect['margin'].mean() if len(wbfm_incorrect) > 0 else 0,
    }
    
    # Per-SNR accuracy
    snr_accuracy = {}
    for snr in sorted(df['snr'].unique()):
        wbfm_snr = wbfm[wbfm['snr'] == snr]
        if len(wbfm_snr) > 0:
            snr_accuracy[snr] = (wbfm_snr['is_correct'].sum() / len(wbfm_snr)) * 100
    
    results['wbfm_high_snr_acc'] = np.mean([v for k, v in snr_accuracy.items() if k >= 10]) if any(k >= 10 for k in snr_accuracy.keys()) else 0
    results['wbfm_low_snr_acc'] = np.mean([v for k, v in snr_accuracy.items() if k <= -10]) if any(k <= -10 for k in snr_accuracy.keys()) else 0
    
    # Misclassification breakdown
    misclass = {}
    for idx, mod in enumerate(mod_names):
        count = len(wbfm_incorrect[wbfm_incorrect['pred_label'] == idx])
        if count > 0:
            subset = wbfm_incorrect[wbfm_incorrect['pred_label'] == idx]
            misclass[mod] = {
                'count': count,
                'pct': (count / len(wbfm_incorrect)) * 100 if len(wbfm_incorrect) > 0 else 0,
                'conf_mean': subset['conf_top1'].mean(),
                'margin_mean': subset['margin'].mean(),
            }
    
    results['wbfm_misclass'] = misclass
    results['wbfm_snr_accuracy'] = snr_accuracy
    
    return results


def analyze_am_ssb_sink(df, mod_names):
    """Analyze AM-SSB sink behavior."""
    am_ssb_idx = mod_names.index('AM-SSB')
    am_ssb_preds = df[df['pred_label'] == am_ssb_idx]
    
    correct_am_ssb = am_ssb_preds[am_ssb_preds['true_label'] == am_ssb_idx]
    sink_to_am_ssb = am_ssb_preds[am_ssb_preds['true_label'] != am_ssb_idx]
    
    results = {
        'am_ssb_correct_count': len(correct_am_ssb),
        'am_ssb_sink_count': len(sink_to_am_ssb),
        'am_ssb_correct_conf': correct_am_ssb['conf_top1'].mean() if len(correct_am_ssb) > 0 else 0,
        'am_ssb_sink_conf': sink_to_am_ssb['conf_top1'].mean() if len(sink_to_am_ssb) > 0 else 0,
        'am_ssb_correct_margin': correct_am_ssb['margin'].mean() if len(correct_am_ssb) > 0 else 0,
        'am_ssb_sink_margin': sink_to_am_ssb['margin'].mean() if len(sink_to_am_ssb) > 0 else 0,
    }
    
    # Per-SNR breakdown
    sink_by_snr = {}
    for snr in sorted(df['snr'].unique()):
        snr_data = am_ssb_preds[am_ssb_preds['snr'] == snr]
        snr_sink = snr_data[snr_data['true_label'] != am_ssb_idx]
        if len(snr_data) > 0:
            sink_by_snr[snr] = {
                'sink_count': len(snr_sink),
                'sink_pct': (len(snr_sink) / len(snr_data)) * 100,
                'sink_conf_mean': snr_sink['conf_top1'].mean() if len(snr_sink) > 0 else 0,
            }
    
    # Breakdown by source class
    source_breakdown = {}
    for idx, mod in enumerate(mod_names):
        if idx != am_ssb_idx:
            subset = sink_to_am_ssb[sink_to_am_ssb['true_label'] == idx]
            if len(subset) > 0:
                source_breakdown[mod] = {
                    'count': len(subset),
                    'conf_mean': subset['conf_top1'].mean(),
                }
    
    results['am_ssb_sink_by_snr'] = sink_by_snr
    results['am_ssb_source_breakdown'] = source_breakdown
    
    return results


def build_output_table(df, mod_names, snr_values, global_stats, wbfm_stats, am_ssb_stats):
    """Build unified long-format output table."""
    rows = []
    
    # Global stats
    rows.append({'table_type': 'global_stats', 'metric': 'overall_accuracy', 'value': global_stats['overall_accuracy']})
    rows.append({'table_type': 'global_stats', 'metric': 'conf_correct_mean', 'value': global_stats['conf_correct_mean']})
    rows.append({'table_type': 'global_stats', 'metric': 'conf_incorrect_mean', 'value': global_stats['conf_incorrect_mean']})
    rows.append({'table_type': 'global_stats', 'metric': 'margin_correct_mean', 'value': global_stats['margin_correct_mean']})
    rows.append({'table_type': 'global_stats', 'metric': 'margin_incorrect_mean', 'value': global_stats['margin_incorrect_mean']})
    
    # WBFM stats
    rows.append({'table_type': 'wbfm_stats', 'metric': 'accuracy', 'value': wbfm_stats['wbfm_accuracy']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'conf_correct', 'value': wbfm_stats['wbfm_conf_correct']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'conf_incorrect', 'value': wbfm_stats['wbfm_conf_incorrect']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'margin_correct', 'value': wbfm_stats['wbfm_margin_correct']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'margin_incorrect', 'value': wbfm_stats['wbfm_margin_incorrect']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'high_snr_acc', 'value': wbfm_stats['wbfm_high_snr_acc']})
    rows.append({'table_type': 'wbfm_stats', 'metric': 'low_snr_acc', 'value': wbfm_stats['wbfm_low_snr_acc']})
    
    # WBFM per-SNR
    for snr, acc in wbfm_stats['wbfm_snr_accuracy'].items():
        rows.append({'table_type': 'wbfm_snr_accuracy', 'snr': snr, 'value': acc})
    
    # WBFM misclassifications
    for mod, stats in wbfm_stats['wbfm_misclass'].items():
        rows.append({'table_type': 'wbfm_misclass', 'pred_class': mod, 'metric': 'count', 'value': stats['count']})
        rows.append({'table_type': 'wbfm_misclass', 'pred_class': mod, 'metric': 'pct', 'value': stats['pct']})
        rows.append({'table_type': 'wbfm_misclass', 'pred_class': mod, 'metric': 'conf', 'value': stats['conf_mean']})
        rows.append({'table_type': 'wbfm_misclass', 'pred_class': mod, 'metric': 'margin', 'value': stats['margin_mean']})
    
    # AM-SSB stats
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'correct_count', 'value': am_ssb_stats['am_ssb_correct_count']})
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'sink_count', 'value': am_ssb_stats['am_ssb_sink_count']})
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'correct_conf', 'value': am_ssb_stats['am_ssb_correct_conf']})
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'sink_conf', 'value': am_ssb_stats['am_ssb_sink_conf']})
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'correct_margin', 'value': am_ssb_stats['am_ssb_correct_margin']})
    rows.append({'table_type': 'am_ssb_stats', 'metric': 'sink_margin', 'value': am_ssb_stats['am_ssb_sink_margin']})
    
    # AM-SSB per-SNR
    for snr, data in am_ssb_stats['am_ssb_sink_by_snr'].items():
        rows.append({'table_type': 'am_ssb_snr', 'snr': snr, 'metric': 'sink_pct', 'value': data['sink_pct']})
        rows.append({'table_type': 'am_ssb_snr', 'snr': snr, 'metric': 'sink_conf', 'value': data['sink_conf_mean']})
    
    # AM-SSB source breakdown
    for source, data in am_ssb_stats['am_ssb_source_breakdown'].items():
        rows.append({'table_type': 'am_ssb_source', 'source_class': source, 'metric': 'count', 'value': data['count']})
        rows.append({'table_type': 'am_ssb_source', 'source_class': source, 'metric': 'conf', 'value': data['conf_mean']})
    
    # Class x SNR accuracy
    for class_idx, mod in enumerate(mod_names):
        for snr in sorted(df['snr'].unique()):
            subset = df[(df['true_label'] == class_idx) & (df['snr'] == snr)]
            if len(subset) > 0:
                acc = (subset['is_correct'].sum() / len(subset)) * 100
                rows.append({'table_type': 'class_snr_accuracy', 'class': mod, 'snr': snr, 'value': acc})
    
    return pd.DataFrame(rows)


def generate_summary(global_stats, wbfm_stats, am_ssb_stats):
    """Generate human-readable summary."""
    summary = []
    summary.append("="*80)
    summary.append("CONFIDENCE AND FAILURE ANALYSIS - SUMMARY")
    summary.append("="*80)
    
    summary.append("\n[GLOBAL STATISTICS]")
    summary.append(f"Overall Test Accuracy: {global_stats['overall_accuracy']:.2f}%")
    summary.append(f"Correct predictions - Avg confidence: {global_stats['conf_correct_mean']:.4f}")
    summary.append(f"Incorrect predictions - Avg confidence: {global_stats['conf_incorrect_mean']:.4f}")
    summary.append(f"Correct predictions - Avg margin (top1-top2): {global_stats['margin_correct_mean']:.4f}")
    summary.append(f"Incorrect predictions - Avg margin (top1-top2): {global_stats['margin_incorrect_mean']:.4f}")
    
    conf_gap = global_stats['conf_correct_mean'] - global_stats['conf_incorrect_mean']
    margin_gap = global_stats['margin_correct_mean'] - global_stats['margin_incorrect_mean']
    
    summary.append(f"\nConfidence gap (correct - incorrect): {conf_gap:.4f}")
    summary.append(f"  Interpretation: {'Good separation' if conf_gap > 0.1 else 'Weak separation'} between correct/incorrect")
    
    summary.append(f"\nMargin gap (correct - incorrect): {margin_gap:.4f}")
    summary.append(f"  Interpretation: {'Clear decision boundary' if margin_gap > 0.1 else 'Ambiguous decisions'}")
    
    summary.append("\n" + "="*80)
    summary.append("[WBFM FAILURE ANALYSIS]")
    summary.append("="*80)
    
    summary.append(f"\nWBFM Accuracy: {wbfm_stats['wbfm_accuracy']:.2f}%")
    summary.append(f"  High SNR (>=10 dB): {wbfm_stats['wbfm_high_snr_acc']:.2f}%")
    summary.append(f"  Low SNR (<=-10 dB): {wbfm_stats['wbfm_low_snr_acc']:.2f}%")
    
    summary.append(f"\nWBFM Correct Predictions:")
    summary.append(f"  Average confidence: {wbfm_stats['wbfm_conf_correct']:.4f}")
    summary.append(f"  Average margin: {wbfm_stats['wbfm_margin_correct']:.4f}")
    
    summary.append(f"\nWBFM Incorrect Predictions:")
    summary.append(f"  Average confidence: {wbfm_stats['wbfm_conf_incorrect']:.4f}")
    summary.append(f"  Average margin: {wbfm_stats['wbfm_margin_incorrect']:.4f}")
    
    wbfm_conf_diff = wbfm_stats['wbfm_conf_correct'] - wbfm_stats['wbfm_conf_incorrect']
    if wbfm_conf_diff > 0.1:
        summary.append(f"\n  ✓ DIAGNOSIS: UNCERTAINTY-DRIVEN ERRORS")
        summary.append(f"    Confidence gap: {wbfm_conf_diff:.4f}")
        summary.append(f"    Model is uncertain about WBFM, leading to errors.")
    elif wbfm_conf_diff < -0.05:
        summary.append(f"\n  ✗ DIAGNOSIS: REPRESENTATION FAILURE")
        summary.append(f"    Confidence gap: {wbfm_conf_diff:.4f} (NEGATIVE!)")
        summary.append(f"    Model is MORE confident in wrong predictions than correct ones.")
        summary.append(f"    This indicates feature/representation mismatch.")
    else:
        summary.append(f"\n  ? DIAGNOSIS: AMBIGUOUS")
        summary.append(f"    Confidence gap: {wbfm_conf_diff:.4f}")
    
    summary.append(f"\nWBFM Misclassification Breakdown:")
    for mod, data in sorted(wbfm_stats['wbfm_misclass'].items(), key=lambda x: x[1]['count'], reverse=True):
        summary.append(f"  {mod}: {data['count']:3d} cases ({data['pct']:5.1f}%) - Avg confidence: {data['conf_mean']:.4f}, Margin: {data['margin_mean']:.4f}")
    
    summary.append("\n" + "="*80)
    summary.append("[AM-SSB SINK ANALYSIS]")
    summary.append("="*80)
    
    summary.append(f"\nAM-SSB Predictions:")
    summary.append(f"  Correct (true label = AM-SSB): {am_ssb_stats['am_ssb_correct_count']} cases")
    summary.append(f"  Sink (true label ≠ AM-SSB): {am_ssb_stats['am_ssb_sink_count']} cases")
    
    total_am_ssb = am_ssb_stats['am_ssb_correct_count'] + am_ssb_stats['am_ssb_sink_count']
    sink_rate = (am_ssb_stats['am_ssb_sink_count'] / total_am_ssb * 100) if total_am_ssb > 0 else 0
    summary.append(f"  Sink rate: {sink_rate:.1f}%")
    
    summary.append(f"\nCorrect AM-SSB Predictions:")
    summary.append(f"  Average confidence: {am_ssb_stats['am_ssb_correct_conf']:.4f}")
    summary.append(f"  Average margin: {am_ssb_stats['am_ssb_correct_margin']:.4f}")
    
    summary.append(f"\nSink (Incorrect) AM-SSB Predictions:")
    summary.append(f"  Average confidence: {am_ssb_stats['am_ssb_sink_conf']:.4f}")
    summary.append(f"  Average margin: {am_ssb_stats['am_ssb_sink_margin']:.4f}")
    
    sink_conf_diff = am_ssb_stats['am_ssb_correct_conf'] - am_ssb_stats['am_ssb_sink_conf']
    if am_ssb_stats['am_ssb_sink_conf'] < 0.3:
        summary.append(f"\n  ✓ DIAGNOSIS: FALLBACK BEHAVIOR (Low Confidence)")
        summary.append(f"    Sink predictions have low confidence ({am_ssb_stats['am_ssb_sink_conf']:.4f})")
        summary.append(f"    Model falls back to AM-SSB when uncertain.")
    elif sink_conf_diff < 0.05:
        summary.append(f"\n  ✗ DIAGNOSIS: SYSTEMATIC BIAS (Similar High Confidence)")
        summary.append(f"    Confidence gap: {sink_conf_diff:.4f} (small)")
        summary.append(f"    Model acts as if multiple classes ARE AM-SSB.")
        summary.append(f"    This indicates classifier bias or feature collapse.")
    else:
        summary.append(f"\n  ? DIAGNOSIS: MIXED BEHAVIOR")
        summary.append(f"    Confidence gap: {sink_conf_diff:.4f}")
    
    summary.append(f"\nAM-SSB Sink Sources (Top contributors):")
    for source, data in sorted(am_ssb_stats['am_ssb_source_breakdown'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
        summary.append(f"  {source}: {data['count']:3d} cases - Avg confidence: {data['conf_mean']:.4f}")
    
    summary.append("\n" + "="*80)
    summary.append("[KEY DIAGNOSTIC QUESTIONS]")
    summary.append("="*80)
    
    summary.append(f"\nQ1: Are WBFM errors due to uncertainty or confident misclassification?")
    wbfm_margin = wbfm_stats['wbfm_margin_incorrect']
    if wbfm_margin < 0.1:
        summary.append(f"  A: UNCERTAINTY - Margin for wrong WBFM predictions: {wbfm_margin:.4f} (low)")
        summary.append(f"     Model is unsure, not confidently wrong.")
    else:
        summary.append(f"  A: CONFIDENT MISCLASSIFICATION - Margin: {wbfm_margin:.4f}")
        summary.append(f"     Model is confident but wrong → representation failure.")
    
    summary.append(f"\nQ2: Is AM-SSB acting as a fallback class or a biased attractor?")
    if am_ssb_stats['am_ssb_sink_conf'] < 0.3:
        summary.append(f"  A: FALLBACK - Low confidence sink predictions ({am_ssb_stats['am_ssb_sink_conf']:.4f})")
        summary.append(f"     Under uncertainty, model defaults to AM-SSB.")
    else:
        summary.append(f"  A: BIASED ATTRACTOR - High confidence ({am_ssb_stats['am_ssb_sink_conf']:.4f})")
        summary.append(f"     Model systematically confuses other classes as AM-SSB.")
    
    summary.append(f"\nQ3: Is the bottleneck due to representation failure or decision ambiguity?")
    avg_margin = (global_stats['margin_correct_mean'] + global_stats['margin_incorrect_mean']) / 2
    if avg_margin < 0.15:
        summary.append(f"  A: DECISION AMBIGUITY - Mean margin: {avg_margin:.4f} (low)")
        summary.append(f"     Model lacks clear confidence, suggesting input uncertainty.")
    else:
        summary.append(f"  A: LIKELY REPRESENTATION - Mean margin: {avg_margin:.4f}")
        summary.append(f"     Model makes confident decisions, but wrong on specific classes.")
    
    summary.append("\n" + "="*80)
    
    return "\n".join(summary)


def plot_heatmaps(df, mod_names, snr_values):
    """Generate heatmap visualizations."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Build class x SNR accuracy matrix
    accuracy_matrix = np.zeros((len(mod_names), len(snr_values)))
    for i, mod_idx in enumerate(range(len(mod_names))):
        for j, snr in enumerate(snr_values):
            subset = df[(df['true_label'] == mod_idx) & (df['snr'] == snr)]
            if len(subset) > 0:
                acc = (subset['is_correct'].sum() / len(subset)) * 100
                accuracy_matrix[i, j] = acc
            else:
                accuracy_matrix[i, j] = np.nan
    
    # Build AM-SSB sink matrix (% of each class predicted as AM-SSB)
    am_ssb_idx = mod_names.index('AM-SSB')
    sink_matrix = np.zeros((len(mod_names), len(snr_values)))
    for i, mod_idx in enumerate(range(len(mod_names))):
        for j, snr in enumerate(snr_values):
            subset = df[(df['true_label'] == mod_idx) & (df['snr'] == snr)]
            if len(subset) > 0:
                sink = (subset['pred_label'] == am_ssb_idx).sum() / len(subset) * 100
                sink_matrix[i, j] = sink
            else:
                sink_matrix[i, j] = np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy heatmap
    sns.heatmap(accuracy_matrix, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=100,
                xticklabels=[f'{s}dB' for s in snr_values],
                yticklabels=mod_names,
                cbar_kws={'label': 'Accuracy (%)'}, ax=axes[0])
    axes[0].set_title('Class × SNR Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('SNR Level')
    axes[0].set_ylabel('Modulation')
    
    # AM-SSB sink heatmap
    sns.heatmap(sink_matrix, annot=True, fmt='.0f', cmap='YlOrRd', vmin=0, vmax=100,
                xticklabels=[f'{s}dB' for s in snr_values],
                yticklabels=mod_names,
                cbar_kws={'label': '% Predicted as AM-SSB'}, ax=axes[1])
    axes[1].set_title('AM-SSB Sink Effect by SNR (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('SNR Level')
    axes[1].set_ylabel('True Modulation')
    
    plt.tight_layout()
    plt.savefig('analysis_heatmaps.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved heatmaps to: analysis_heatmaps.png")
    plt.close()


def main():
    print("\n" + "="*80)
    print("CONFIDENCE AND FAILURE ANALYSIS FOR RADIOML CNN")
    print("="*80)
    
    checkpoint_path = IMPORT_CONFIG['checkpoint_path']
    model_type = "STFT" if IS_STFT else "Baseline"
    print(f"\nModel Type: {model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model and data
    print("\n[1/6] Loading checkpoint...")
    checkpoint = load_checkpoint()
    print(f"[OK] Loaded checkpoint: {checkpoint_path}")
    print(f"     Best validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    print("\n[2/6] Building model...")
    model, device = build_model(checkpoint)
    print(f"[OK] Model loaded on device: {device}")
    
    print("\n[3/6] Loading test dataloader...")
    test_loader, meta = get_dataloader(checkpoint)
    mod_names = meta["mods"]
    snr_values = meta["snrs"]
    print(f"[OK] Test set loaded: {len(test_loader.dataset)} samples")
    print(f"     Modulations: {', '.join(mod_names)}")
    
    print("\n[4/6] Running inference and collecting predictions...")
    df = collect_predictions(model, test_loader, device, len(mod_names), mod_names, snr_values)
    print(f"[OK] Collected {len(df)} predictions")
    
    print("\n[5/6] Analyzing confidence and failure modes...")
    global_stats = analyze_global_confidence(df)
    wbfm_stats = analyze_wbfm(df, mod_names)
    am_ssb_stats = analyze_am_ssb_sink(df, mod_names)
    print("[OK] Analysis complete")
    
    print("\n[6/6] Generating outputs...")
    
    # Generate summary
    summary_text = generate_summary(global_stats, wbfm_stats, am_ssb_stats)
    with open('analysis_summary.txt', 'w') as f:
        f.write(summary_text)
    print("[OK] Saved summary to: analysis_summary.txt")
    
    # Generate table
    output_table = build_output_table(df, mod_names, snr_values, global_stats, wbfm_stats, am_ssb_stats)
    output_table.to_csv('analysis_tables.csv', index=False)
    print("[OK] Saved tables to: analysis_tables.csv")
    
    # Generate heatmaps
    plot_heatmaps(df, mod_names, snr_values)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - analysis_summary.txt (human-readable interpretation)")
    print("  - analysis_tables.csv (unified long-format data table)")
    print("  - analysis_heatmaps.png (visualizations)")
    print("\nKey findings written to: analysis_summary.txt")
    print("="*80 + "\n")
    
    # Print summary to console
    print(summary_text)


if __name__ == "__main__":
    main()
