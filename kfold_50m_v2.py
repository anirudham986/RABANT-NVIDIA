#!/usr/bin/env python3
# =============================================================================
# kfold_50m_v2.py — 5-Fold Cross-Validation + Ablation Study
#                    for the NTv2 50M Model (ALL 12 LAYERS UNFROZEN)
# =============================================================================
#
# Usage (on GPU server):
#   python3 kfold_50m_v2.py
#
# What this script does:
#   1. Runs 5-fold stratified cross-validation on the consolidated dataset
#      (05_consolidated_balanced.csv with pre-assigned FOLD_ID 0–4).
#   2. For each fold:
#      - Trains a fresh NTv2 50M model (all 12 layers, full fine-tuning)
#      - Evaluates on the held-out fold
#      - Performs per-layer weight magnitude ablation analysis
#      - Saves best weights .pth for the fold
#   3. Aggregates results across all 5 folds:
#      - Mean ± Std for AUROC, Accuracy, F1, MCC, Precision, Recall, Specificity
#      - 95% confidence intervals
#      - Averaged per-layer ablation statistics
#   4. Prints publication-ready summary tables
#
# NTv2-50M Architecture (official HuggingFace config.json):
#   - 12 transformer layers
#   - 16 attention heads per layer
#   - 512 hidden dimension
#   - 2048 FFN intermediate size (4× hidden)
#   - ~50M total parameters
#   - Vocabulary size: 4107 (6-mer tokenization)
#
# All hyperparameters are IDENTICAL to the 100M training:
#   epochs=20, batch_size=32, grad_accum=4, backbone_lr=5e-6, head_lr=5e-4,
#   focal_gamma=1.5, label_smoothing=0.05, patience=4, dropout=0.2
#
# CRITICAL: num_layers_to_unfreeze=12 (ALL layers, full fine-tuning)
#   The original train_50m_baseline.py incorrectly used 4, which only
#   unfroze 4 of 12 layers. This script correctly unfreezes all 12.
#
# Output:
#   output/kfold_50m_v2/
#     fold_0_best.pth               — weights for fold 0
#     fold_0_metrics.json           — metrics for fold 0
#     fold_0_history.json           — training history for fold 0
#     fold_0_ablation.json          — per-layer weight analysis for fold 0
#     ...                           — (same for folds 1–4)
#     kfold_summary.json            — aggregated results (all folds)
#     ablation_summary.json         — aggregated per-layer analysis
#
# Expected time: ~8–12 hours (5 folds × ~1.5–2.5 hours each)
# =============================================================================

import os
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,max_split_size_mb:256'
)

import sys
import json
import time
import datetime
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from copy import deepcopy
from collections import OrderedDict

warnings.filterwarnings('ignore')

# Ensure project root is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Local imports
from config import get_config
from data import load_dataset, DualSeqDataset, run_audit, LeakageAuditError
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate, print_metrics
from utils import load_hg38, set_seed, get_device, supports_amp


# =============================================================================
# CONFIGURATION — 50M model, ALL 12 layers unfrozen
# =============================================================================

CFG = {
    # Model — NTv2 50M with ALL 12 layers unfrozen
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
    'seq_length': 1000,
    'num_layers_to_unfreeze': 12,      # ALL 12 layers (full fine-tuning)
    'dropout': 0.2,

    # Training — identical to 100M training
    'epochs': 20,
    'batch_size': 32,
    'grad_accum_steps': 4,             # Effective batch = 128
    'backbone_lr': 5e-6,
    'head_lr': 5e-4,
    'weight_decay': 0.01,
    'warmup_fraction': 0.15,
    'label_smoothing': 0.05,
    'focal_gamma': 1.5,
    'max_grad_norm': 1.0,
    'patience': 4,
    'seed': 42,

    # Logging
    'log_every_n_steps': 50,
    'verbose': True,

    # Performance
    'cudnn_benchmark': True,
    'num_workers': 4,

    # Data
    'max_per_class': 50_000,

    # K-Fold
    'n_folds': 5,

    # Output
    'save_dir': os.path.join('.', 'output', 'kfold_50m_v2'),
}


# =============================================================================
# ABLATION: PER-LAYER WEIGHT MAGNITUDE ANALYSIS
# =============================================================================

def compute_layer_ablation(model, fold_id):
    """
    Analyse weight magnitudes per layer of the trained dense model.

    This pre-pruning analysis reveals:
      - Which layers have the smallest weights (most vulnerable to pruning)
      - Weight magnitude distribution across the architecture
      - Predicted pruning vulnerability at 50% global sparsity

    Returns:
        dict with per-layer statistics
    """
    print(f"\n   {'─'*60}")
    print(f"   ABLATION: Per-Layer Weight Magnitude Analysis (Fold {fold_id})")
    print(f"   {'─'*60}")

    layer_stats = OrderedDict()

    # Collect all prunable weight magnitudes globally for threshold computation
    all_magnitudes = []
    prunable_params = {}

    for name, param in model.named_parameters():
        if param.dim() >= 2 and 'embedding' not in name.lower():
            magnitudes = param.data.abs().cpu().flatten()
            all_magnitudes.append(magnitudes)
            prunable_params[name] = magnitudes

    # Global 50% threshold (what magnitude pruning at 50% would use)
    global_concat = torch.cat(all_magnitudes)
    n_total = global_concat.numel()
    n_prune = int(n_total * 0.50)
    global_threshold = torch.kthvalue(global_concat, n_prune).values.item()

    # Convert to numpy for statistical operations (torch.quantile crashes on
    # tensors with >~16M elements; numpy handles any size)
    global_np = global_concat.numpy()
    global_mean = float(np.mean(global_np))
    global_std = float(np.std(global_np))
    global_median = float(np.median(global_np))
    global_q25 = float(np.percentile(global_np, 25))
    global_q75 = float(np.percentile(global_np, 75))

    print(f"\n   Global statistics (all prunable weights):")
    print(f"     Total prunable parameters:  {n_total:,}")
    print(f"     Global mean |w|:            {global_mean:.6f}")
    print(f"     Global std  |w|:            {global_std:.6f}")
    print(f"     Global median |w|:          {global_median:.6f}")
    print(f"     Global Q25 |w|:             {global_q25:.6f}")
    print(f"     Global Q75 |w|:             {global_q75:.6f}")
    print(f"     50% pruning threshold:      {global_threshold:.6f}")

    # Per-layer analysis
    print(f"\n   {'Layer Name':<55} {'Params':>10} {'Mean|w|':>10} "
          f"{'Std|w|':>10} {'Med|w|':>10} {'<Thresh%':>10} {'Quartile':>10}")
    print(f"   {'─'*55} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    # Group by logical layer (backbone layers, head layers)
    layer_groups = OrderedDict()

    for name, magnitudes in prunable_params.items():
        n_params = magnitudes.numel()
        mean_mag = magnitudes.mean().item()
        std_mag = magnitudes.std().item()
        median_mag = magnitudes.median().item()
        below_threshold = (magnitudes < global_threshold).sum().item()
        pct_below = 100.0 * below_threshold / n_params

        # Determine which quartile this layer's mean falls in
        if mean_mag < global_q25:
            quartile = "LOW"
        elif mean_mag < global_threshold:
            quartile = "MED-LOW"
        elif mean_mag < global_q75:
            quartile = "MED-HIGH"
        else:
            quartile = "HIGH"

        stats = {
            'n_params': n_params,
            'mean_magnitude': mean_mag,
            'std_magnitude': std_mag,
            'median_magnitude': median_mag,
            'min_magnitude': magnitudes.min().item(),
            'max_magnitude': magnitudes.max().item(),
            'pct_below_50pct_threshold': pct_below,
            'n_below_threshold': below_threshold,
            'quartile': quartile,
        }
        layer_stats[name] = stats

        # Truncate name for display
        display_name = name if len(name) <= 53 else '...' + name[-50:]
        print(f"   {display_name:<55} {n_params:>10,} {mean_mag:>10.6f} "
              f"{std_mag:>10.6f} {median_mag:>10.6f} {pct_below:>9.2f}% "
              f"{quartile:>10}")

        # Group into logical groups
        if 'backbone.encoder.layer.' in name:
            parts = name.split('.')
            layer_idx = int(parts[3])
            group_key = f"backbone.encoder.layer.{layer_idx}"
        elif 'classifier' in name:
            group_key = "classifier_head"
        elif 'backbone.layer_norm' in name:
            group_key = "backbone.layer_norm"
        elif 'backbone.contact_head' in name:
            group_key = "backbone.contact_head"
        else:
            group_key = "other"

        if group_key not in layer_groups:
            layer_groups[group_key] = {
                'total_params': 0,
                'total_below_threshold': 0,
                'magnitudes': [],
            }
        layer_groups[group_key]['total_params'] += n_params
        layer_groups[group_key]['total_below_threshold'] += below_threshold
        layer_groups[group_key]['magnitudes'].append(magnitudes)

    # Grouped summary (by transformer layer index)
    print(f"\n\n   {'─'*60}")
    print(f"   GROUPED ABLATION: Per Transformer Layer Summary")
    print(f"   {'─'*60}")
    print(f"\n   {'Layer Group':<35} {'Params':>12} {'Mean|w|':>12} "
          f"{'Predicted':>12} {'Vulnerability':>14}")
    print(f"   {'':>35} {'':>12} {'':>12} {'Pruned%':>12} {'':>14}")
    print(f"   {'─'*35} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")

    grouped_stats = OrderedDict()

    for group_key in sorted(layer_groups.keys(),
                            key=lambda x: (0, int(x.split('.')[-1]))
                            if 'encoder.layer.' in x
                            else (1 if x == 'backbone.layer_norm' else
                                  2 if x == 'classifier_head' else 3, 0)):
        group = layer_groups[group_key]
        all_mags = torch.cat(group['magnitudes'])
        total_p = group['total_params']
        total_below = group['total_below_threshold']
        pct_pruned = 100.0 * total_below / total_p
        mean_mag = all_mags.mean().item()

        # Vulnerability assessment
        if pct_pruned > 60:
            vuln = " HIGH"
        elif pct_pruned > 45:
            vuln = " MODERATE"
        else:
            vuln = " LOW"

        grouped_stats[group_key] = {
            'total_params': total_p,
            'mean_magnitude': mean_mag,
            'std_magnitude': all_mags.std().item(),
            'pct_below_threshold': pct_pruned,
            'vulnerability': vuln.split(' ')[-1],
        }

        display_group = group_key if len(group_key) <= 33 else '...' + group_key[-30:]
        print(f"   {display_group:<35} {total_p:>12,} {mean_mag:>12.6f} "
              f"{pct_pruned:>11.2f}% {vuln:>14}")

    # Summary statistics
    total_prunable = sum(g['total_params'] for g in layer_groups.values())
    total_below = sum(g['total_below_threshold'] for g in layer_groups.values())
    print(f"   {'─'*35} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")
    print(f"   {'TOTAL':<35} {total_prunable:>12,} {'':>12} "
          f"{100.0 * total_below / total_prunable:>11.2f}%")

    print(f"\n   Key insight: Layers with higher 'Predicted Pruned%' have")
    print(f"   smaller weight magnitudes and will lose more connections")
    print(f"   during magnitude pruning at 50% sparsity.\n")

    return {
        'per_parameter': {k: v for k, v in layer_stats.items()},
        'per_group': grouped_stats,
        'global': {
            'total_prunable_params': n_total,
            'global_mean_magnitude': global_mean,
            'global_std_magnitude': global_std,
            'global_median_magnitude': global_median,
            'global_q25_magnitude': global_q25,
            'global_q75_magnitude': global_q75,
            'threshold_50pct': global_threshold,
        },
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_banner():
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  5-FOLD CROSS-VALIDATION — NTv2 50M MODEL (ALL 12 LAYERS)")
    print(f"  with Per-Layer Weight Magnitude Ablation Study")
    print(f"{sep}")
    print(f"  Model:      {CFG['model_name']}")
    print(f"  Layers:     12 (all unfrozen — full fine-tuning)")
    print(f"  Heads:      16 attention heads per layer")
    print(f"  Hidden:     512")
    print(f"  FFN:        2048 (4× hidden)")
    print(f"  Dataset:    consolidated (5-fold stratified, FOLD_ID 0–4)")
    print(f"  Approach:   Dual-sequence + Focal loss + Full fine-tuning")
    print(f"  Context:    {CFG['seq_length']}bp from hg38")
    print(f"  Epochs:     {CFG['epochs']} (patience={CFG['patience']})")
    print(f"  Batch:      {CFG['batch_size']} × {CFG['grad_accum_steps']} "
          f"= {CFG['batch_size'] * CFG['grad_accum_steps']} effective")
    print(f"  LR:         backbone={CFG['backbone_lr']:.0e}  head={CFG['head_lr']:.0e}")
    print(f"  Loss:       Focal(γ={CFG['focal_gamma']}) + LS={CFG['label_smoothing']}")
    print(f"  Output:     {CFG['save_dir']}")
    print(f"  Started:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}\n")


def print_fold_header(fold_id, n_folds):
    sep = "═" * 78
    print(f"\n\n{sep}")
    print(f"  ╔═══════════════════════════════════════════════════════╗")
    print(f"  ║         FOLD {fold_id + 1} / {n_folds}  "
          f"(val_fold = {fold_id})"
          f"{' ' * (30 - len(str(fold_id + 1)) - len(str(n_folds)))}║")
    print(f"  ╚═══════════════════════════════════════════════════════╝")
    print(f"{sep}")


def print_fold_results(fold_id, metrics, elapsed_str):
    sep = "━" * 60
    print(f"\n   {sep}")
    print(f"   ▶ FOLD {fold_id} — FINAL EVALUATION RESULTS")
    print(f"   {sep}")
    print(f"   │  Accuracy:     {metrics['accuracy']:8.4f}%")
    print(f"   │  AUROC:        {metrics['auroc']:8.4f}%")
    print(f"   │  F1:           {metrics['f1']:8.4f}%")
    print(f"   │  MCC:          {metrics['mcc']:8.6f}")
    print(f"   │  Precision:    {metrics['precision']:8.4f}%")
    print(f"   │  Recall:       {metrics['recall']:8.4f}%")
    print(f"   │  Specificity:  {metrics['specificity']:8.4f}%")
    print(f"   │  ──────────────────────────────────")
    print(f"   │  TP={metrics['tp']:6d}  FP={metrics['fp']:6d}  "
          f"TN={metrics['tn']:6d}  FN={metrics['fn']:6d}")
    print(f"   │  Samples:      {metrics['n_samples']:,}")
    print(f"   │  Time:         {elapsed_str}")
    print(f"   {sep}\n")


def print_kfold_summary(all_metrics, total_time_str):
    """Print the final publication-ready 5-fold summary."""
    sep = "═" * 78
    dsep = "─" * 78

    metric_keys = [
        ('auroc', 'AUROC (%)', 4),
        ('accuracy', 'Accuracy (%)', 4),
        ('f1', 'F1-Score (%)', 4),
        ('mcc', 'MCC', 6),
        ('precision', 'Precision (%)', 4),
        ('recall', 'Recall / Sensitivity (%)', 4),
        ('specificity', 'Specificity (%)', 4),
    ]

    print(f"\n\n{sep}")
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║       5-FOLD CROSS-VALIDATION RESULTS — NTv2 50M (ALL 12 LAYERS)    ║")
    print(f"  ║                   PUBLICATION-READY SUMMARY                          ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"{sep}")

    # Per-fold results table
    print(f"\n  ┌─{'─'*22}─┬─{'─'*10}─┬─{'─'*10}─┬─{'─'*10}─┬─{'─'*10}─┬─{'─'*10}─┐")
    print(f"  │ {'Metric':<22} │ {'Fold 0':>10} │ {'Fold 1':>10} │ "
          f"{'Fold 2':>10} │ {'Fold 3':>10} │ {'Fold 4':>10} │")
    print(f"  ├─{'─'*22}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┤")

    for key, label, decimals in metric_keys:
        values = [m[key] for m in all_metrics]
        fmt = f"{{:>10.{decimals}f}}"
        row = f"  │ {label:<22} │"
        for v in values:
            row += f" {fmt.format(v)} │"
        print(row)

    print(f"  └─{'─'*22}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*10}─┘")

    # Aggregated statistics
    print(f"\n  {dsep}")
    print(f"  AGGREGATED STATISTICS (Mean ± Std | 95% CI)")
    print(f"  {dsep}")
    print(f"\n  ┌─{'─'*28}─┬─{'─'*18}─┬─{'─'*8}─┬─{'─'*8}─┬─{'─'*22}─┐")
    print(f"  │ {'Metric':<28} │ {'Mean ± Std':>18} │ {'Min':>8} │ "
          f"{'Max':>8} │ {'95% CI':>22} │")
    print(f"  ├─{'─'*28}─┼─{'─'*18}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*22}─┤")

    for key, label, decimals in metric_keys:
        values = np.array([m[key] for m in all_metrics])
        mean = values.mean()
        std = values.std(ddof=1)  # Sample std
        ci_half = 1.96 * std / np.sqrt(len(values))
        ci_lo = mean - ci_half
        ci_hi = mean + ci_half
        v_min = values.min()
        v_max = values.max()

        fmt_m = f".{decimals}f"
        fmt_s = f".{decimals}f"

        mean_std_str = f"{mean:{fmt_m}} ± {std:{fmt_s}}"
        ci_str = f"[{ci_lo:{fmt_m}}, {ci_hi:{fmt_m}}]"

        print(f"  │ {label:<28} │ {mean_std_str:>18} │ "
              f"{v_min:>8{fmt_m}} │ {v_max:>8{fmt_m}} │ {ci_str:>22} │")

    print(f"  └─{'─'*28}─┴─{'─'*18}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*22}─┘")

    # Confusion matrix summary
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_tn = sum(m['tn'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)
    total_samples = sum(m['n_samples'] for m in all_metrics)

    print(f"\n  Aggregate Confusion Matrix (summed across all 5 folds):")
    print(f"  ┌─────────────────┬──────────────┬──────────────┐")
    print(f"  │                 │ Pred Benign  │ Pred Pathog  │")
    print(f"  ├─────────────────┼──────────────┼──────────────┤")
    print(f"  │ Actual Benign   │ TN={total_tn:<8,} │ FP={total_fp:<8,} │")
    print(f"  │ Actual Pathog   │ FN={total_fn:<8,} │ TP={total_tp:<8,} │")
    print(f"  └─────────────────┴──────────────┴──────────────┘")
    print(f"  Total samples evaluated: {total_samples:,}")

    # Training summary
    print(f"\n  {dsep}")
    print(f"  TRAINING CONFIGURATION")
    print(f"  {dsep}")
    print(f"  Model:          NTv2-50M (12 layers, 512 hidden, 16 heads, ~50M params)")
    print(f"  Fine-tuning:    Full (all 12 layers unfrozen)")
    print(f"  Dataset:        consolidated (5-fold stratified, FOLD_ID 0–4)")
    print(f"  Epochs:         {CFG['epochs']} per fold (early stop patience={CFG['patience']})")
    print(f"  Batch:          {CFG['batch_size']} × {CFG['grad_accum_steps']} "
          f"= {CFG['batch_size'] * CFG['grad_accum_steps']} effective")
    print(f"  LR:             backbone={CFG['backbone_lr']:.0e}  head={CFG['head_lr']:.0e}")
    print(f"  Loss:           Focal(γ={CFG['focal_gamma']}) + LS={CFG['label_smoothing']}")
    print(f"  Optimizer:      AdamW (wd={CFG['weight_decay']})")
    print(f"  Schedule:       Cosine warmup ({CFG['warmup_fraction']*100:.0f}%)")
    print(f"  Total time:     {total_time_str}")
    print(f"{sep}\n")


def print_ablation_summary(all_ablations):
    """Print averaged per-layer ablation results across all 5 folds."""
    sep = "═" * 78
    dsep = "─" * 78

    print(f"\n\n{sep}")
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║   PER-LAYER WEIGHT MAGNITUDE ABLATION — NTv2 50M (5-FOLD AVG)      ║")
    print(f"  ║               (Pre-Pruning Vulnerability Analysis)                  ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"{sep}")

    # Average the per-group stats across folds
    all_groups = set()
    for abl in all_ablations:
        all_groups.update(abl['per_group'].keys())

    # Sort groups: encoder layers first (by index), then others
    def sort_key(g):
        if 'encoder.layer.' in g:
            idx = int(g.split('.')[-1])
            return (0, idx)
        elif 'layer_norm' in g:
            return (1, 0)
        elif 'classifier' in g:
            return (2, 0)
        else:
            return (3, 0)

    sorted_groups = sorted(all_groups, key=sort_key)

    print(f"\n  ┌─{'─'*35}─┬─{'─'*12}─┬─{'─'*12}─┬─{'─'*16}─┬─{'─'*14}─┐")
    print(f"  │ {'Layer Group':<35} │ {'Params':>12} │ {'Mean |w|':>12} │ "
          f"{'Predicted':>16} │ {'Vulnerability':>14} │")
    print(f"  │ {'':>35} │ {'':>12} │ {'(mean±std)':>12} │ "
          f"{'Pruned% (±std)':>16} │ {'':>14} │")
    print(f"  ├─{'─'*35}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*16}─┼─{'─'*14}─┤")

    for group in sorted_groups:
        # Collect across folds
        means = []
        pcts = []
        params = 0

        for abl in all_ablations:
            if group in abl['per_group']:
                g = abl['per_group'][group]
                means.append(g['mean_magnitude'])
                pcts.append(g['pct_below_threshold'])
                params = g['total_params']

        if not means:
            continue

        avg_mean = np.mean(means)
        std_mean = np.std(means, ddof=1) if len(means) > 1 else 0.0
        avg_pct = np.mean(pcts)
        std_pct = np.std(pcts, ddof=1) if len(pcts) > 1 else 0.0

        # Vulnerability assessment
        if avg_pct > 60:
            vuln = " HIGH"
        elif avg_pct > 45:
            vuln = " MODERATE"
        else:
            vuln = " LOW"

        mean_str = f"{avg_mean:.6f}"
        pct_str = f"{avg_pct:.2f}±{std_pct:.2f}%"

        display_group = group if len(group) <= 33 else '...' + group[-30:]
        print(f"  │ {display_group:<35} │ {params:>12,} │ "
              f"{mean_str:>12} │ {pct_str:>16} │ {vuln:>14} │")

    print(f"  └─{'─'*35}─┴─{'─'*12}─┴─{'─'*12}─┴─{'─'*16}─┴─{'─'*14}─┘")

    # Global statistics averaged across folds
    avg_threshold = np.mean([a['global']['threshold_50pct'] for a in all_ablations])
    std_threshold = np.std([a['global']['threshold_50pct'] for a in all_ablations], ddof=1)
    avg_global_mean = np.mean([a['global']['global_mean_magnitude'] for a in all_ablations])

    print(f"\n  Global Weight Statistics (averaged across 5 folds):")
    print(f"    Total prunable parameters:  "
          f"{all_ablations[0]['global']['total_prunable_params']:,}")
    print(f"    Global mean |w|:            {avg_global_mean:.6f}")
    print(f"    50% pruning threshold:      {avg_threshold:.6f} ± {std_threshold:.6f}")

    print(f"\n  Interpretation:")
    print(f"    • Layers with HIGH vulnerability have smaller weights and will")
    print(f"      lose >60% of connections during magnitude pruning at 50% sparsity.")
    print(f"    • Layers with  LOW vulnerability encode critical representations")
    print(f"      that the model preserves (larger weights survive pruning).")
    print(f"    • Consistent vulnerability across folds indicates structural patterns")
    print(f"      rather than random variation.\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    pipeline_start = time.time()
    print_banner()

    # --- Seed & Device ---
    set_seed(CFG['seed'], benchmark=CFG.get('cudnn_benchmark', True))
    device = get_device()
    use_amp = supports_amp()

    print(f"  Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU:    {props.name}  ({vram_gb:.1f} GB VRAM)")

    # --- Reference Genome (loaded once, shared across all folds) ---
    print(f"\n{'─'*78}")
    print(f"  STEP 0: Loading Shared Resources (hg38 + Tokenizer)")
    print(f"{'─'*78}")

    genome, has_chr = load_hg38()

    tokenizer = AutoTokenizer.from_pretrained(
        CFG['model_name'], trust_remote_code=True
    )
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"  Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Create output directory ---
    save_dir = CFG['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # 5-FOLD CROSS-VALIDATION LOOP
    # =========================================================================

    all_fold_metrics = []
    all_fold_ablations = []
    fold_times = []

    for fold_id in range(CFG['n_folds']):
        fold_start = time.time()
        print_fold_header(fold_id, CFG['n_folds'])

        # --- Reset seed for reproducibility per fold ---
        fold_seed = CFG['seed'] + fold_id * 1000
        set_seed(fold_seed, benchmark=CFG.get('cudnn_benchmark', True))

        # --- Load data for this fold ---
        print(f"\n  Loading consolidated dataset (val_fold={fold_id})...")

        train_df, val_df = load_dataset(
            'consolidated',
            max_per_class=CFG['max_per_class'],
            seed=CFG['seed'],
            val_fold=fold_id,
        )

        # --- Leakage audit ---
        try:
            run_audit(
                train_df, val_df,
                dataset_name='consolidated',
                val_fold=fold_id,
            )
        except LeakageAuditError as e:
            print(f"\n  🚨 LEAKAGE AUDIT FAILED for fold {fold_id}!")
            print(str(e))
            sys.exit(1)

        # --- Build datasets ---
        print(f"\n  Building DualSeqDatasets for fold {fold_id}...")

        train_dataset = DualSeqDataset(
            train_df, genome, tokenizer, has_chr,
            seq_len=CFG['seq_length'],
            max_tokens=max_tokens,
            seed=fold_seed,
        )

        val_dataset = DualSeqDataset(
            val_df, genome, tokenizer, has_chr,
            seq_len=CFG['seq_length'],
            max_tokens=max_tokens,
            seed=fold_seed + 1,
        )

        pin_memory = (device.type == 'cuda')
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG['batch_size'],
            shuffle=True,
            num_workers=CFG['num_workers'],
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG['batch_size'],
            shuffle=False,
            num_workers=CFG['num_workers'],
            pin_memory=pin_memory,
        )

        print(f"\n  Train: {len(train_dataset):,} samples | "
              f"Val: {len(val_dataset):,} samples")
        print(f"  Batches/epoch: {len(train_loader)} train | {len(val_loader)} val")

        # --- Build fresh model for this fold ---
        print(f"\n  Building fresh NTv2 50M model for fold {fold_id}...")

        model = NTv2DualSeqClassifier(
            model_name=CFG['model_name'],
            num_layers_to_unfreeze=CFG['num_layers_to_unfreeze'],
            dropout=CFG['dropout'],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        n_layers = model.backbone.config.num_hidden_layers
        n_heads = model.backbone.config.num_attention_heads

        print(f"  Total params:  {total_params:,}")
        print(f"  Trainable:     {trainable_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")
        print(f"  Layers:        {n_layers} (unfrozen: {CFG['num_layers_to_unfreeze']})")
        print(f"  Heads:         {n_heads}")

        # Sanity check
        if CFG['num_layers_to_unfreeze'] >= n_layers:
            print(f"  ✅ FULL fine-tuning confirmed (all {n_layers} layers trainable)")
        else:
            print(f"  ⚠️  PARTIAL fine-tuning: {CFG['num_layers_to_unfreeze']}/{n_layers} layers")

        # --- Train ---
        print(f"\n  {'─'*60}")
        print(f"  TRAINING — Fold {fold_id} / {CFG['n_folds'] - 1}")
        print(f"  {'─'*60}")

        # Build config dict compatible with engine.train()
        fold_cfg = {**CFG, 'save_dir': os.path.join(save_dir, f'fold_{fold_id}')}
        os.makedirs(fold_cfg['save_dir'], exist_ok=True)

        criterion = FocalLoss(
            gamma=CFG['focal_gamma'],
            label_smoothing=CFG['label_smoothing'],
        )

        best_acc, best_auroc, history = train(
            model, train_loader, val_loader, device, criterion, fold_cfg,
            use_amp=use_amp,
            resume_from=None,
        )

        # --- Final evaluation on best model ---
        final_metrics = evaluate(
            model, val_loader, device, use_amp=use_amp,
            desc=f"Final Eval Fold {fold_id}",
        )

        fold_elapsed = time.time() - fold_start
        fold_elapsed_str = str(datetime.timedelta(seconds=int(fold_elapsed)))
        fold_times.append(fold_elapsed)

        print_fold_results(fold_id, final_metrics, fold_elapsed_str)

        # --- Ablation study ---
        ablation = compute_layer_ablation(model, fold_id)

        # --- Store results ---
        all_fold_metrics.append(final_metrics)
        all_fold_ablations.append(ablation)

        # --- Save fold outputs ---
        # Weights
        weights_path = os.path.join(save_dir, f'fold_{fold_id}_best.pth')
        torch.save(model.state_dict(), weights_path)
        print(f"  💾 Fold {fold_id} weights: {weights_path} "
              f"({os.path.getsize(weights_path)/1e6:.1f} MB)")

        # Metrics
        metrics_path = os.path.join(save_dir, f'fold_{fold_id}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)

        # History
        history_path = os.path.join(save_dir, f'fold_{fold_id}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Ablation
        ablation_path = os.path.join(save_dir, f'fold_{fold_id}_ablation.json')
        # Convert numpy types for JSON serialization
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(ablation_path, 'w') as f:
            json.dump(_make_serializable(ablation), f, indent=2)

        print(f"  📊 Fold {fold_id} results saved to {save_dir}/")

        # --- Clean up GPU memory ---
        del model, train_loader, val_loader, train_dataset, val_dataset
        del train_df, val_df
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- Progress update ---
        completed = fold_id + 1
        remaining = CFG['n_folds'] - completed
        avg_fold_time = sum(fold_times) / completed
        eta = avg_fold_time * remaining
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print(f"\n  ⏱ Folds completed: {completed}/{CFG['n_folds']} | "
              f"Avg fold time: {str(datetime.timedelta(seconds=int(avg_fold_time)))} | "
              f"ETA: {eta_str}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_time = time.time() - pipeline_start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # Print 5-fold summary
    print_kfold_summary(all_fold_metrics, total_time_str)

    # Print ablation summary
    print_ablation_summary(all_fold_ablations)

    # =========================================================================
    # SAVE MASTER SUMMARY JSON
    # =========================================================================

    metric_keys = ['auroc', 'accuracy', 'f1', 'mcc', 'precision',
                   'recall', 'specificity']

    summary = {
        'experiment': '5-Fold Cross-Validation — NTv2 50M (All 12 Layers)',
        'timestamp': datetime.datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'total_time_human': total_time_str,
        'model': {
            'name': CFG['model_name'],
            'type': 'dense (no compression) — NTv2 50M baseline v2',
            'num_hidden_layers': 12,
            'num_attention_heads': 16,
            'hidden_size': 512,
            'ffn_intermediate_size': 2048,
            'total_params': '~50M',
            'num_layers_to_unfreeze': CFG['num_layers_to_unfreeze'],
        },
        'training_config': {
            'epochs': CFG['epochs'],
            'batch_size': CFG['batch_size'],
            'effective_batch': CFG['batch_size'] * CFG['grad_accum_steps'],
            'backbone_lr': CFG['backbone_lr'],
            'head_lr': CFG['head_lr'],
            'focal_gamma': CFG['focal_gamma'],
            'label_smoothing': CFG['label_smoothing'],
            'patience': CFG['patience'],
            'weight_decay': CFG['weight_decay'],
            'warmup_fraction': CFG['warmup_fraction'],
        },
        'per_fold_metrics': {
            f'fold_{i}': m for i, m in enumerate(all_fold_metrics)
        },
        'aggregated': {},
        'fold_times_seconds': fold_times,
    }

    for key in metric_keys:
        values = np.array([m[key] for m in all_fold_metrics])
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        ci_half = 1.96 * std / np.sqrt(len(values))
        summary['aggregated'][key] = {
            'mean': mean,
            'std': std,
            'min': float(values.min()),
            'max': float(values.max()),
            'ci_95_lower': mean - ci_half,
            'ci_95_upper': mean + ci_half,
            'values': values.tolist(),
        }

    summary_path = os.path.join(save_dir, 'kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✅ Master summary: {summary_path}")

    # Save ablation summary
    ablation_summary = {
        'experiment': 'Per-Layer Weight Magnitude Ablation — NTv2 50M (All 12 Layers)',
        'description': (
            'Pre-pruning vulnerability analysis: weight magnitude distribution '
            'per layer, averaged across 5 folds. Shows which layers would lose '
            'the most weights during magnitude pruning at 50% sparsity.'
        ),
        'per_fold_global': [a['global'] for a in all_fold_ablations],
        'averaged_per_group': {},
    }

    # Average per-group stats
    all_groups = set()
    for abl in all_fold_ablations:
        all_groups.update(abl['per_group'].keys())

    for group in sorted(all_groups):
        vals_mean = [a['per_group'][group]['mean_magnitude']
                     for a in all_fold_ablations if group in a['per_group']]
        vals_pct = [a['per_group'][group]['pct_below_threshold']
                    for a in all_fold_ablations if group in a['per_group']]
        if vals_mean:
            ablation_summary['averaged_per_group'][group] = {
                'total_params': all_fold_ablations[0]['per_group'].get(
                    group, {}).get('total_params', 0),
                'mean_magnitude': float(np.mean(vals_mean)),
                'std_magnitude_across_folds': float(np.std(vals_mean, ddof=1))
                    if len(vals_mean) > 1 else 0.0,
                'pct_below_threshold_mean': float(np.mean(vals_pct)),
                'pct_below_threshold_std': float(np.std(vals_pct, ddof=1))
                    if len(vals_pct) > 1 else 0.0,
            }

    ablation_summary_path = os.path.join(save_dir, 'ablation_summary.json')
    with open(ablation_summary_path, 'w') as f:
        json.dump(ablation_summary, f, indent=2, default=str)
    print(f"  ✅ Ablation summary: {ablation_summary_path}")

    # =========================================================================
    # FINAL BANNER
    # =========================================================================
    sep = "═" * 78
    auroc_vals = np.array([m['auroc'] for m in all_fold_metrics])
    acc_vals = np.array([m['accuracy'] for m in all_fold_metrics])
    f1_vals = np.array([m['f1'] for m in all_fold_metrics])
    mcc_vals = np.array([m['mcc'] for m in all_fold_metrics])

    print(f"\n{sep}")
    print(f"  🎯 5-FOLD CROSS-VALIDATION COMPLETE — NTv2 50M (ALL 12 LAYERS)")
    print(f"{sep}")
    print(f"  AUROC:       {auroc_vals.mean():.4f} ± {auroc_vals.std(ddof=1):.4f}%")
    print(f"  Accuracy:    {acc_vals.mean():.4f} ± {acc_vals.std(ddof=1):.4f}%")
    print(f"  F1:          {f1_vals.mean():.4f} ± {f1_vals.std(ddof=1):.4f}%")
    print(f"  MCC:         {mcc_vals.mean():.6f} ± {mcc_vals.std(ddof=1):.6f}")
    print(f"  ──────────────────────────────────────")
    print(f"  Folds:       {CFG['n_folds']}")
    print(f"  Total time:  {total_time_str}")
    print(f"  Output:      {save_dir}/")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    main()
