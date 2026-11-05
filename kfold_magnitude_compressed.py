import os
os.environ.setdefault(
    #environment setup placehoder
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

# Local imports — data, model, engine (same as kfold_dense_100m.py)
from config import get_config
from data import load_dataset, DualSeqDataset, run_audit, LeakageAuditError
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate, print_metrics
from utils import load_hg38, set_seed, get_device, supports_amp

# Compression imports — reusing ALL existing LTH machinery
from compression.lth_core import (
    compute_magnitude_scores,
    percentile_rank,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
    _is_prunable,
)
from compression.shared import (
    evaluate_full,
    count_sparsity,
    save_weights_only,
    print_metrics_block,
    COMPRESSION_CFG,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dense training config — IDENTICAL to kfold_dense_100m.py / train.py
DENSE_CFG = {
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'seq_length': 1000,
    'num_layers_to_unfreeze': 22,      # ALL layers (full fine-tuning)
    'dropout': 0.2,

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

    'log_every_n_steps': 50,
    'verbose': True,
    'cudnn_benchmark': True,
    'num_workers': 4,
    'max_per_class': 50_000,
}

# LTH compression config — from COMPRESSION_CFG in compression/shared.py
LTH_CFG = {
    'lth_epochs': COMPRESSION_CFG['lth_epochs'],              # 15
    'lth_backbone_lr': COMPRESSION_CFG['lth_backbone_lr'],    # 4e-6
    'lth_head_lr': COMPRESSION_CFG['lth_head_lr'],            # 4e-4
    'lth_weight_decay': COMPRESSION_CFG['lth_weight_decay'],  # 0.01
    'lth_warmup_fraction': COMPRESSION_CFG['lth_warmup_fraction'],  # 0.12
    'lth_patience': COMPRESSION_CFG['lth_patience'],          # 5
    'lth_grad_accum': COMPRESSION_CFG['lth_grad_accum'],      # 8
    'lth_max_grad_norm': COMPRESSION_CFG['lth_max_grad_norm'],  # 1.0
    'focal_gamma': COMPRESSION_CFG['focal_gamma'],            # 2.0
    'label_smoothing': COMPRESSION_CFG['label_smoothing'],    # 0.08
    'swa_start_fraction': COMPRESSION_CFG['swa_start_fraction'],  # 0.40
}

# K-Fold & target sparsity
N_FOLDS = 5
TARGET_SPARSITY = 0.50   # 50% — the optimal operating point

# Output
SAVE_DIR = os.path.join('.', 'output', 'kfold_magnitude_compressed')


# =============================================================================
# ABLATION: PER-LAYER SPARSITY DISTRIBUTION ANALYSIS
# =============================================================================

def compute_compressed_ablation(model, masks, fold_id):
    """
    Analyse the per-layer sparsity distribution after magnitude pruning.

    Unlike the dense ablation (which predicts vulnerability), this measures
    the ACTUAL sparsity achieved per layer under global magnitude pruning.
    This reveals which layers the pruning algorithm considered most redundant.

    Returns:
        dict with per-layer and per-group sparsity statistics
    """
    print(f"\n   {'─'*60}")
    print(f"   ABLATION: Per-Layer Sparsity Distribution (Fold {fold_id})")
    print(f"   {'─'*60}")

    layer_stats = OrderedDict()
    layer_groups = OrderedDict()

    total_params = 0
    total_pruned = 0

    print(f"\n   {'Layer Name':<55} {'Params':>10} {'Pruned':>10} "
          f"{'Sparsity%':>10} {'Surviving':>10}")
    print(f"   {'─'*55} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, param in model.named_parameters():
        if name not in masks:
            continue

        mask = masks[name]
        n_params = mask.numel()
        n_pruned = int((mask == 0).sum().item())
        n_surviving = n_params - n_pruned
        sparsity_pct = 100.0 * n_pruned / n_params if n_params > 0 else 0.0

        # Per-parameter stats
        surviving_weights = param.data[mask.to(param.device) > 0]
        mean_surviving = surviving_weights.abs().mean().item() if surviving_weights.numel() > 0 else 0.0

        stats = {
            'n_params': n_params,
            'n_pruned': n_pruned,
            'n_surviving': n_surviving,
            'sparsity_pct': sparsity_pct,
            'mean_surviving_magnitude': mean_surviving,
        }
        layer_stats[name] = stats
        total_params += n_params
        total_pruned += n_pruned

        display_name = name if len(name) <= 53 else '...' + name[-50:]
        print(f"   {display_name:<55} {n_params:>10,} {n_pruned:>10,} "
              f"{sparsity_pct:>9.2f}% {n_surviving:>10,}")

        # Group by logical layer
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
                'total_pruned': 0,
            }
        layer_groups[group_key]['total_params'] += n_params
        layer_groups[group_key]['total_pruned'] += n_pruned

    # Grouped summary
    print(f"\n\n   {'─'*60}")
    print(f"   GROUPED ABLATION: Per Transformer Layer Sparsity")
    print(f"   {'─'*60}")
    print(f"\n   {'Layer Group':<35} {'Params':>12} {'Pruned':>12} "
          f"{'Sparsity%':>12} {'Impact':>14}")
    print(f"   {'─'*35} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")

    grouped_stats = OrderedDict()

    for group_key in sorted(layer_groups.keys(),
                            key=lambda x: (0, int(x.split('.')[-1]))
                            if 'encoder.layer.' in x
                            else (1 if x == 'backbone.layer_norm' else
                                  2 if x == 'classifier_head' else 3, 0)):
        group = layer_groups[group_key]
        g_total = group['total_params']
        g_pruned = group['total_pruned']
        g_sparsity = 100.0 * g_pruned / g_total if g_total > 0 else 0.0

        # Impact assessment relative to global 50%
        if g_sparsity > 60:
            impact = " HEAVY"
        elif g_sparsity > 45:
            impact = " MODERATE"
        else:
            impact = " LIGHT"

        grouped_stats[group_key] = {
            'total_params': g_total,
            'total_pruned': g_pruned,
            'sparsity_pct': g_sparsity,
            'impact': impact.strip(),
        }

        display_group = group_key if len(group_key) <= 33 else '...' + group_key[-30:]
        print(f"   {display_group:<35} {g_total:>12,} {g_pruned:>12,} "
              f"{g_sparsity:>11.2f}% {impact:>14}")

    global_sparsity = 100.0 * total_pruned / total_params if total_params > 0 else 0.0
    print(f"   {'─'*35} {'─'*12} {'─'*12} {'─'*12} {'─'*14}")
    print(f"   {'TOTAL':<35} {total_params:>12,} {total_pruned:>12,} "
          f"{global_sparsity:>11.2f}%")

    print(f"\n   Key insight: Layers with higher sparsity had smaller")
    print(f"   weight magnitudes — the pruning algorithm considered them")
    print(f"   more redundant. Layers with lower sparsity encode critical")
    print(f"   representations that the model preserves.\n")

    return {
        'per_parameter': {k: v for k, v in layer_stats.items()},
        'per_group': grouped_stats,
        'global': {
            'total_prunable_params': total_params,
            'total_pruned': total_pruned,
            'achieved_sparsity_pct': global_sparsity,
            'target_sparsity_pct': TARGET_SPARSITY * 100,
        },
    }


# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_banner():
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  5-FOLD CROSS-VALIDATION — MAGNITUDE-COMPRESSED NTv2 100M")
    print(f"  LTH @ {TARGET_SPARSITY*100:.0f}% Sparsity | Weight Rewinding | LR Rewinding")
    print(f"{sep}")
    print(f"  Model:        {DENSE_CFG['model_name']}")
    print(f"  Architecture: 22 layers, 512 hidden (~95.9M params)")
    print(f"  Sparsity:     {TARGET_SPARSITY*100:.0f}% global magnitude pruning")
    print(f"  Dataset:      consolidated (5-fold stratified, FOLD_ID 0–4)")
    print(f"  Dense phase:  {DENSE_CFG['epochs']} epochs (patience={DENSE_CFG['patience']})")
    print(f"  LTH phase:    {LTH_CFG['lth_epochs']} epochs (patience={LTH_CFG['lth_patience']})")
    print(f"  LTH LR:       backbone={LTH_CFG['lth_backbone_lr']:.0e}  "
          f"head={LTH_CFG['lth_head_lr']:.0e}")
    print(f"  Output:       {SAVE_DIR}")
    print(f"  Started:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}\n")


def print_fold_header(fold_id):
    sep = "═" * 78
    print(f"\n\n{sep}")
    print(f"  ╔═══════════════════════════════════════════════════════╗")
    print(f"  ║    FOLD {fold_id + 1} / {N_FOLDS}  "
          f"(val_fold = {fold_id})"
          f"{' ' * (30 - len(str(fold_id + 1)) - len(str(N_FOLDS)))}║")
    print(f"  ╚═══════════════════════════════════════════════════════╝")
    print(f"{sep}")


def print_comparison_table(dense_metrics, compressed_metrics, fold_id):
    """Print side-by-side comparison for a single fold."""
    sep = "━" * 68
    print(f"\n   {sep}")
    print(f"   ▶ FOLD {fold_id} — DENSE vs COMPRESSED COMPARISON")
    print(f"   {sep}")

    keys = [
        ('accuracy', 'Accuracy (%)', 4),
        ('auroc', 'AUROC (%)', 4),
        ('f1', 'F1-Score (%)', 4),
        ('mcc', 'MCC', 6),
        ('precision', 'Precision (%)', 4),
        ('recall', 'Recall (%)', 4),
        ('specificity', 'Specificity (%)', 4),
    ]

    print(f"   │ {'Metric':<22} │ {'Dense':>10} │ {'Compressed':>10} │ {'Δ':>10} │")
    print(f"   ├─{'─'*22}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}─┤")

    for key, label, dec in keys:
        d = dense_metrics[key]
        c = compressed_metrics[key]
        delta = c - d
        fmt = f".{dec}f"
        sign = "+" if delta >= 0 else ""
        print(f"   │ {label:<22} │ {d:>10{fmt}} │ {c:>10{fmt}} │ "
              f"{sign}{delta:>9{fmt}} │")

    print(f"   └─{'─'*22}─┴─{'─'*10}─┴─{'─'*10}─┴─{'─'*10}─┘")


def print_kfold_summary(all_dense, all_compressed, total_time_str):
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
    print(f"  ║  5-FOLD CV — MAGNITUDE-COMPRESSED NTv2 100M @ {TARGET_SPARSITY*100:.0f}% SPARSITY        ║")
    print(f"  ║                   PUBLICATION-READY SUMMARY                          ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"{sep}")

    # ── Dense baseline summary ──
    print(f"\n  {dsep}")
    print(f"  DENSE BASELINE (per fold)")
    print(f"  {dsep}")
    print(f"\n  ┌─{'─'*28}─┬─{'─'*18}─┬─{'─'*8}─┬─{'─'*8}─┬─{'─'*22}─┐")
    print(f"  │ {'Metric':<28} │ {'Mean ± Std':>18} │ {'Min':>8} │ "
          f"{'Max':>8} │ {'95% CI':>22} │")
    print(f"  ├─{'─'*28}─┼─{'─'*18}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*22}─┤")

    for key, label, decimals in metric_keys:
        values = np.array([m[key] for m in all_dense])
        mean = values.mean()
        std = values.std(ddof=1)
        ci_half = 1.96 * std / np.sqrt(len(values))
        fmt_m = f".{decimals}f"
        mean_std_str = f"{mean:{fmt_m}} ± {std:{fmt_m}}"
        ci_str = f"[{mean - ci_half:{fmt_m}}, {mean + ci_half:{fmt_m}}]"
        print(f"  │ {label:<28} │ {mean_std_str:>18} │ "
              f"{values.min():>8{fmt_m}} │ {values.max():>8{fmt_m}} │ {ci_str:>22} │")

    print(f"  └─{'─'*28}─┴─{'─'*18}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*22}─┘")

    # ── Compressed summary ──
    print(f"\n  {dsep}")
    print(f"  MAGNITUDE-COMPRESSED @ {TARGET_SPARSITY*100:.0f}% SPARSITY (per fold)")
    print(f"  {dsep}")
    print(f"\n  ┌─{'─'*28}─┬─{'─'*18}─┬─{'─'*8}─┬─{'─'*8}─┬─{'─'*22}─┐")
    print(f"  │ {'Metric':<28} │ {'Mean ± Std':>18} │ {'Min':>8} │ "
          f"{'Max':>8} │ {'95% CI':>22} │")
    print(f"  ├─{'─'*28}─┼─{'─'*18}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*22}─┤")

    for key, label, decimals in metric_keys:
        values = np.array([m[key] for m in all_compressed])
        mean = values.mean()
        std = values.std(ddof=1)
        ci_half = 1.96 * std / np.sqrt(len(values))
        fmt_m = f".{decimals}f"
        mean_std_str = f"{mean:{fmt_m}} ± {std:{fmt_m}}"
        ci_str = f"[{mean - ci_half:{fmt_m}}, {mean + ci_half:{fmt_m}}]"
        print(f"  │ {label:<28} │ {mean_std_str:>18} │ "
              f"{values.min():>8{fmt_m}} │ {values.max():>8{fmt_m}} │ {ci_str:>22} │")

    print(f"  └─{'─'*28}─┴─{'─'*18}─┴─{'─'*8}─┴─{'─'*8}─┴─{'─'*22}─┘")

    # ── Delta comparison ──
    print(f"\n  {dsep}")
    print(f"  Δ (COMPRESSED − DENSE) PER FOLD")
    print(f"  {dsep}")
    print(f"\n  ┌─{'─'*28}─┬─{'─'*18}─┬─{'─'*10}─┐")
    print(f"  │ {'Metric':<28} │ {'Mean Δ ± Std':>18} │ {'Verdict':>10} │")
    print(f"  ├─{'─'*28}─┼─{'─'*18}─┼─{'─'*10}─┤")

    for key, label, decimals in metric_keys:
        d_vals = np.array([m[key] for m in all_dense])
        c_vals = np.array([m[key] for m in all_compressed])
        deltas = c_vals - d_vals
        mean_d = deltas.mean()
        std_d = deltas.std(ddof=1)
        fmt_m = f".{decimals}f"
        sign = "+" if mean_d >= 0 else ""
        delta_str = f"{sign}{mean_d:{fmt_m}} ± {std_d:{fmt_m}}"
        verdict = "OK" if abs(mean_d) < 2.0 else ("DROP" if mean_d < 0 else "GAIN")
        print(f"  │ {label:<28} │ {delta_str:>18} │ {verdict:>10} │")

    print(f"  └─{'─'*28}─┴─{'─'*18}─┴─{'─'*10}─┘")

    print(f"\n  Total time: {total_time_str}")
    print(f"{sep}\n")


def print_ablation_summary(all_ablations):
    """Print averaged per-layer sparsity distribution across all 5 folds."""
    sep = "═" * 78
    print(f"\n\n{sep}")
    print(f"  ╔═══════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  PER-LAYER SPARSITY DISTRIBUTION — AVERAGED ACROSS 5 FOLDS          ║")
    print(f"  ║     (Global Magnitude Pruning @ {TARGET_SPARSITY*100:.0f}% Sparsity)"
          f"{'':>22}║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════╝")
    print(f"{sep}")

    all_groups = set()
    for abl in all_ablations:
        all_groups.update(abl['per_group'].keys())

    def sort_key(g):
        if 'encoder.layer.' in g:
            return (0, int(g.split('.')[-1]))
        elif 'layer_norm' in g:
            return (1, 0)
        elif 'classifier' in g:
            return (2, 0)
        return (3, 0)

    sorted_groups = sorted(all_groups, key=sort_key)

    print(f"\n  ┌─{'─'*35}─┬─{'─'*12}─┬─{'─'*18}─┬─{'─'*14}─┐")
    print(f"  │ {'Layer Group':<35} │ {'Params':>12} │ "
          f"{'Sparsity% (±std)':>18} │ {'Impact':>14} │")
    print(f"  ├─{'─'*35}─┼─{'─'*12}─┼─{'─'*18}─┼─{'─'*14}─┤")

    for group in sorted_groups:
        sparsities = []
        params = 0

        for abl in all_ablations:
            if group in abl['per_group']:
                g = abl['per_group'][group]
                sparsities.append(g['sparsity_pct'])
                params = g['total_params']

        if not sparsities:
            continue

        avg_sp = np.mean(sparsities)
        std_sp = np.std(sparsities, ddof=1) if len(sparsities) > 1 else 0.0

        if avg_sp > 60:
            impact = " HEAVY"
        elif avg_sp > 45:
            impact = " MODERATE"
        else:
            impact = " LIGHT"

        sp_str = f"{avg_sp:.2f}±{std_sp:.2f}%"
        display_group = group if len(group) <= 33 else '...' + group[-30:]
        print(f"  │ {display_group:<35} │ {params:>12,} │ "
              f"{sp_str:>18} │ {impact:>14} │")

    print(f"  └─{'─'*35}─┴─{'─'*12}─┴─{'─'*18}─┴─{'─'*14}─┘")

    # Global stats
    avg_global = np.mean([a['global']['achieved_sparsity_pct'] for a in all_ablations])
    std_global = np.std([a['global']['achieved_sparsity_pct'] for a in all_ablations], ddof=1)
    print(f"\n  Global sparsity: {avg_global:.2f}% ± {std_global:.2f}%")
    print(f"  Target sparsity: {TARGET_SPARSITY*100:.0f}%")

    print(f"\n  Interpretation:")
    print(f"    • Layers with HEAVY pruning (>60%) had small weight magnitudes —")
    print(f"      the model considered them redundant.")
    print(f"    • Layers with LIGHT pruning (<45%) encode critical representations")
    print(f"      that magnitude pruning preserves.")
    print(f"    • Consistent per-layer sparsity across folds indicates structural")
    print(f"      patterns in the pre-trained model, not random variation.\n")


# =============================================================================
# JSON SERIALIZATION HELPER
# =============================================================================

def _make_serializable(obj):
    """Convert numpy/torch types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


# =============================================================================
# MAIN
# =============================================================================

def main():
    pipeline_start = time.time()
    print_banner()

    # --- Seed & Device ---
    set_seed(DENSE_CFG['seed'], benchmark=DENSE_CFG.get('cudnn_benchmark', True))
    device = get_device()
    use_amp = supports_amp()

    print(f"  Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU:    {props.name}  ({vram_gb:.1f} GB VRAM)")

    # --- Reference Genome & Tokenizer (loaded once, shared across all folds) ---
    print(f"\n{'─'*78}")
    print(f"  STEP 0: Loading Shared Resources (hg38 + Tokenizer)")
    print(f"{'─'*78}")

    genome, has_chr = load_hg38()

    tokenizer = AutoTokenizer.from_pretrained(
        DENSE_CFG['model_name'], trust_remote_code=True
    )
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"  Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Create output directory ---
    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================================================================
    # 5-FOLD CROSS-VALIDATION LOOP
    # =========================================================================

    all_dense_metrics = []
    all_compressed_metrics = []
    all_ablations = []
    fold_times = []

    for fold_id in range(N_FOLDS):
        fold_start = time.time()
        print_fold_header(fold_id)

        # --- Reset seed for reproducibility per fold ---
        fold_seed = DENSE_CFG['seed'] + fold_id * 1000
        set_seed(fold_seed, benchmark=DENSE_CFG.get('cudnn_benchmark', True))

        # --- Load data for this fold ---
        print(f"\n  Loading consolidated dataset (val_fold={fold_id})...")

        train_df, val_df = load_dataset(
            'consolidated',
            max_per_class=DENSE_CFG['max_per_class'],
            seed=DENSE_CFG['seed'],
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
            print(f"\nLEAKAGE AUDIT FAILED for fold {fold_id}!")
            print(str(e))
            sys.exit(1)

        # --- Build datasets ---
        print(f"\n  Building DualSeqDatasets for fold {fold_id}...")

        train_dataset = DualSeqDataset(
            train_df, genome, tokenizer, has_chr,
            seq_len=DENSE_CFG['seq_length'],
            max_tokens=max_tokens,
            seed=fold_seed,
        )

        val_dataset = DualSeqDataset(
            val_df, genome, tokenizer, has_chr,
            seq_len=DENSE_CFG['seq_length'],
            max_tokens=max_tokens,
            seed=fold_seed + 1,
        )

        pin_memory = (device.type == 'cuda')

        # Dense training uses batch_size=32
        train_loader_dense = DataLoader(
            train_dataset,
            batch_size=DENSE_CFG['batch_size'],
            shuffle=True,
            num_workers=DENSE_CFG['num_workers'],
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader_dense = DataLoader(
            val_dataset,
            batch_size=DENSE_CFG['batch_size'],
            shuffle=False,
            num_workers=DENSE_CFG['num_workers'],
            pin_memory=pin_memory,
        )

        # LTH fine-tuning uses batch_size=8 (from COMPRESSION_CFG)
        train_loader_lth = DataLoader(
            train_dataset,
            batch_size=COMPRESSION_CFG.get('batch_size', 8),
            shuffle=True,
            num_workers=COMPRESSION_CFG.get('num_workers', 2),
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader_lth = DataLoader(
            val_dataset,
            batch_size=COMPRESSION_CFG.get('batch_size', 8),
            shuffle=False,
            num_workers=COMPRESSION_CFG.get('num_workers', 2),
            pin_memory=pin_memory,
        )

        print(f"\n  Train: {len(train_dataset):,} samples | "
              f"Val: {len(val_dataset):,} samples")

        # =====================================================================
        # PHASE A: DENSE TRAINING
        # =====================================================================
        print(f"\n  {'═'*60}")
        print(f"  PHASE A: Dense Training — Fold {fold_id}")
        print(f"  {'═'*60}")

        # Build fresh model
        model = NTv2DualSeqClassifier(
            model_name=DENSE_CFG['model_name'],
            num_layers_to_unfreeze=DENSE_CFG['num_layers_to_unfreeze'],
            dropout=DENSE_CFG['dropout'],
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"  Total params:  {total_params:,}")
        print(f"  Trainable:     {trainable_params:,} "
              f"({100 * trainable_params / total_params:.1f}%)")

        # Save pre-trained initial weights BEFORE any training (for LTH rewind)
        initial_weights = {
            name: param.data.clone().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        print(f"  Saved {len(initial_weights)} initial weight tensors for LTH rewind")

        # Train dense model
        fold_cfg = {**DENSE_CFG, 'save_dir': os.path.join(SAVE_DIR, f'fold_{fold_id}_dense')}
        os.makedirs(fold_cfg['save_dir'], exist_ok=True)

        criterion = FocalLoss(
            gamma=DENSE_CFG['focal_gamma'],
            label_smoothing=DENSE_CFG['label_smoothing'],
        )

        best_acc, best_auroc, history = train(
            model, train_loader_dense, val_loader_dense, device, criterion,
            fold_cfg, use_amp=use_amp, resume_from=None,
        )

        # Evaluate dense model on validation fold
        dense_metrics = evaluate(
            model, val_loader_dense, device, use_amp=use_amp,
            desc=f"Dense Eval Fold {fold_id}",
        )
        all_dense_metrics.append(dense_metrics)

        print(f"\n  Dense Fold {fold_id}: AUROC={dense_metrics['auroc']:.4f}%  "
              f"Acc={dense_metrics['accuracy']:.4f}%")

        # Save dense metrics
        with open(os.path.join(SAVE_DIR, f'fold_{fold_id}_dense_metrics.json'), 'w') as f:
            json.dump(dense_metrics, f, indent=2)

        # Free dense-specific data loaders
        del train_loader_dense, val_loader_dense

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # =====================================================================
        # PHASE B: LTH MAGNITUDE COMPRESSION
        # =====================================================================
        print(f"\n  {'═'*60}")
        print(f"  PHASE B: LTH Magnitude Compression @ {TARGET_SPARSITY*100:.0f}% — Fold {fold_id}")
        print(f"  {'═'*60}")

        # Step 1: Compute magnitude scores from the dense-trained model
        print(f"\n  Computing magnitude scores |w_i| from dense-trained model...")
        raw_scores = compute_magnitude_scores(model)
        ranked_scores = {name: percentile_rank(s) for name, s in raw_scores.items()}
        print(f"  Scored {len(ranked_scores)} weight tensors")

        # Step 2: Create global masks at 50% sparsity
        masks, achieved_sparsity = create_masks(ranked_scores, TARGET_SPARSITY)
        total_prunable = sum(s.numel() for s in ranked_scores.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Target sparsity: {TARGET_SPARSITY*100:.1f}%  |  "
              f"Achieved: {achieved_sparsity*100:.2f}%")
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # Step 3: LTH weight rewinding — surviving weights ← initial pre-trained values
        print(f"\n  LTH weight rewinding: surviving weights ← pre-trained init...")
        lth_rewind(model, masks, initial_weights, device)

        # Step 4: Fine-tune with frozen masks (LR rewinding happens automatically —
        # lth_finetune creates a fresh optimizer and cosine schedule from step 0)
        print(f"\n  Starting LTH fine-tuning ({LTH_CFG['lth_epochs']} epochs, "
              f"patience={LTH_CFG['lth_patience']})...")

        best_state, compressed_metrics = lth_finetune(
            model, masks, train_loader_lth, val_loader_lth, device,
            cfg=LTH_CFG,
            sparsity_label=f"mag@{TARGET_SPARSITY*100:.0f}% fold{fold_id}",
        )

        # Load best compressed weights
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)  # ensure masks enforced
        actual_sp = count_sparsity(model, masks)

        print(f"\nCompressed Fold {fold_id}: AUROC={compressed_metrics['auroc']:.4f}%  "
              f"Acc={compressed_metrics['accuracy']:.4f}%  "
              f"Actual sparsity={actual_sp*100:.2f}%")

        all_compressed_metrics.append(compressed_metrics)

        # Step 5: Comparison
        print_comparison_table(dense_metrics, compressed_metrics, fold_id)

        # Step 6: Ablation — per-layer sparsity distribution
        ablation = compute_compressed_ablation(model, masks, fold_id)
        all_ablations.append(ablation)

        # --- Save fold outputs ---
        # Compressed weights
        weights_path = os.path.join(SAVE_DIR, f'fold_{fold_id}_compressed_best.pth')
        save_weights_only(model, weights_path)

        # Compressed metrics
        with open(os.path.join(SAVE_DIR, f'fold_{fold_id}_compressed_metrics.json'), 'w') as f:
            json.dump(compressed_metrics, f, indent=2)

        # Ablation
        with open(os.path.join(SAVE_DIR, f'fold_{fold_id}_ablation.json'), 'w') as f:
            json.dump(_make_serializable(ablation), f, indent=2)

        # --- Fold timing ---
        fold_elapsed = time.time() - fold_start
        fold_elapsed_str = str(datetime.timedelta(seconds=int(fold_elapsed)))
        fold_times.append(fold_elapsed)
        print(f"\n  📊 Fold {fold_id} complete in {fold_elapsed_str}")

        # --- Clean up GPU memory ---
        del model, train_loader_lth, val_loader_lth
        del train_dataset, val_dataset, train_df, val_df
        del masks, raw_scores, ranked_scores, initial_weights, best_state
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # --- Progress update ---
        completed = fold_id + 1
        remaining = N_FOLDS - completed
        avg_fold_time = sum(fold_times) / completed
        eta = avg_fold_time * remaining
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print(f"\n  ⏱ Folds completed: {completed}/{N_FOLDS} | "
              f"Avg fold time: {str(datetime.timedelta(seconds=int(avg_fold_time)))} | "
              f"ETA: {eta_str}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_time = time.time() - pipeline_start
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print_kfold_summary(all_dense_metrics, all_compressed_metrics, total_time_str)
    print_ablation_summary(all_ablations)

    # =========================================================================
    # SAVE MASTER SUMMARY JSON
    # =========================================================================

    metric_keys = ['auroc', 'accuracy', 'f1', 'mcc', 'precision',
                   'recall', 'specificity']

    summary = {
        'experiment': f'5-Fold CV — Magnitude-Compressed NTv2 100M @ {TARGET_SPARSITY*100:.0f}%',
        'timestamp': datetime.datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'total_time_human': total_time_str,
        'model': {
            'name': DENSE_CFG['model_name'],
            'type': f'magnitude_compressed_{TARGET_SPARSITY*100:.0f}pct',
            'num_hidden_layers': 22,
            'hidden_size': 512,
            'total_params': '~95.9M',
            'target_sparsity': TARGET_SPARSITY,
        },
        'dense_training_config': {
            'epochs': DENSE_CFG['epochs'],
            'batch_size': DENSE_CFG['batch_size'],
            'effective_batch': DENSE_CFG['batch_size'] * DENSE_CFG['grad_accum_steps'],
            'backbone_lr': DENSE_CFG['backbone_lr'],
            'head_lr': DENSE_CFG['head_lr'],
            'focal_gamma': DENSE_CFG['focal_gamma'],
            'label_smoothing': DENSE_CFG['label_smoothing'],
            'patience': DENSE_CFG['patience'],
        },
        'lth_finetuning_config': {
            'epochs': LTH_CFG['lth_epochs'],
            'backbone_lr': LTH_CFG['lth_backbone_lr'],
            'head_lr': LTH_CFG['lth_head_lr'],
            'grad_accum': LTH_CFG['lth_grad_accum'],
            'focal_gamma': LTH_CFG['focal_gamma'],
            'label_smoothing': LTH_CFG['label_smoothing'],
            'patience': LTH_CFG['lth_patience'],
            'swa_start_fraction': LTH_CFG['swa_start_fraction'],
        },
        'per_fold_dense': {
            f'fold_{i}': m for i, m in enumerate(all_dense_metrics)
        },
        'per_fold_compressed': {
            f'fold_{i}': m for i, m in enumerate(all_compressed_metrics)
        },
        'aggregated_dense': {},
        'aggregated_compressed': {},
        'aggregated_delta': {},
        'fold_times_seconds': fold_times,
    }

    for key in metric_keys:
        # Dense
        d_vals = np.array([m[key] for m in all_dense_metrics])
        d_mean = float(d_vals.mean())
        d_std = float(d_vals.std(ddof=1))
        d_ci = 1.96 * d_std / np.sqrt(len(d_vals))
        summary['aggregated_dense'][key] = {
            'mean': d_mean, 'std': d_std,
            'min': float(d_vals.min()), 'max': float(d_vals.max()),
            'ci_95_lower': d_mean - d_ci, 'ci_95_upper': d_mean + d_ci,
            'values': d_vals.tolist(),
        }

        # Compressed
        c_vals = np.array([m[key] for m in all_compressed_metrics])
        c_mean = float(c_vals.mean())
        c_std = float(c_vals.std(ddof=1))
        c_ci = 1.96 * c_std / np.sqrt(len(c_vals))
        summary['aggregated_compressed'][key] = {
            'mean': c_mean, 'std': c_std,
            'min': float(c_vals.min()), 'max': float(c_vals.max()),
            'ci_95_lower': c_mean - c_ci, 'ci_95_upper': c_mean + c_ci,
            'values': c_vals.tolist(),
        }

        # Delta
        deltas = c_vals - d_vals
        summary['aggregated_delta'][key] = {
            'mean': float(deltas.mean()),
            'std': float(deltas.std(ddof=1)),
            'values': deltas.tolist(),
        }

    summary_path = os.path.join(SAVE_DIR, 'kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Master summary: {summary_path}")

    # Save ablation summary
    ablation_summary = {
        'experiment': f'Per-Layer Sparsity Distribution — Magnitude @ {TARGET_SPARSITY*100:.0f}%',
        'description': (
            'Post-pruning analysis: actual sparsity distribution per layer '
            'after global magnitude pruning, averaged across 5 folds. '
            'Shows which layers lost the most/fewest weights.'
        ),
        'per_fold_global': [_make_serializable(a['global']) for a in all_ablations],
        'averaged_per_group': {},
    }

    all_groups = set()
    for abl in all_ablations:
        all_groups.update(abl['per_group'].keys())

    for group in sorted(all_groups):
        sparsities = [a['per_group'][group]['sparsity_pct']
                      for a in all_ablations if group in a['per_group']]
        params = next((a['per_group'][group]['total_params']
                       for a in all_ablations if group in a['per_group']), 0)
        if sparsities:
            ablation_summary['averaged_per_group'][group] = {
                'total_params': params,
                'sparsity_pct_mean': float(np.mean(sparsities)),
                'sparsity_pct_std': float(np.std(sparsities, ddof=1))
                    if len(sparsities) > 1 else 0.0,
            }

    ablation_path = os.path.join(SAVE_DIR, 'ablation_summary.json')
    with open(ablation_path, 'w') as f:
        json.dump(ablation_summary, f, indent=2, default=str)
    print(f"Ablation summary: {ablation_path}")

    # =========================================================================
    # FINAL BANNER
    # =========================================================================
    sep = "═" * 78
    d_auroc = np.array([m['auroc'] for m in all_dense_metrics])
    c_auroc = np.array([m['auroc'] for m in all_compressed_metrics])
    c_acc = np.array([m['accuracy'] for m in all_compressed_metrics])
    c_f1 = np.array([m['f1'] for m in all_compressed_metrics])
    c_mcc = np.array([m['mcc'] for m in all_compressed_metrics])
    delta_auroc = c_auroc - d_auroc

    print(f"\n{sep}")
    print(f"5-FOLD CV COMPLETE — MAGNITUDE-COMPRESSED NTv2 100M @ {TARGET_SPARSITY*100:.0f}%")
    print(f"{sep}")
    print(f"  Dense AUROC:      {d_auroc.mean():.4f} ± {d_auroc.std(ddof=1):.4f}%")
    print(f"  Compressed AUROC: {c_auroc.mean():.4f} ± {c_auroc.std(ddof=1):.4f}%")
    print(f"  Δ AUROC:          {delta_auroc.mean():+.4f} ± {delta_auroc.std(ddof=1):.4f}")
    print(f"  ──────────────────────────────────────")
    print(f"  Compressed Acc:   {c_acc.mean():.4f} ± {c_acc.std(ddof=1):.4f}%")
    print(f"  Compressed F1:    {c_f1.mean():.4f} ± {c_f1.std(ddof=1):.4f}%")
    print(f"  Compressed MCC:   {c_mcc.mean():.6f} ± {c_mcc.std(ddof=1):.6f}")
    print(f"  ──────────────────────────────────────")
    print(f"  Folds:       {N_FOLDS}")
    print(f"  Total time:  {total_time_str}")
    print(f"  Output:      {SAVE_DIR}/")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    main()
