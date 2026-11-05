# =============================================================================
# compression/shared.py
# Shared utilities: model loading, data loading, evaluation, I/O helpers.
# All compression scorers import from here.
# =============================================================================

import os
import sys
import json
import time
import glob
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix,
)
from copy import deepcopy
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.classifier import NTv2DualSeqClassifier
from data.dataset import DualSeqDataset
from data.loader import load_dataset
from data.build_holdout import build_holdout
from utils.genome import load_hg38
from utils.device import get_device
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'

# Dense model search paths (in priority order)
DENSE_MODEL_SEARCH_PATHS = [
    '/media/rvcse22/CSERV/dnai/dnai-gpu1/output/ntv2_consolidated_full_trained/ntv2_consolidated_full_final.pth',
    './output/ntv2_consolidated_full_trained/ntv2_consolidated_full_final.pth',
    './output/ntv2_consolidated_full_final.pth',
    './ntv2_consolidated_full_final.pth',
    os.path.join(_ROOT, 'output', 'ntv2_consolidated_full_trained', 'ntv2_consolidated_full_final.pth'),
]

SPARSITIES = [0.10, 0.20, 0.30, 0.50, 0.60]


# =============================================================================
# 1. MODEL LOADING
# =============================================================================

def find_dense_model_path(explicit_path: str = None) -> str:
    """
    Locate the dense model .pth file.
    Searches explicit path first, then DENSE_MODEL_SEARCH_PATHS,
    then walks common output directories.
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(DENSE_MODEL_SEARCH_PATHS)

    for p in candidates:
        if p and os.path.isfile(p):
            print(f"Dense model found: {p}")
            return p

    # Recursive search under project root
    print("   Searching for .pth files under project root...")
    for root, dirs, files in os.walk(_ROOT):
        # Skip compression output dirs to avoid picking up compressed weights
        dirs[:] = [d for d in dirs if d not in ('lth_compressed', '__pycache__', '.git')]
        for f in files:
            if 'consolidated_full_final' in f and f.endswith('.pth'):
                full = os.path.join(root, f)
                print(f"Dense model found: {full}")
                return full

    raise FileNotFoundError(
        "Cannot find ntv2_consolidated_full_final.pth.\n"
        f"Searched: {candidates}\n"
        "Set explicit_path= in compression_pipeline.py or place the file at one of the above paths."
    )


def load_dense_model(dense_path: str, device: torch.device) -> NTv2DualSeqClassifier:
    """
    Load the dense (uncompressed) NTv2DualSeqClassifier from a weights-only .pth.
    The file must contain a raw state_dict (flat {name: tensor}).
    All parameters are made trainable for LTH fine-tuning.
    """
    model = NTv2DualSeqClassifier(
        model_name=MODEL_NAME,
        num_layers_to_unfreeze=22,  # full fine-tuning
        dropout=0.2,
    ).to(device)

    state = torch.load(dense_path, map_location=device, weights_only=True)

    # Handle both raw state_dict and wrapped checkpoint
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    model.load_state_dict(state, strict=True)

    # Unfreeze everything for LTH
    for p in model.parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    print(f"Dense model loaded | {n_total:,} parameters | all trainable")
    return model


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def build_data_loaders(cfg: dict, device: torch.device) -> tuple:
    """
    Build train and validation DataLoaders from dataset 05 (train) and
    dataset 06 (holdout validation).

    Returns: (train_loader, val_loader, tokenizer)
    """
    print("\n   Loading hg38 reference genome...")
    genome, has_chr = load_hg38()

    print("\n   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer vocab={tokenizer.vocab_size} max_tokens={max_tokens}")

    print("\n   Loading dataset 05 (train) + 06 (holdout val)...")

    # Auto-generate holdout if missing
    from data.build_holdout import _find_data_dir as _hfind
    try:
        ddir = _hfind()
        holdout_path = os.path.join(ddir, '06_holdout_25k_unseen.csv')
        if not os.path.exists(holdout_path):
            print("Holdout CSV not found — auto-generating...")
            build_holdout(ddir, ddir)
    except Exception as exc:
        print(f"Holdout auto-gen check: {exc}")

    train_df, val_df = load_dataset(
        'consolidated_full',
        max_per_class=cfg.get('max_per_class', 50_000),
        seed=cfg.get('seed', 42),
    )

    print("\n   Building DualSeqDataset — train...")
    train_ds = DualSeqDataset(
        train_df, genome, tokenizer, has_chr,
        seq_len=cfg.get('seq_length', 1000),
        max_tokens=max_tokens,
        seed=cfg.get('seed', 42),
    )

    print("\n   Building DualSeqDataset — val...")
    val_ds = DualSeqDataset(
        val_df, genome, tokenizer, has_chr,
        seq_len=cfg.get('seq_length', 1000),
        max_tokens=max_tokens,
        seed=cfg.get('seed', 42) + 1,
    )

    pin = (device.type == 'cuda')
    n_workers = cfg.get('num_workers', 4)
    bs = cfg.get('batch_size', 32)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=n_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=n_workers, pin_memory=pin,
    )

    print(f"\n   Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, tokenizer


# =============================================================================
# 3. EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader,
                  device: torch.device, use_amp: bool = True,
                  desc: str = "Evaluating") -> dict:
    """
    Full evaluation returning all metrics as a dict.
    Keys: accuracy, auroc, f1, mcc, precision, recall, specificity,
          tp, fp, tn, fn, n_samples
    """
    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    pbar = tqdm(loader, desc=f"   {desc}", leave=False)
    for batch in pbar:
        ri = batch['ref_ids'].to(device)
        rm = batch['ref_mask'].to(device)
        ai = batch['alt_ids'].to(device)
        am = batch['alt_mask'].to(device)

        if use_amp and device.type == 'cuda':
            from torch.amp import autocast
            with autocast('cuda'):
                logits = model(ri, rm, ai, am)
        else:
            logits = model(ri, rm, ai, am)

        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, 1).cpu().numpy()
        labels = batch['labels'].numpy()

        probs_all.extend(probs)
        preds_all.extend(preds)
        labels_all.extend(labels)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    probs_all = np.array(probs_all)

    acc = accuracy_score(labels_all, preds_all) * 100
    auroc = roc_auc_score(labels_all, probs_all) * 100
    f1 = f1_score(labels_all, preds_all, zero_division=0) * 100
    mcc = matthews_corrcoef(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all, zero_division=0) * 100
    rec = recall_score(labels_all, preds_all, zero_division=0) * 100
    tn, fp, fn, tp = confusion_matrix(labels_all, preds_all).ravel()
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0

    return {
        'accuracy': float(acc),
        'auroc': float(auroc),
        'f1': float(f1),
        'mcc': float(mcc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'n_samples': int(len(labels_all)),
    }


# =============================================================================
# 4. SPARSITY COUNTING
# =============================================================================

def count_sparsity(model: nn.Module, masks: dict) -> float:
    """Return fraction of prunable weights that are zero."""
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        if name in masks:
            total += param.numel()
            zeros += (param.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


# =============================================================================
# 5. SAVING WEIGHTS ONLY
# =============================================================================

def save_weights_only(model: nn.Module, path: str):
    """
    Save model.state_dict() to path — weights only, no wrapper.
    This is the format expected by the testing pipeline.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved weights: {path} ({size_mb:.1f} MB)")


def save_results_json(results: dict, path: str):
    """Save a results dictionary as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {path}")


# =============================================================================
# 6. DOCUMENTATION
# =============================================================================

def print_metrics_block(metrics: dict, label: str = "", indent: str = "   "):
    """Print a full metrics block with optional label."""
    sep = "─" * 52
    print(f"\n{indent}┌{sep}┐")
    if label:
        print(f"{indent}│  {label:<50}│")
        print(f"{indent}├{sep}┤")
    print(f"{indent}│  Accuracy:    {metrics['accuracy']:8.4f}%{' '*28}│")
    print(f"{indent}│  AUROC:       {metrics['auroc']:8.4f}%{' '*28}│")
    print(f"{indent}│  F1:          {metrics['f1']:8.4f}%{' '*28}│")
    print(f"{indent}│  MCC:         {metrics['mcc']:8.6f}{' '*29}│")
    print(f"{indent}│  Precision:   {metrics['precision']:8.4f}%{' '*28}│")
    print(f"{indent}│  Recall:      {metrics['recall']:8.4f}%{' '*28}│")
    print(f"{indent}│  Specificity: {metrics['specificity']:8.4f}%{' '*28}│")
    print(f"{indent}│  TP={metrics['tp']:6d}  FP={metrics['fp']:6d}  "
          f"TN={metrics['tn']:6d}  FN={metrics['fn']:6d}  │")
    print(f"{indent}│  Samples:     {metrics['n_samples']:,}{' '*35}│")
    print(f"{indent}└{sep}┘")


def print_sparsity_table(scorer_name: str, results: dict, baseline: dict):
    """
    Print a summary table: sparsity × metrics, with Δ vs baseline.
    results: {sparsity_float: metrics_dict}
    """
    header = f"\n{'═'*80}"
    print(header)
    print(f"  SCORER: {scorer_name.upper()} — RESULTS ACROSS ALL SPARSITY LEVELS")
    print(f"{'═'*80}")
    print(f"  {'Sparsity':>9} │ {'Acc':>8} {'AUROC':>8} {'F1':>8} "
          f"{'MCC':>8} │ {'ΔAUROC':>8} {'ΔAcc':>8}")
    print(f"  {'-'*9}─┼─{'-'*8}─{'-'*8}─{'-'*8}─{'-'*8}─┼─{'-'*8}─{'-'*8}")
    b_acc = baseline.get('accuracy', 0)
    b_auroc = baseline.get('auroc', 0)
    for sp, m in sorted(results.items()):
        d_auroc = m['auroc'] - b_auroc
        d_acc = m['accuracy'] - b_acc
        flag = "Yes" if d_auroc > 0 else "No"
        print(f"  {sp*100:8.0f}%  │ {m['accuracy']:8.4f} {m['auroc']:8.4f} "
              f"{m['f1']:8.4f} {m['mcc']:8.4f} │ "
              f"{d_auroc:+8.4f} {d_acc:+8.4f}  {flag}")
    print(f"{'═'*80}\n")


def print_final_comparison(all_results: dict, baseline: dict):
    """
    Print the final cross-scorer comparison matrix.
    all_results: {scorer_name: {sparsity: metrics_dict}}
    """
    scorers = list(all_results.keys())
    print(f"\n{'═'*100}")
    print(f"  FINAL COMPARISON — AUROC @ EACH SPARSITY LEVEL")
    print(f"  Baseline dense AUROC: {baseline.get('auroc', 0):.4f}%")
    print(f"{'═'*100}")

    header = f"  {'Sparsity':>9} │"
    for s in scorers:
        header += f" {s.upper():>12}"
    print(header + " │")
    print(f"  {'-'*9}─┼─" + "─".join([f"{'-'*12}" for _ in scorers]) + "─│")

    # Collecting all sparsity levels
    sp_levels = sorted({sp for res in all_results.values() for sp in res.keys()})
    for sp in sp_levels:
        row = f"  {sp*100:8.0f}%  │"
        auroc_vals = []
        for s in scorers:
            m = all_results[s].get(sp, {})
            v = m.get('auroc', float('nan'))
            auroc_vals.append(v)
        best_v = max(v for v in auroc_vals if not np.isnan(v))
        for v in auroc_vals:
            mark = "Holds" if (not np.isnan(v) and abs(v - best_v) < 0.001) else "  "
            row += f" {v:10.4f}{mark}"
        row += " │"
        print(row)

    print(f"{'═'*100}\n")

    # Printing full metrics for hybrid at each sparsity
    if 'hybrid' in all_results:
        print(f"\n  ── HYBRID FULL METRICS ──")
        for sp, m in sorted(all_results['hybrid'].items()):
            print_metrics_block(
                m, label=f"Hybrid @ {sp*100:.0f}% sparsity", indent="  "
            )


# =============================================================================
# 7. DEFAULT COMPRESSION CONFIG
# =============================================================================

COMPRESSION_CFG = {
    # Data
    'model_name': MODEL_NAME,
    'seq_length': 1000,
    'max_per_class': 50_000,
    'batch_size': 8,            
    'num_workers': 2,            
    'seed': 42,

    # EMA warmup
    'ema_warmup_epochs': 5,        
    'ema_beta': 0.999,

    # Fisher
    'fisher_batches': 512,         

    # Movement warmup — short pass to get w_final before movement scoring
    'movement_warmup_epochs': 3,

    # LTH fine-tuning per sparsity level
    'lth_epochs': 15,              
    'lth_backbone_lr': 4e-6,       
    'lth_head_lr': 4e-4,
    'lth_weight_decay': 0.01,
    'lth_warmup_fraction': 0.12,
    'lth_patience': 5,             # AUROC-based early stopping
    'lth_grad_accum': 8,        
    'lth_max_grad_norm': 1.0,
    'focal_gamma': 2.0,
    'label_smoothing': 0.08,

    # SWA
    'swa_start_fraction': 0.40,    

    # Sparsity levels
    'sparsities': [0.10, 0.20, 0.30, 0.50, 0.60],
}
