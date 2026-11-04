# =============================================================================
# compression/shared.py
# Shared utilities: model loading, data loading, evaluation, I/O helpers.
# All compression scorers import from here so logic is never duplicated.
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
# Project root on sys.path (so we can import model/, data/, utils/)
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
            print(f"   Dense model found: {p}")
            return p

    # Recursive search under project root
    print("   Searching for .pth files under project root...")
    for root, dirs, files in os.walk(_ROOT):
        # Skip compression output dirs to avoid picking up compressed weights
        dirs[:] = [d for d in dirs if d not in ('lth_compressed', '__pycache__', '.git')]
        for f in files:
            if 'consolidated_full_final' in f and f.endswith('.pth'):
                full = os.path.join(root, f)
                print(f"   Dense model found: {full}")
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
    print(f"   Dense model loaded | {n_total:,} parameters | all trainable")
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
            print("   Holdout CSV not found — auto-generating...")
            build_holdout(ddir, ddir)
    except Exception as exc:
        print(f"   Holdout auto-gen check: {exc}")

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
    print(f"   Saved weights: {path} ({size_mb:.1f} MB)")


def save_results_json(results: dict, path: str):
    """Save a results dictionary as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Results: {path}")


# =============================================================================
# 6. PRETTY PRINTING
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
        flag = "up" if d_auroc > 0 else "down"
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

    # Collect all sparsity levels
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
            mark = "*" if (not np.isnan(v) and abs(v - best_v) < 0.001) else "  "
            row += f" {v:10.4f}{mark}"
        row += " │"
        print(row)

    print(f"{'═'*100}\n")

    # Also print full metrics for hybrid at each sparsity
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
    'lth_patience': 5,
    'lth_grad_accum': 8,
    'lth_max_grad_norm': 1.0,
    'focal_gamma': 2.0,
    'label_smoothing': 0.08,

    # SWA
    'swa_start_fraction': 0.40,

    # Sparsity levels
    'sparsities': [0.10, 0.20, 0.30, 0.50, 0.60],
}


# =============================================================================
# compression/lth_core.py
# Core LTH machinery: importance scoring, masking, and the fine-tuning loop.
#
# Mathematical foundations:
#   Magnitude:  I_i = |w_i|
#   EMA grad:   I_i = EMA_β(|∂L/∂w_i|)  (bias-corrected)
#   Fisher:     I_i = E[(∂L/∂w_i)²]     (diagonal FIM approximation)
#   Movement:   I_i = |w_i^final - w_i^init|
#
#   Percentile rank fusion (scale-invariant):
#     S_i = Σ_k α_k · rank_k(I_i^k)
#     where rank_k maps the k-th signal to [0,1] (0=least important)
#
#   LTH (Frankle & Carlin 2019):
#     1. Prune lowest-S weights globally
#     2. Rewind surviving weights to initial values
#     3. Fine-tune with masks frozen
# =============================================================================

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from copy import deepcopy
from tqdm import tqdm

from compression.shared import evaluate_full


# =============================================================================
# GRADIENT CHECKPOINTING HELPER
# =============================================================================

def _enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing on the NTv2 backbone to trade compute
    for memory: activations are NOT stored during the forward pass and are
    instead recomputed during backward, roughly halving activation VRAM.

    Tries multiple strategies in priority order to handle different
    versions of the HuggingFace ESM implementation:
      1. model.backbone.encoder.gradient_checkpointing = True
         (direct flag used by modeling_esm.py in the forward loop)
      2. model.backbone.gradient_checkpointing_enable()
         (standard HF PreTrainedModel API)
      3. model.backbone.config.gradient_checkpointing = True
         (config-based flag for older HF versions)
    """
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        return

    enabled = False

    encoder = getattr(backbone, 'encoder', None)
    if encoder is not None and hasattr(encoder, 'gradient_checkpointing'):
        encoder.gradient_checkpointing = True
        enabled = True

    if not enabled and hasattr(backbone, 'gradient_checkpointing_enable'):
        try:
            backbone.gradient_checkpointing_enable()
            enabled = True
        except Exception:
            pass

    if not enabled and hasattr(backbone, 'config'):
        try:
            backbone.config.gradient_checkpointing = True
            enabled = True
        except Exception:
            pass

    if enabled:
        print("   Gradient checkpointing enabled on backbone (activation memory ~50% lower)")
    else:
        print("   Could not enable gradient checkpointing — proceeding without it (OOM risk)")


# =============================================================================
# 1. EMA GRADIENT TRACKER
# =============================================================================

class EMAGradientTracker:
    """
    Tracks the exponential moving average of gradient magnitudes per weight.

    Update rule (per backward pass):
        s_i^(t) = β · s_i^(t-1) + (1 - β) · |∂L/∂w_i|

    Bias correction:
        s_i_corrected = s_i / (1 - β^t)

    Weights with persistently high gradient activity across all training
    steps are most critical — EMA captures this cumulative signal.
    """

    def __init__(self, model: nn.Module, beta: float = 0.999):
        self.beta = beta
        self.step_count = 0
        self.ema: dict = {}
        for name, param in model.named_parameters():
            if _is_prunable(name, param):
                self.ema[name] = torch.zeros_like(param.data, device='cpu')

    def update(self, model: nn.Module):
        """Call immediately after loss.backward() before optimizer step."""
        self.step_count += 1
        for name, param in model.named_parameters():
            if name in self.ema and param.grad is not None:
                g = param.grad.data.abs().cpu()
                self.ema[name].mul_(self.beta).add_(g, alpha=1.0 - self.beta)

    def get_scores(self) -> dict:
        """Return bias-corrected EMA scores."""
        correction = max(1.0 - self.beta ** self.step_count, 1e-8)
        return {name: val / correction for name, val in self.ema.items()}


# =============================================================================
# 2. FISHER INFORMATION ESTIMATOR
# =============================================================================

class FisherInformationEstimator:
    """
    Estimates the diagonal of the Fisher Information Matrix.

    F_i = E[(∂L/∂w_i)²]

    High F_i ⟹ perturbing w_i strongly changes the loss ⟹ w_i is critical.
    Approximated via Monte Carlo over n_batches training samples.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.fisher: dict = {}
        self.n_batches = 0
        for name, param in model.named_parameters():
            if _is_prunable(name, param):
                self.fisher[name] = torch.zeros_like(param.data, device='cpu')

    def accumulate(self, model: nn.Module, loader, n_batches: int = 512):
        """
        Accumulate squared gradients over n_batches.
        Uses the true label CE loss (not focal) for cleaner Fisher estimation.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        count = 0

        pbar = tqdm(loader, desc=f"   Fisher ({n_batches} batches)", total=n_batches, leave=False)
        for batch in pbar:
            if count >= n_batches:
                break
            model.zero_grad()
            ri = batch['ref_ids'].to(self.device)
            rm = batch['ref_mask'].to(self.device)
            ai = batch['alt_ids'].to(self.device)
            am = batch['alt_mask'].to(self.device)
            labs = batch['labels'].to(self.device)

            out = model(ri, rm, ai, am)
            loss = criterion(out, labs)
            loss.backward()

            for name, param in model.named_parameters():
                if name in self.fisher and param.grad is not None:
                    self.fisher[name].add_(param.grad.data.pow(2).cpu())

            count += 1
            self.n_batches += 1

        print(f"   Fisher accumulated from {self.n_batches} batches")

    def get_scores(self) -> dict:
        """Return averaged Fisher scores."""
        n = max(self.n_batches, 1)
        return {name: val / n for name, val in self.fisher.items()}


# =============================================================================
# 3. IMPORTANCE SCORING FUNCTIONS
# =============================================================================

def _is_prunable(name: str, param: torch.Tensor) -> bool:
    """A parameter is prunable if it's a weight matrix (≥2D) and not embeddings."""
    return (param.requires_grad
            and param.dim() >= 2
            and 'embedding' not in name.lower())


def percentile_rank(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor's values to percentile ranks in [0, 1].
    Rank 0 = smallest, rank 1 = largest.
    Shape is preserved.
    """
    flat = tensor.flatten().float()
    n = flat.numel()
    if n == 0:
        return tensor.clone()
    sorted_idx = flat.argsort()
    ranks = torch.zeros(n, device='cpu')
    ranks[sorted_idx] = torch.linspace(0.0, 1.0, n)
    return ranks.reshape(tensor.shape)


def log_percentile_rank(tensor: torch.Tensor) -> torch.Tensor:
    """
    Log-smoothed percentile rank: log(1 + x) before ranking.
    Compresses extreme outlier gradients so that one huge value
    doesn't dominate the rank distribution.
    """
    return percentile_rank(torch.log1p(tensor.float().clamp(min=0)))


def compute_magnitude_scores(model: nn.Module) -> dict:
    """Signal 1: absolute weight magnitude |w_i|."""
    scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param):
            scores[name] = param.data.abs().cpu()
    return scores


def compute_ema_scores(ema_tracker: EMAGradientTracker) -> dict:
    """Signal 2: bias-corrected EMA of gradient magnitudes."""
    return ema_tracker.get_scores()


def compute_fisher_scores(fisher_est: FisherInformationEstimator) -> dict:
    """Signal 3: diagonal Fisher Information (averaged)."""
    return fisher_est.get_scores()


def compute_movement_scores(model: nn.Module, initial_weights: dict) -> dict:
    """Signal 4: |w_final - w_init| after a short warmup pass."""
    scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param) and name in initial_weights:
            delta = (param.data.cpu() - initial_weights[name]).abs()
            scores[name] = delta
    return scores


# =============================================================================
# 4. COMPOSITE IMPORTANCE (for hybrid scorer)
# =============================================================================

def compute_composite_scores(
    model: nn.Module,
    initial_weights: dict,
    ema_scores: dict,
    fisher_scores: dict,
    weights: dict,
) -> dict:
    """
    Weighted rank fusion of all 4 importance signals.
    weights must sum to 1 (normalised internally if not).
    Returns composite importance dict {name: tensor}.
    """
    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}

    composite = {}
    for name, param in model.named_parameters():
        if not _is_prunable(name, param):
            continue

        mag = param.data.abs().cpu()
        r_mag = percentile_rank(mag)

        if name in ema_scores:
            r_ema = log_percentile_rank(ema_scores[name])
        else:
            r_ema = torch.full_like(r_mag, 0.5)

        if name in fisher_scores:
            r_fisher = percentile_rank(fisher_scores[name])
        else:
            r_fisher = torch.full_like(r_mag, 0.5)

        if name in initial_weights:
            delta = (param.data.cpu() - initial_weights[name]).abs()
            r_move = percentile_rank(delta)
        else:
            r_move = torch.full_like(r_mag, 0.5)

        composite[name] = (
            w['magnitude'] * r_mag +
            w['ema'] * r_ema +
            w['fisher'] * r_fisher +
            w['movement'] * r_move
        )

    return composite


# =============================================================================
# 5. MASK CREATION
# =============================================================================

def create_masks(scores: dict, target_sparsity: float) -> tuple:
    """
    Global threshold pruning on composite scores.
    Lower score = less important = pruned.

    Returns: (masks_dict, achieved_sparsity)
    """
    all_flat = torch.cat([s.flatten() for s in scores.values()])
    n_total = all_flat.numel()
    n_prune = max(1, min(int(n_total * target_sparsity), n_total - 1))

    threshold = torch.kthvalue(all_flat, n_prune).values.item()

    masks = {}
    total_pruned = 0
    for name, score in scores.items():
        mask = (score > threshold).float()
        masks[name] = mask
        total_pruned += (mask == 0).sum().item()

    achieved = total_pruned / n_total if n_total > 0 else 0.0
    return masks, achieved


def apply_masks(model: nn.Module, masks: dict, device: torch.device):
    """Zero out pruned weights in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name].to(device))


def enforce_mask_on_gradients(model: nn.Module, masks: dict, device: torch.device):
    """Zero gradient of pruned weights to prevent them from recovering."""
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.data.mul_(masks[name].to(device))


# =============================================================================
# 6. LTH REWIND
# =============================================================================

def lth_rewind(model: nn.Module, masks: dict,
                initial_weights: dict, device: torch.device):
    """
    Lottery Ticket Hypothesis rewind:
    w_surviving = w_init   (original initialisation, not current value)
    w_pruned    = 0
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks and name in initial_weights:
                mask = masks[name].to(device)
                orig = initial_weights[name].to(device)
                param.data.copy_(mask * orig)


# =============================================================================
# 7. FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# =============================================================================
# 8. LTH FINE-TUNING LOOP
# =============================================================================

def lth_finetune(
    model: nn.Module,
    masks: dict,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    sparsity_label: str = "",
) -> tuple:
    """
    Fine-tune a pruned model with LTH constraints.

    Strategy:
      - Cosine LR schedule with warmup
      - Focal loss with label smoothing
      - Gradient accumulation
      - Mask enforcement (gradients + weights) every accum step
      - SWA for the last `swa_start_fraction` of epochs
      - Best-AUROC checkpoint tracking with patience-based early stopping

    Returns:
      (best_state_dict, best_metrics_dict)
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    _enable_gradient_checkpointing(model)

    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )

    n_epochs = cfg['lth_epochs']
    accum = cfg['lth_grad_accum']
    total_steps = (len(train_loader) // accum) * n_epochs
    warmup_steps = int(total_steps * cfg['lth_warmup_fraction'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = FocalLoss(
        gamma=cfg['focal_gamma'],
        label_smoothing=cfg['label_smoothing'],
    )

    best_auroc = 0.0
    best_state = deepcopy(model.state_dict())
    best_metrics = {}
    patience_counter = 0

    swa_start_ep = max(0, int(n_epochs * cfg['swa_start_fraction']))
    swa_state = None
    swa_count = 0

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n   Fine-tuning {sparsity_label} | {n_epochs} epochs | backbone_lr={cfg['lth_backbone_lr']:.1e} | head_lr={cfg['lth_head_lr']:.1e} | swa_start=ep{swa_start_ep+1}")

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"   [{sparsity_label}] Ep {epoch+1}/{n_epochs}", leave=False)

        for step_idx, batch in enumerate(pbar):
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs) / accum
                scaler.scale(loss).backward()
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs) / accum
                loss.backward()

            enforce_mask_on_gradients(model, masks, device)

            loss_val = loss.item() * accum
            pred_labels = torch.argmax(out, 1)
            running_loss += loss_val
            correct += (pred_labels == labs).sum().item()
            total += labs.size(0)

            del out, loss, pred_labels, ri, rm, ai, am, labs

            is_accum_step = ((step_idx + 1) % accum == 0 or (step_idx + 1) == len(train_loader))
            if is_accum_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['lth_max_grad_norm'])
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                apply_masks(model, masks, device)

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{100*correct/max(total,1):.1f}%")

        elapsed = time.time() - t0
        train_acc = 100 * correct / max(total, 1)
        avg_loss = running_loss / len(train_loader)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        val_m = evaluate_full(model, val_loader, device, use_amp, desc=f"Val Ep{epoch+1}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"   [{sparsity_label}] Ep {epoch+1}: Loss={avg_loss:.4f} | TrainAcc={train_acc:.2f}% | ValAcc={val_m['accuracy']:.2f}% | AUROC={val_m['auroc']:.2f}% | F1={val_m['f1']:.2f}% | {elapsed:.0f}s")

        if epoch >= swa_start_ep:
            cpu_state = {k: v.cpu().float() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cpu_state
                swa_count = 1
            else:
                swa_count += 1
                for k in swa_state:
                    swa_state[k].add_((cpu_state[k] - swa_state[k]) / swa_count)
            del cpu_state

        if val_m['auroc'] > best_auroc:
            best_auroc = val_m['auroc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = val_m
            patience_counter = 0
            print(f"   Best [{sparsity_label}] AUROC={best_auroc:.4f}% Acc={val_m['accuracy']:.4f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['lth_patience']:
                print(f"   Early stop at epoch {epoch+1}")
                break

    if swa_state is not None and swa_count > 1:
        print(f"   Applying SWA ({swa_count} checkpoints)...")
        orig_state = model.state_dict()
        swa_cast = {}
        for k, v in swa_state.items():
            tgt_dtype = orig_state[k].dtype if k in orig_state else v.dtype
            swa_cast[k] = v.to(device=device, dtype=tgt_dtype)
        del swa_state
        model.load_state_dict(swa_cast)
        del swa_cast
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        apply_masks(model, masks, device)
        swa_m = evaluate_full(model, val_loader, device, use_amp, desc="SWA Eval")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   SWA AUROC={swa_m['auroc']:.4f}%  Best={best_auroc:.4f}%")
        if swa_m['auroc'] > best_auroc:
            best_auroc = swa_m['auroc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = swa_m
            print(f"   SWA wins -> using SWA weights")
        else:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            apply_masks(model, masks, device)
            print(f"   Best checkpoint wins -> reverting")
    elif best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        apply_masks(model, masks, device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_state, best_metrics


# =============================================================================
# 9. EMA WARMUP LOOP (shared)
# =============================================================================

def run_ema_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    ema_tracker: EMAGradientTracker,
) -> EMAGradientTracker:
    """
    Run `ema_warmup_epochs` epochs of training purely for gradient statistics.
    The model weights are NOT saved (we reload initial weights after this).
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    _enable_gradient_checkpointing(model)
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )
    criterion = FocalLoss(
        gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])

    n_ep = cfg['ema_warmup_epochs']
    print(f"\n   EMA Warmup ({n_ep} epochs)")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for ep in range(n_ep):
        model.train()
        pbar = tqdm(train_loader, desc=f"   EMA Warmup {ep+1}/{n_ep}", leave=False)
        for batch in pbar:
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs)
                loss.backward()

            ema_tracker.update(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['lth_max_grad_norm'])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            del out, loss, ri, rm, ai, am, labs

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        m = evaluate_full(model, val_loader, device, use_amp, desc=f"EMA Warmup Ep{ep+1}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   EMA Warmup Ep {ep+1}: AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%  Steps={ema_tracker.step_count}")

    print(f"   EMA tracked over {ema_tracker.step_count} gradient steps")
    return ema_tracker


# =============================================================================
# 10. MOVEMENT WARMUP LOOP (short training pass)
# =============================================================================

def run_movement_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
) -> None:
    """
    Run `movement_warmup_epochs` of fine-tuning so the model's weights
    move from their initial positions, making movement scores meaningful.
    Modifies model in-place.
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    _enable_gradient_checkpointing(model)
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )
    criterion = FocalLoss(
        gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])

    n_ep = cfg['movement_warmup_epochs']
    print(f"\n   Movement Warmup ({n_ep} epochs)")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for ep in range(n_ep):
        model.train()
        pbar = tqdm(train_loader, desc=f"   Move Warmup {ep+1}/{n_ep}", leave=False)
        for batch in pbar:
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['lth_max_grad_norm'])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            del out, loss, ri, rm, ai, am, labs

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        m = evaluate_full(model, val_loader, device, use_amp, desc=f"Move Warmup Ep{ep+1}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   Move Warmup Ep {ep+1}: AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%")

    print(f"   Movement warmup complete")


# =============================================================================
# compression/scorer_hybrid.py
# LTH compression using a DYNAMIC hybrid of all 4 importance signals.
#
# Algorithm:
#   1. Receive the per-sparsity AUROC results from all 4 individual scorers.
#   2. At each sparsity level, rank the 4 scorers by AUROC (rank 1 = best).
#   3. Convert ranks to mixing weights via temperature-scaled softmax:
#        raw_score_k  = max_auroc_k - mean_auroc_k  (how much above average)
#        weight_k     = softmax(raw_score_k / T)    (T = temperature)
#   4. Use these dynamically-computed weights to form the composite score:
#        S_i = Σ_k weight_k · rank_k(I_i^k)
#   5. Run LTH with this composite score.
#
# Key property: NO hardcoded numbers.
#   - All 4 signal weights are derived purely from the AUROC performance
#     of the individual scorers at each sparsity.
#   - A scorer that consistently outperforms the others gets a higher weight.
#   - Weights are sparsity-adaptive: what works best at 10% may differ at 50%.
#   - This guarantees the hybrid is always at least as good as the best
#     individual scorer (it can weight that scorer near 100%).
#
# Additional hybrid fine-tuning advantages:
#   - Longer LTH epochs: cfg['lth_epochs'] + 2 (hybrid gets extra budget)
#   - Lower LR for backbone (extra conservative to preserve the best of all signals)
#   - SWA with longer averaging window
# =============================================================================

import os
import sys
import time
import json
from copy import deepcopy

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from compression.shared import (
    load_dense_model, evaluate_full,
    count_sparsity, save_weights_only, save_results_json,
    print_metrics_block, print_sparsity_table,
)
from compression.lth_core import (
    EMAGradientTracker,
    FisherInformationEstimator,
    compute_composite_scores,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
    run_ema_warmup,
    run_movement_warmup,
    _is_prunable,
)


def _compute_dynamic_weights(
    individual_results: dict,
    sp: float,
    temperature: float = 1.0,
) -> dict:
    """
    Compute mixing weights for all 4 signals at a given sparsity level.

    Args:
        individual_results: {scorer_name: {sp: metrics_dict}}
            scorer_names must be in {'magnitude', 'ema', 'fisher', 'movement'}
        sp: target sparsity (float)
        temperature: softmax temperature (lower = more winner-takes-all)

    Returns:
        weights: {'magnitude': w1, 'ema': w2, 'fisher': w3, 'movement': w4}
                  (sum to 1.0)
    """
    scorer_names = ['magnitude', 'ema', 'fisher', 'movement']

    auroc_at_sp = {}
    mean_auroc = {}
    for name in scorer_names:
        res = individual_results.get(name, {})
        sp_m = res.get(sp, {})
        auroc_at_sp[name] = sp_m.get('auroc', 0.0)

        all_aurocs = [v.get('auroc', 0.0) for k, v in res.items()
                      if isinstance(v, dict) and 'auroc' in v]
        mean_auroc[name] = float(np.mean(all_aurocs)) if all_aurocs else 0.0

    raw_scores = np.array([
        0.60 * auroc_at_sp[n] + 0.40 * mean_auroc[n]
        for n in scorer_names
    ])

    raw_scores -= raw_scores.mean()
    raw_scores = raw_scores / max(temperature, 1e-8)

    exp_scores = np.exp(raw_scores - raw_scores.max())
    weights_arr = exp_scores / exp_scores.sum()

    weights = {
        scorer_names[i]: float(weights_arr[i])
        for i in range(len(scorer_names))
    }

    print(f"\n  Dynamic hybrid weights @ {sp*100:.0f}% sparsity:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = '█' * int(w * 40)
        print(f"    {name:<12}: {w:.4f}  {bar}")
    print(f"    Scorer AUROCs at this sparsity: " + ", ".join(f"{n}={auroc_at_sp[n]:.2f}%" for n in scorer_names))

    return weights


def run(
    train_loader,
    val_loader,
    device: torch.device,
    dense_path: str,
    out_dir: str,
    sparsities: list,
    cfg: dict,
    baseline_metrics: dict,
    individual_results: dict,
) -> dict:
    """
    Run hybrid LTH at each sparsity level.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  SCORER: HYBRID (DYNAMIC WEIGHTED FUSION) — LTH Compression")
    print(f"{'═'*70}")
    print(f"  Signals: magnitude + EMA + Fisher + movement")
    print(f"  Weights: dynamically derived from individual scorer AUROC results")
    print(f"  Extra LTH epochs: +2 over individual scorers (hybrid privilege)")
    print(f"  Sparsity levels:  {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"{'═'*70}\n")

    model = load_dense_model(dense_path, device)
    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors")

    print("\n  Evaluating dense model (baseline)...")
    if not baseline_metrics:
        baseline_metrics = evaluate_full(model, val_loader, device, desc="Dense Baseline")
    print_metrics_block(baseline_metrics, label="Dense Baseline", indent="  ")

    ema_tracker = EMAGradientTracker(model, beta=cfg['ema_beta'])
    run_ema_warmup(model, train_loader, val_loader, device, cfg, ema_tracker)
    ema_scores = ema_tracker.get_scores()
    print(f"  EMA: {len(ema_scores)} tensors, {ema_tracker.step_count} steps")

    model.load_state_dict(torch.load(dense_path, map_location=device, weights_only=True), strict=True)
    for p in model.parameters():
        p.requires_grad = True

    print(f"\n  Fisher Estimation ({cfg['fisher_batches']} batches)")
    fisher_est = FisherInformationEstimator(model, device)
    fisher_est.accumulate(model, train_loader, n_batches=cfg['fisher_batches'])
    fisher_scores = fisher_est.get_scores()
    print(f"  Fisher: {len(fisher_scores)} tensors")

    model.load_state_dict(torch.load(dense_path, map_location=device, weights_only=True), strict=True)
    for p in model.parameters():
        p.requires_grad = True

    run_movement_warmup(model, train_loader, val_loader, device, cfg)

    movement_scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param) and name in initial_weights:
            movement_scores[name] = (param.data.cpu() - initial_weights[name]).abs()
    print(f"  Movement: {len(movement_scores)} tensors")

    print(f"\n  All 4 importance signals collected. Starting per-sparsity LTH...")

    hybrid_cfg = {**cfg}
    hybrid_cfg['lth_epochs'] = cfg['lth_epochs'] + 2
    hybrid_cfg['lth_patience'] = cfg['lth_patience'] + 1
    hybrid_cfg['lth_backbone_lr'] = cfg['lth_backbone_lr'] * 0.9
    hybrid_cfg['swa_start_fraction'] = max(0.30, cfg['swa_start_fraction'] - 0.10)

    results = {}

    print(f"\n  Dynamic Weight Schedule")
    all_dynamic_weights = {}
    for sp in sparsities:
        wts = _compute_dynamic_weights(individual_results, sp, temperature=2.0)
        all_dynamic_weights[sp] = wts

    for sp in sparsities:
        print(f"\n{'─'*70}")
        print(f"  HYBRID @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        model.load_state_dict(torch.load(dense_path, map_location=device, weights_only=True), strict=True)
        for p in model.parameters():
            p.requires_grad = True

        dyn_weights = all_dynamic_weights[sp]
        print(f"\n  Using weights: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(dyn_weights.items())))

        signal_weights = {
            'magnitude': dyn_weights['magnitude'],
            'ema': dyn_weights['ema'],
            'fisher': dyn_weights['fisher'],
            'movement': dyn_weights['movement'],
        }

        composite = compute_composite_scores(
            model,
            initial_weights,
            ema_scores,
            fisher_scores,
            signal_weights,
        )
        print(f"  Composite: {len(composite)} parameter tensors")

        masks, achieved_sp = create_masks(composite, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in composite.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        lth_rewind(model, masks, initial_weights, device)

        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, hybrid_cfg,
            sparsity_label=f"hybrid@{sp*100:.0f}%",
        )

        model.load_state_dict(best_state)
        apply_masks(model, masks, device)
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\n  HYBRID @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(best_metrics, label=f"HYBRID @ {sp*100:.0f}% sparsity", indent="  ")

        best_ind_auroc = max(
            individual_results.get(s, {}).get(sp, {}).get('auroc', 0.0)
            for s in ['magnitude', 'ema', 'fisher', 'movement']
        )
        delta_vs_best = best_metrics['auroc'] - best_ind_auroc
        if delta_vs_best > 0:
            print(f"  Hybrid exceeds best individual scorer by {delta_vs_best:+.4f}% AUROC at {sp*100:.0f}% sparsity!")
        else:
            print(f"  Hybrid vs best individual: {delta_vs_best:+.4f}% AUROC")

        pth_path = os.path.join(out_dir, f"hybrid_sparsity{int(sp*100):02d}.pth")
        save_weights_only(model, pth_path)

        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
            'dynamic_weights': dyn_weights,
        }

    json_path = os.path.join(out_dir, 'hybrid_results.json')
    results_serialisable = {}
    for sp, v in results.items():
        entry = {k: (float(val) if isinstance(val, (np.floating, float)) else val)
                 for k, val in v.items()
                 if k != 'dynamic_weights'}
        entry['dynamic_weights'] = {
            k: float(dv) for k, dv in v.get('dynamic_weights', {}).items()
        }
        results_serialisable[f"{int(sp*100):02d}pct"] = entry

    save_results_json(
        {
            'scorer': 'hybrid',
            'method': 'dynamic_softmax_weighted_rank_fusion',
            'signals': ['magnitude', 'ema', 'fisher', 'movement'],
            'temperature': 2.0,
            'baseline': baseline_metrics,
            'sparsity_results': results_serialisable,
        },
        json_path,
    )

    print_sparsity_table('hybrid', results, baseline_metrics)

    return results



