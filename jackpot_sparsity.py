DENSE_MODEL_PATH = (
    #insert data directory here
    #insert data path here
)

# ── 0. Environment setup ─────────────────────────────────────────────────────
import os
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,max_split_size_mb:256'
)

import subprocess, sys

def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"   Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

_ensure('transformers==4.40.2', 'transformers')
_ensure('pyfaidx', 'pyfaidx')
_ensure('scikit-learn', 'sklearn')
_ensure('tqdm')

# ── 1. Standard imports ──────────────────────────────────────────────────────
import json
import time
import datetime
import warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')

# Add project root to path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── 2. Local imports ─────────────────────────────────────────────────────────
from utils.seed import set_seed
from utils.device import get_device

from compression.shared import (
    find_dense_model_path,
    load_dense_model,
    build_data_loaders,
    evaluate_full,
    print_metrics_block,
    print_sparsity_table,
    save_results_json,
    save_weights_only,
    count_sparsity,
    COMPRESSION_CFG,
)

from compression.lth_core import (
    compute_magnitude_scores,
    percentile_rank,
    create_masks,
    apply_masks,
    lth_rewind,
    lth_finetune,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PIPELINE_CFG = {
    **COMPRESSION_CFG,
}

# The 4 new sparsity levels to evaluate
SPARSITIES = [0.55, 0.70, 0.80, 0.90]

# Output directory (separate from original magnitude results)
OUTPUT_DIR = os.path.join(_HERE, 'output', 'lth_compressed', 'magnitude_jackpot')


# =============================================================================
# BANNER
# =============================================================================

def print_banner():
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  JACKPOT SPARSITY — MAGNITUDE LTH SWEET SPOT IDENTIFICATION")
    print(f"{sep}")
    print(f"  Technique:  Lottery Ticket Hypothesis (LTH) — Magnitude Only")
    print(f"  Signal:     |w_i|  (weight magnitude)")
    print(f"  Sparsities: {[f'{s*100:.0f}%' for s in SPARSITIES]}")
    print(f"  Purpose:    Find critical sparsity threshold for monotonic decline")
    print(f"  Output:     {OUTPUT_DIR}/")
    print(f"{sep}")
    print(f"  Dense model: {DENSE_MODEL_PATH}")
    print(f"  Started:     {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    pipeline_start = time.time()
    print_banner()

    # ── Seed + device ────────────────────────────────────────────────────────
    set_seed(PIPELINE_CFG['seed'])
    device = get_device()
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"  GPU:    {props.name}  ({vram_gb:.1f} GB VRAM)")

    # ── Locate dense model ────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  STEP 0: Locating dense model weights")
    print(f"{'─'*78}")
    dense_path = find_dense_model_path(DENSE_MODEL_PATH)

    # ── Build shared DataLoaders ─────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  STEP 1: Building DataLoaders (train=dataset05, val=dataset06)")
    print(f"{'─'*78}")
    train_loader, val_loader, tokenizer = build_data_loaders(PIPELINE_CFG, device)

    # ── Baseline evaluation (dense model) ────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  STEP 2: Dense model baseline evaluation")
    print(f"{'─'*78}")
    dense_model = load_dense_model(dense_path, device)
    baseline_metrics = evaluate_full(
        dense_model, val_loader, device,
        desc="Dense Baseline"
    )
    del dense_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    print_metrics_block(baseline_metrics, label="DENSE BASELINE", indent="  ")

    # ── Create output directory ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # MAGNITUDE LTH AT EACH JACKPOT SPARSITY
    # =========================================================================
    print(f"\n{'═'*78}")
    print(f"  MAGNITUDE LTH — JACKPOT SPARSITY SWEEP")
    print(f"{'═'*78}")

    # Load fresh dense model for scoring
    model = load_dense_model(dense_path, device)

    # Save initial weights for LTH rewind (same as main pipeline)
    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors for LTH rewind")

    results = {}

    for idx, sp in enumerate(SPARSITIES, 1):
        sp_pct = int(sp * 100)
        print(f"\n{'─'*78}")
        print(f"  [{idx}/{len(SPARSITIES)}] MAGNITUDE @ {sp_pct}% target sparsity")
        print(f"{'─'*78}")

        t_start = time.time()

        # Step 1: Reload dense weights (fresh start for each sparsity)
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Step 2: Compute magnitude scores and convert to percentile ranks
        raw_scores = compute_magnitude_scores(model)
        ranked_scores = {name: percentile_rank(s) for name, s in raw_scores.items()}

        # Step 3: Create global masks
        masks, achieved_sp = create_masks(ranked_scores, sp)
        total_prunable = sum(s.numel() for s in ranked_scores.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")
        print(f"  Surviving weights: {total_prunable - total_pruned:,}")

        # Step 4: LTH Rewind — surviving weights get INITIAL pre-trained values
        lth_rewind(model, masks, initial_weights, device)

        # Step 5: Fine-tune with frozen masks (exact same as main pipeline)
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, PIPELINE_CFG,
            sparsity_label=f"magnitude@{sp_pct}%",
        )

        # Step 6: Load best checkpoint and verify actual sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)  # ensure masks enforced
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\nMagnitude @ {sp_pct}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"Magnitude @ {sp_pct}% sparsity",
            indent="  ",
        )

        # Step 7: Save weights-only .pth
        pth_path = os.path.join(OUTPUT_DIR, f"magnitude_sparsity{sp_pct:02d}.pth")
        save_weights_only(model, pth_path)

        # Store result
        results[sp] = {
            **best_metrics,
            'actual_sparsity': actual_sp,
            'target_sparsity': sp,
            'time_seconds': t_elapsed,
            'weights_path': pth_path,
        }

        # ── Print running comparison so far ──────────────────────────────────
        print(f"\n{'━'*78}")
        print(f"  ▶ RESULTS SO FAR ({idx}/{len(SPARSITIES)} complete)")
        print(f"{'━'*78}")
        print_sparsity_table('magnitude', results, baseline_metrics)

        # Clean up GPU
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # =========================================================================
    # SAVE RESULTS JSON
    # =========================================================================
    json_path = os.path.join(OUTPUT_DIR, 'magnitude_jackpot_results.json')
    save_results_json(
        {
            'scorer': 'magnitude',
            'purpose': 'jackpot_sparsity_sweep',
            'sparsities': SPARSITIES,
            'baseline': baseline_metrics,
            'sparsity_results': {
                f"{int(sp*100):02d}pct": v for sp, v in results.items()
            },
        },
        json_path,
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - pipeline_start
    total_str = str(datetime.timedelta(seconds=int(total_time)))

    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  JACKPOT SPARSITY — FINAL VALIDATION RESULTS")
    print(f"{sep}")

    print_sparsity_table('magnitude', results, baseline_metrics)

    # Print per-sparsity details
    for sp in SPARSITIES:
        m = results.get(sp, {})
        print_metrics_block(m, label=f"Magnitude @ {sp*100:.0f}%", indent="  ")

    # Δ vs baseline
    print(f"\n  ── Δ vs Dense Baseline (AUROC = {baseline_metrics['auroc']:.4f}%) ──")
    print(f"  {'Sparsity':>9} | {'AUROC':>10} | {'Δ AUROC':>10} | {'Acc':>8} | {'Δ Acc':>8}")
    print(f"  {'─'*9}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}")
    for sp in SPARSITIES:
        m = results.get(sp, {})
        a = m.get('auroc', 0)
        acc = m.get('accuracy', 0)
        da = a - baseline_metrics['auroc']
        dacc = acc - baseline_metrics['accuracy']
        print(f"  {sp*100:8.0f}%  | {a:>10.4f} | {da:>+10.4f} | {acc:>8.4f} | {dacc:>+8.4f}")

    print(f"\n  .pth files saved:")
    for sp in SPARSITIES:
        fn = f"magnitude_sparsity{int(sp*100):02d}.pth"
        fp = os.path.join(OUTPUT_DIR, fn)
        size = os.path.getsize(fp)/1e6 if os.path.isfile(fp) else -1
        size_str = f"{size:.1f}MB" if size > 0 else "MISSING"
        print(f"    {fp}  [{size_str}]")

    print(f"\n  Results JSON: {json_path}")
    print(f"  Total time:   {total_str}")
    print(f"\n  Next step: run test_compressed.py on these .pth files")
    print(f"  to evaluate on the 3 unseen holdout test sets (07/08/09)")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    main()
