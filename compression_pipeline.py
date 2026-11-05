DENSE_MODEL_PATH = (
    #insert model directory here
    #insert data path here
)

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
    print_final_comparison,
    save_results_json,
    COMPRESSION_CFG,
)

import compression.scorer_magnitude as scorer_magnitude
import compression.scorer_ema as scorer_ema
import compression.scorer_fisher as scorer_fisher
import compression.scorer_movement as scorer_movement
import compression.scorer_hybrid as scorer_hybrid


# =============================================================================
# CONFIGURATION — edit sparsities or fine-tuning settings here if desired
# =============================================================================

PIPELINE_CFG = {
    **COMPRESSION_CFG,
    # Override any settings here:
    # 'lth_epochs': 15,           # default is already 15
    # 'fisher_batches': 512,      # default is already 512
    # 'ema_warmup_epochs': 5,     # default is already 5
}

# Sparsity levels to evaluate (do not change unless explicitly requested)
SPARSITIES = [0.10, 0.20, 0.30, 0.50, 0.60]

# Output root
OUTPUT_ROOT = os.path.join(_HERE, 'output', 'lth_compressed')


# =============================================================================
# BANNER
# =============================================================================

def print_banner():
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"  LTH COMPRESSION PIPELINE  —  NTv2 Dual-Sequence Variant Classifier")
    print(f"{sep}")
    print(f"  Technique:  Lottery Ticket Hypothesis (LTH)  [NO quantization]")
    print(f"  Signals:    Weight Magnitude | EMA Gradient | Fisher | Movement")
    print(f"  Sparsities: {[f'{s*100:.0f}%' for s in SPARSITIES]}")
    print(f"  Strategy:   4 individual scorers → 1 dynamic hybrid scorer")
    print(f"  Priority:   AUROC — model must outperform dense baseline ≤50% sparsity")
    print(f"  Output:     {OUTPUT_ROOT}/")
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

    # ── Build shared DataLoaders (once, used by all scorers) ─────────────────
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
    print(f"\n  {'─'*40}")
    print(f"  Target to beat (≤50% sparsity):")
    print(f"    AUROC ≥ {baseline_metrics['auroc']:.4f}%")
    print(f"    Acc   ≥ {baseline_metrics['accuracy']:.4f}%")
    print(f"  {'─'*40}\n")

    # ── Save output dirs ──────────────────────────────────────────────────────
    scorer_dirs = {
        'magnitude': os.path.join(OUTPUT_ROOT, 'magnitude'),
        'ema':       os.path.join(OUTPUT_ROOT, 'ema'),
        'fisher':    os.path.join(OUTPUT_ROOT, 'fisher'),
        'movement':  os.path.join(OUTPUT_ROOT, 'movement'),
        'hybrid':    os.path.join(OUTPUT_ROOT, 'hybrid'),
    }
    for d in scorer_dirs.values():
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # STEPS 3–5: LOAD COMPLETED SCORER RESULTS FROM DISK
    # (Magnitude, EMA, Fisher already completed — loading saved JSON)
    # =========================================================================

    def _load_saved_results(scorer_name, out_dir):
        """Load a completed scorer's results JSON and convert back to
        {sparsity_float: metrics_dict} format."""
        json_path = os.path.join(out_dir, f'{scorer_name}_results.json')
        print(f"\nLoading saved {scorer_name} results from {json_path}")
        with open(json_path, 'r') as f:
            saved = json.load(f)
        results = {}
        for key, metrics in saved['sparsity_results'].items():
            # key is like "10pct", "20pct", etc.
            sp_float = int(key.replace('pct', '')) / 100.0
            results[sp_float] = metrics
        print(f"     Loaded {len(results)} sparsity levels: "
              f"{[f'{s*100:.0f}%' for s in sorted(results.keys())]}")
        return results

    print(f"\n{'═'*78}")
    print(f"  STEPS 3–5: LOADING COMPLETED RESULTS (magnitude, ema, fisher)")
    print(f"{'═'*78}")

    results_magnitude = _load_saved_results('magnitude', scorer_dirs['magnitude'])
    print_sparsity_table('magnitude', results_magnitude, baseline_metrics)

    results_ema = _load_saved_results('ema', scorer_dirs['ema'])
    print_sparsity_table('ema', results_ema, baseline_metrics)

    results_fisher = _load_saved_results('fisher', scorer_dirs['fisher'])
    print_sparsity_table('fisher', results_fisher, baseline_metrics)

    # =========================================================================
    # STEP 6: MOVEMENT SCORER
    # =========================================================================
    print(f"\n{'═'*78}")
    print(f"  STEP 6 / 7: WEIGHT MOVEMENT SCORER")
    print(f"{'═'*78}")
    t6 = time.time()

    results_movement = scorer_movement.run(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        dense_path=dense_path,
        out_dir=scorer_dirs['movement'],
        sparsities=SPARSITIES,
        cfg=PIPELINE_CFG,
        baseline_metrics=baseline_metrics,
    )

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    print(f"\nMovement scorer complete ({(time.time()-t6)/3600:.2f}h)")

    # ── Print movement results ────────────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  ▶ MOVEMENT SCORER — FINAL RESULTS (printed before next scorer starts)")
    print(f"{'━'*78}")
    print_sparsity_table('movement', results_movement, baseline_metrics)
    for sp in SPARSITIES:
        m = results_movement.get(sp, {})
        print_metrics_block(m, label=f"Movement @ {sp*100:.0f}%", indent="  ")

    # =========================================================================
    # STEP 7: HYBRID SCORER
    # Receives all 4 individual results — computes dynamic weights internally
    # =========================================================================
    print(f"\n{'═'*78}")
    print(f"  STEP 7 / 7: HYBRID DYNAMIC WEIGHTED SCORER")
    print(f"{'═'*78}")
    print(f"\n  Individual scorer AUROC summary (input to dynamic weighting):")
    print(f"  {'Sparsity':>9} | {'Magnitude':>10} {'EMA':>10} {'Fisher':>10} {'Movement':>10}")
    print(f"  {'-'*9}─┼─{'-'*10}─{'-'*10}─{'-'*10}─{'-'*10}")
    for sp in SPARSITIES:
        row = f"  {sp*100:8.0f}%  |"
        for r in [results_magnitude, results_ema, results_fisher, results_movement]:
            v = r.get(sp, {}).get('auroc', float('nan'))
            row += f" {v:10.4f}"
        print(row)
    print()

    t7 = time.time()

    individual_results = {
        'magnitude': results_magnitude,
        'ema': results_ema,
        'fisher': results_fisher,
        'movement': results_movement,
    }

    results_hybrid = scorer_hybrid.run(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        dense_path=dense_path,
        out_dir=scorer_dirs['hybrid'],
        sparsities=SPARSITIES,
        cfg=PIPELINE_CFG,
        baseline_metrics=baseline_metrics,
        individual_results=individual_results,
    )

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    print(f"\nHybrid scorer complete ({(time.time()-t7)/3600:.2f}h)")

    # ── Print hybrid results ──────────────────────────────────────────────────
    print(f"\n{'━'*78}")
    print(f"  ▶ HYBRID SCORER — FINAL RESULTS")
    print(f"{'━'*78}")
    print_sparsity_table('hybrid', results_hybrid, baseline_metrics)
    for sp in SPARSITIES:
        m = results_hybrid.get(sp, {})
        print_metrics_block(m, label=f"HYBRID @ {sp*100:.0f}%", indent="  ")

    # =========================================================================
    # FINAL CROSS-SCORER COMPARISON
    # =========================================================================
    all_results = {
        'magnitude': results_magnitude,
        'ema': results_ema,
        'fisher': results_fisher,
        'movement': results_movement,
        'hybrid': results_hybrid,
    }

    print_final_comparison(all_results, baseline_metrics)

    # ── Verify hybrid outperforms individuals ─────────────────────────────────
    print(f"\n  ── Outperformance Verification ──")
    print(f"  {'Sparsity':>9} | {'Best Individual AUROC':>22} | {'Hybrid AUROC':>14} | {'Δ':>8}")
    print(f"  {'-'*9}─┼─{'-'*22}─┼─{'-'*14}─┼─{'-'*8}")
    hybrid_wins = 0
    for sp in SPARSITIES:
        best_ind = max(
            r.get(sp, {}).get('auroc', 0.0)
            for r in [results_magnitude, results_ema, results_fisher, results_movement]
        )
        best_ind_name = max(
            individual_results.keys(),
            key=lambda k: individual_results[k].get(sp, {}).get('auroc', 0.0)
        )
        hybrid_auroc = results_hybrid.get(sp, {}).get('auroc', 0.0)
        delta = hybrid_auroc - best_ind
        mark = "YES" if delta > 0 else "NO"
        if delta > 0:
            hybrid_wins += 1
        print(f"  {sp*100:8.0f}%  | {best_ind:>18.4f}% ({best_ind_name:<10}) "
              f"| {hybrid_auroc:>14.4f}% | {delta:>+8.4f}%  {mark}")
    print(f"\n  Hybrid wins {hybrid_wins}/{len(SPARSITIES)} sparsity levels vs best individual")

    # ── Outperforms dense baseline check ─────────────────────────────────────
    b_auroc = baseline_metrics['auroc']
    print(f"\n  ── Dense Baseline Outperformance (AUROC={b_auroc:.4f}%) ──")
    print(f"  {'Sparsity':>9} | {'Hybrid AUROC':>14} | {'Δ vs Dense':>12} | Status")
    print(f"  {'-'*9}─┼─{'-'*14}─┼─{'-'*12}─┼───────")
    for sp in SPARSITIES:
        h_auroc = results_hybrid.get(sp, {}).get('auroc', 0.0)
        delta = h_auroc - b_auroc
        if sp <= 0.50:
            status = "EXCEEDS" if delta > 0 else "BELOW target"
        else:
            status = "Decline (expected)" if delta < 0 else "Rewarded bonus"
        print(f"  {sp*100:8.0f}%  | {h_auroc:>14.4f}% | {delta:>+12.4f}% | {status}")

    # =========================================================================
    # SAVE FULL PIPELINE SUMMARY JSON
    # =========================================================================
    total_time = time.time() - pipeline_start
    total_str = str(datetime.timedelta(seconds=int(total_time)))

    def _serialise(results_dict):
        out = {}
        for sp, m in results_dict.items():
            key = f"{int(sp*100):02d}pct"
            out[key] = {
                k: (float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in m.items()
                if k not in ('dynamic_weights',)
            }
            if 'dynamic_weights' in m:
                out[key]['dynamic_weights'] = {
                    k: float(dv) for k, dv in m['dynamic_weights'].items()
                }
        return out

    summary = {
        'pipeline': 'LTH Compression Pipeline',
        'timestamp': datetime.datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'total_time_human': total_str,
        'dense_model_path': dense_path,
        'sparsities': SPARSITIES,
        'baseline': baseline_metrics,
        'results': {
            'magnitude': _serialise(results_magnitude),
            'ema': _serialise(results_ema),
            'fisher': _serialise(results_fisher),
            'movement': _serialise(results_movement),
            'hybrid': _serialise(results_hybrid),
        },
        'output_dirs': scorer_dirs,
    }

    summary_path = os.path.join(OUTPUT_ROOT, 'pipeline_summary.json')
    save_results_json(summary, summary_path)

    # =========================================================================
    # FINAL BANNER
    # =========================================================================
    sep = "═" * 78
    print(f"\n{sep}")
    print(f"LTH COMPRESSION PIPELINE COMPLETE")
    print(f"{sep}")
    print(f"  Total time:  {total_str}")
    print(f"  Scorers run: magnitude | ema | fisher | movement | hybrid")
    print(f"  Sparsities:  {[f'{s*100:.0f}%' for s in SPARSITIES]}")
    print(f"  Output dir:  {OUTPUT_ROOT}/")
    print(f"")
    print(f"  Checkpoint files (weights-only, ready for testing):")
    for scorer, d in scorer_dirs.items():
        for sp in SPARSITIES:
            fn = f"{scorer}_sparsity{int(sp*100):02d}.pth"
            fp = os.path.join(d, fn)
            size = os.path.getsize(fp)/1e6 if os.path.isfile(fp) else -1
            size_str = f"{size:.1f}MB" if size > 0 else "MISSING"
            print(f"    {fp}  [{size_str}]")
    print(f"")
    print(f"  Summary JSON: {summary_path}")
    print(f"{sep}\n")


# =============================================================================
if __name__ == '__main__':
    main()
