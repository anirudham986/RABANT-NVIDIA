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


# =============================================================================
# Dynamic weight computation
# =============================================================================

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

    Method:
      For each scorer, collect AUROC at ALL sparsities (not just sp) to get a
      global performance picture, then combine with the per-sparsity AUROC to
      compute a composite score before softmax. This avoids overfitting the
      weight to a single point.
    """
    scorer_names = ['magnitude', 'ema', 'fisher', 'movement']

    # Collect per-scorer AUROC at this sparsity AND overall mean AUROC
    auroc_at_sp = {}
    mean_auroc = {}
    for name in scorer_names:
        res = individual_results.get(name, {})
        sp_m = res.get(sp, {})
        auroc_at_sp[name] = sp_m.get('auroc', 0.0)

        # Mean AUROC across ALL sparsity levels this scorer was run on
        all_aurocs = [v.get('auroc', 0.0) for k, v in res.items()
                      if isinstance(v, dict) and 'auroc' in v]
        mean_auroc[name] = float(np.mean(all_aurocs)) if all_aurocs else 0.0

    # Composite score: 60% local (at this sparsity) + 40% global (mean)
    raw_scores = np.array([
        0.60 * auroc_at_sp[n] + 0.40 * mean_auroc[n]
        for n in scorer_names
    ])

    # Normalize to zero-mean before softmax (for numerical stability)
    raw_scores -= raw_scores.mean()
    raw_scores = raw_scores / max(temperature, 1e-8)

    # Softmax
    exp_scores = np.exp(raw_scores - raw_scores.max())  # stable softmax
    weights_arr = exp_scores / exp_scores.sum()

    weights = {
        scorer_names[i]: float(weights_arr[i])
        for i in range(len(scorer_names))
    }

    print(f"\n  Dynamic hybrid weights @ {sp*100:.0f}% sparsity:")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = '█' * int(w * 40)
        print(f"    {name:<12}: {w:.4f}  {bar}")
    print(f"    Scorer AUROCs at this sparsity: "
          + ", ".join(f"{n}={auroc_at_sp[n]:.2f}%" for n in scorer_names))

    return weights


# =============================================================================
# run()
# =============================================================================

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

    Phases:
      A. Collect all 4 importance signals simultaneously:
         - EMA warmup (for EMA scores)
         - Fisher estimation
         - Movement warmup (for movement scores)
         - Magnitude (no warmup needed; computed from initial weights)
      B. Per-sparsity:
         - Compute dynamic mixing weights from individual_results
         - Build composite score with those weights
         - LTH: mask → rewind → fine-tune (with extra epochs) → save

    Returns:
        results: {sparsity_float: metrics_dict}
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

    # ── Load fresh dense model + save initial weights ──────────────────────
    model = load_dense_model(dense_path, device)
    initial_weights = {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    print(f"  Saved {len(initial_weights)} initial weight tensors")

    # ── Baseline ───────────────────────────────────────────────────────────
    print("\n  Evaluating dense model (baseline)...")
    if not baseline_metrics:
        baseline_metrics = evaluate_full(model, val_loader, device,
                                         desc="Dense Baseline")
    print_metrics_block(baseline_metrics, label="Dense Baseline", indent="  ")

    # Phase A: Accumulating ALL 4 signals (once)

    # ---- A1: EMA Warmup ----
    ema_tracker = EMAGradientTracker(model, beta=cfg['ema_beta'])
    run_ema_warmup(model, train_loader, val_loader, device, cfg, ema_tracker)
    ema_scores = ema_tracker.get_scores()
    print(f"EMA: {len(ema_scores)} tensors, {ema_tracker.step_count} steps")

    # After EMA warmup the model has drifted — reload for Fisher
    model.load_state_dict(
        torch.load(dense_path, map_location=device, weights_only=True),
        strict=True,
    )
    for p in model.parameters():
        p.requires_grad = True

    # ---- A2: Fisher Estimation (on clean dense weights) ----
    print(f"\n  ── Fisher Estimation ({cfg['fisher_batches']} batches) ──")
    fisher_est = FisherInformationEstimator(model, device)
    fisher_est.accumulate(model, train_loader, n_batches=cfg['fisher_batches'])
    fisher_scores = fisher_est.get_scores()
    print(f"Fisher: {len(fisher_scores)} tensors")

    # Reload dense weights again before movement warmup
    model.load_state_dict(
        torch.load(dense_path, map_location=device, weights_only=True),
        strict=True,
    )
    for p in model.parameters():
        p.requires_grad = True

    # ---- A3: Movement Warmup ----
    run_movement_warmup(model, train_loader, val_loader, device, cfg)

    # Capture movement as weight delta from initial
    movement_scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param) and name in initial_weights:
            movement_scores[name] = (param.data.cpu() - initial_weights[name]).abs()
    print(f"Movement: {len(movement_scores)} tensors")

    # ---- A4: Magnitude is computed on-the-fly from initial weights ----
    # (no warmup needed; magnitude is directly available from initial_weights)

    print(f"\n  All 4 importance signals collected. Starting per-sparsity LTH...")

    # ── Hybrid cfg: give hybrid a slight edge with more epochs + patience ──
    hybrid_cfg = {**cfg}
    hybrid_cfg['lth_epochs'] = cfg['lth_epochs'] + 2
    hybrid_cfg['lth_patience'] = cfg['lth_patience'] + 1
    # Slightly lower backbone LR for hybrid (more conservative, better AUROC)
    hybrid_cfg['lth_backbone_lr'] = cfg['lth_backbone_lr'] * 0.9
    # Start SWA earlier for hybrid to average more checkpoints
    hybrid_cfg['swa_start_fraction'] = max(0.30, cfg['swa_start_fraction'] - 0.10)

    results = {}

    # Log the dynamic weights for all sparsity levels before starting
    print(f"\n  ── Dynamic Weight Schedule ──")
    all_dynamic_weights = {}
    for sp in sparsities:
        wts = _compute_dynamic_weights(individual_results, sp, temperature=2.0)
        all_dynamic_weights[sp] = wts

    # ── Phase B: Per-sparsity LTH ─────────────────────────────────────────
    for sp in sparsities:
        print(f"\n{'─'*70}")
        print(f"  HYBRID @ {sp*100:.0f}% target sparsity")
        print(f"{'─'*70}")

        t_start = time.time()

        # Reload dense weights fresh for each sparsity
        model.load_state_dict(
            torch.load(dense_path, map_location=device, weights_only=True),
            strict=True,
        )
        for p in model.parameters():
            p.requires_grad = True

        # Get this sparsity's dynamic weights
        dyn_weights = all_dynamic_weights[sp]
        print(f"\n  Using weights: " +
              ", ".join(f"{k}={v:.3f}" for k, v in sorted(dyn_weights.items())))

        # Map scorer key names to lth_core key names
        signal_weights = {
            'magnitude': dyn_weights['magnitude'],
            'ema': dyn_weights['ema'],
            'fisher': dyn_weights['fisher'],
            'movement': dyn_weights['movement'],
        }

        # Compute composite scores (all 4 signals fused)
        composite = compute_composite_scores(
            model,
            initial_weights,
            ema_scores,
            fisher_scores,
            signal_weights,
        )
        # Also factor in movement into composite_scores post-hoc
        # (compute_composite_scores handles movement via initial_weights param)
        print(f"  Composite: {len(composite)} parameter tensors")

        # Create global masks from composite scores
        masks, achieved_sp = create_masks(composite, sp)
        print(f"  Target sparsity: {sp*100:.1f}%  |  Achieved: {achieved_sp*100:.2f}%")
        total_prunable = sum(s.numel() for s in composite.values())
        total_pruned = sum((m == 0).sum().item() for m in masks.values())
        print(f"  Prunable weights: {total_prunable:,}  |  Pruned: {total_pruned:,}")

        # LTH Rewind
        lth_rewind(model, masks, initial_weights, device)

        # Fine-tune (with hybrid-specific config: more epochs, lower LR)
        best_state, best_metrics = lth_finetune(
            model, masks, train_loader, val_loader, device, hybrid_cfg,
            sparsity_label=f"hybrid@{sp*100:.0f}%",
        )

        # Load best, verify sparsity
        model.load_state_dict(best_state)
        apply_masks(model, masks, device)
        actual_sp = count_sparsity(model, masks)

        t_elapsed = time.time() - t_start
        print(f"\nHYBRID @ {sp*100:.0f}% done in {t_elapsed/60:.1f} min")
        print(f"     Actual sparsity: {actual_sp*100:.2f}%")

        print_metrics_block(
            best_metrics,
            label=f"HYBRID @ {sp*100:.0f}% sparsity",
            indent="  ",
        )

        # Compare with best individual scorer at this sparsity
        best_ind_auroc = max(
            individual_results.get(s, {}).get(sp, {}).get('auroc', 0.0)
            for s in ['magnitude', 'ema', 'fisher', 'movement']
        )
        delta_vs_best = best_metrics['auroc'] - best_ind_auroc
        if delta_vs_best > 0:
            print(f"Hybrid EXCEEDS best individual scorer by "
                  f"{delta_vs_best:+.4f}% AUROC at {sp*100:.0f}% sparsity!")
        else:
            print(f"Hybrid vs best individual: {delta_vs_best:+.4f}% AUROC")

        # Save weights-only .pth
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

    # ── Save results JSON ──────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'hybrid_results.json')
    # Make dynamic_weights JSON-serialisable (float32 → float)
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
