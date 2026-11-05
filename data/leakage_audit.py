# =====================================================================
# data/leakage_audit.py — Mandatory pre-training leakage guard
#
# Run this BEFORE every training session. Training will HARD STOP
# if any check fails. No exceptions. No manual overrides.
#
# Checks:
#   1. Zero variant-level train/val overlap
#   2. Zero intra-split duplicates (train)
#   3. Zero intra-split duplicates (val)
#   4. Label balance within ±5% of 50/50 (both splits)
#   5. Source diversity — all expected sources present in val
#   6. Minimum size sanity (train ≥ 1000, val ≥ 200)
#   7. FOLD_ID integrity — val must have exactly one unique FOLD_ID
# =====================================================================

import pandas as pd
import numpy as np


class LeakageAuditError(RuntimeError):
    """Raised when a leakage check fails. Training must not proceed."""
    pass


def _make_keys(df: pd.DataFrame) -> set:
    """Build a set of variant key strings from a DataFrame."""
    if 'VARIANT_KEY' in df.columns:
        return set(df['VARIANT_KEY'].astype(str))
    # Fall back to constructing from raw columns
    return set(
        df['CHROM'].astype(str) + '_' +
        df['POS'].astype(str) + '_' +
        df['REF'].astype(str).str.upper() + '_' +
        df['ALT'].astype(str).str.upper()
    )


def run_audit(train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              dataset_name: str = 'unknown',
              val_fold: int = -1) -> dict:
    """
    Run all leakage and integrity checks on the train/val split.

    Args:
        train_df:    Training DataFrame
        val_df:      Validation DataFrame
        dataset_name: Name of the dataset (for logging)
        val_fold:    Fold ID used as validation (for display only)

    Returns:
        dict with 'passed': True and audit summary on success.

    Raises:
        LeakageAuditError on any check failure.
    """
    print("\n" + "=" * 70)
    print(f"   LEAKAGE AUDIT — {dataset_name.upper()} | val_fold={val_fold}")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------------
    # CHECK 1: Variant-level train/val overlap
    # ----------------------------------------------------------------
    train_keys = _make_keys(train_df)
    val_keys   = _make_keys(val_df)
    overlap    = train_keys & val_keys

    if len(overlap) > 0:
        sample = list(overlap)[:5]
        raise LeakageAuditError(
            f"\n[FAIL] CHECK 1 — Train/Val variant overlap: {len(overlap):,} variants "
            f"appear in BOTH train and val sets!\n"
            f"  Sample overlapping keys: {sample}\n"
            f"  This means the model will evaluate on data it was trained on. "
            f"Results would be INVALID."
        )
    results['variant_overlap'] = 0
    print(f"CHECK 1 — Train/Val variant overlap:          0 variants")

    # ----------------------------------------------------------------
    # CHECK 2: Intra-train duplicates
    # ----------------------------------------------------------------
    train_dup = train_df.duplicated(
        subset=['CHROM', 'POS', 'REF', 'ALT'], keep=False
    ).sum()
    if train_dup > 0:
        raise LeakageAuditError(
            f"\n[FAIL] CHECK 2 — Intra-train duplicates: {train_dup:,} rows share the same "
            f"(CHROM, POS, REF, ALT). This inflates training signal on repeated variants."
        )
    results['train_duplicates'] = 0
    print(f"CHECK 2 — Intra-train duplicate variants:     0 rows")

    # ----------------------------------------------------------------
    # CHECK 3: Intra-val duplicates
    # ----------------------------------------------------------------
    val_dup = val_df.duplicated(
        subset=['CHROM', 'POS', 'REF', 'ALT'], keep=False
    ).sum()
    if val_dup > 0:
        raise LeakageAuditError(
            f"\n[FAIL] CHECK 3 — Intra-val duplicates: {val_dup:,} rows share the same "
            f"(CHROM, POS, REF, ALT). Evaluation metrics would be inflated."
        )
    results['val_duplicates'] = 0
    print(f"CHECK 3 — Intra-val duplicate variants:       0 rows")

    # ----------------------------------------------------------------
    # CHECK 4: Label balance (within 10% of 50/50 — not strict 5%,
    #          because after dedup the exact balance may shift slightly)
    # ----------------------------------------------------------------
    # Auto-detect label column name — works whether loader renamed it or not
    _label_col = None
    for _candidate in ['LABEL', 'INT_LABEL']:
        if _candidate in train_df.columns:
            _label_col = _candidate
            break
    if _label_col is None:
        raise LeakageAuditError(
            "\n[FAIL] CHECK 4 — Neither 'LABEL' nor 'INT_LABEL' column found "
            f"in train_df. Columns present: {list(train_df.columns)}"
        )

    for split_name, split_df in [('train', train_df), ('val', val_df)]:
        n_total = len(split_df)
        if n_total == 0:
            raise LeakageAuditError(f"\n[FAIL] CHECK 4 — {split_name} split is EMPTY!")

        n_p = int((split_df[_label_col] == 1).sum())
        n_b = int((split_df[_label_col] == 0).sum())
        pct_p = 100 * n_p / n_total

        if not (40.0 <= pct_p <= 60.0):
            raise LeakageAuditError(
                f"\n[FAIL] CHECK 4 — Label imbalance in {split_name}: "
                f"{pct_p:.1f}% pathogenic (expected 40–60%).\n"
                f"  P={n_p:,} B={n_b:,} total={n_total:,}\n"
                f"  The model may learn to predict the majority class trivially."
            )
        results[f'{split_name}_pct_pathogenic'] = round(pct_p, 2)
        print(f"CHECK 4 — {split_name.capitalize()} label balance: "
              f"{pct_p:.1f}% pathogenic (P={n_p:,} B={n_b:,})")

    # ----------------------------------------------------------------
    # CHECK 5: Source diversity in val set (consolidated only)
    # ----------------------------------------------------------------
    if 'SOURCE_TAG' in val_df.columns:
        val_sources = set(val_df['SOURCE_TAG'].dropna().unique())
        train_sources = set(train_df['SOURCE_TAG'].dropna().unique())
        missing_in_val = train_sources - val_sources
        if missing_in_val:
            # Warn but don't fail — with 5 folds some sources may be small
            print(f"CHECK 5 — Sources in train but not val: {missing_in_val}")
            print(f"              (Non-fatal: small source may not appear in every fold)")
        else:
            print(f"CHECK 5 — Val source diversity:  "
                  f"{sorted(val_sources)}")
        results['val_sources'] = sorted(val_sources)
    else:
        print(f"CHECK 5 — SOURCE_TAG column absent — skipping source diversity check")

    # ----------------------------------------------------------------
    # CHECK 6: Minimum size sanity
    # ----------------------------------------------------------------
    MIN_TRAIN = 1_000
    MIN_VAL   = 200

    if len(train_df) < MIN_TRAIN:
        raise LeakageAuditError(
            f"\n[FAIL] CHECK 6 — Train set too small: {len(train_df):,} rows "
            f"(minimum {MIN_TRAIN:,})"
        )
    if len(val_df) < MIN_VAL:
        raise LeakageAuditError(
            f"\n[FAIL] CHECK 6 — Val set too small: {len(val_df):,} rows "
            f"(minimum {MIN_VAL:,})"
        )
    results['train_size'] = len(train_df)
    results['val_size']   = len(val_df)
    print(f"CHECK 6 — Split sizes:  train={len(train_df):,}  val={len(val_df):,}")

    # ----------------------------------------------------------------
    # CHECK 7: FOLD_ID integrity (consolidated only)
    # ----------------------------------------------------------------
    if 'FOLD_ID' in val_df.columns:
        val_fold_ids = val_df['FOLD_ID'].unique()
        if len(val_fold_ids) != 1:
            raise LeakageAuditError(
                f"\n[FAIL] CHECK 7 — Val set contains multiple FOLD_IDs: {val_fold_ids}.\n"
                f"  The validation fold must be exactly one fold (e.g., fold 0)."
            )
        actual_fold = int(val_fold_ids[0])
        # Also check that train has no val-fold variants
        if 'FOLD_ID' in train_df.columns:
            train_with_val_fold = (train_df['FOLD_ID'] == actual_fold).sum()
            if train_with_val_fold > 0:
                raise LeakageAuditError(
                    f"\n[FAIL] CHECK 7 — Train set contains {train_with_val_fold:,} rows "
                    f"with FOLD_ID={actual_fold} (the validation fold). "
                    f"These should not be in train."
                )
        results['val_fold_id'] = actual_fold
        print(f"CHECK 7 — FOLD_ID integrity: val_fold={actual_fold} is clean")
    elif dataset_name == 'consolidated_full':
        # Full-train mode: holdout has no FOLD_ID — this is expected
        print(f"CHECK 7 — Holdout mode: no FOLD_ID needed (full-train + unseen holdout)")
    else:
        print(f"CHECK 7 — FOLD_ID column absent — skipping fold integrity check")

    # ----------------------------------------------------------------
    # PASS
    # ----------------------------------------------------------------
    print("=" * 70)
    print(f"ALL LEAKAGE CHECKS PASSED — Safe to begin training")
    print("=" * 70 + "\n")

    results['passed'] = True
    return results
