#!/usr/bin/env python3
# =====================================================================
# data/build_holdout.py — ONE-TIME holdout test-set builder
#
# Run this ONCE (locally or on Kaggle) to produce:
#     06_holdout_25k_unseen.csv
#
# What it does:
#   1. Loads all 4 source CSVs
#   2. Loads the existing 05_consolidated_balanced.csv (100k training set)
#   3. Builds variant keys for every row
#   4. Filters OUT all variants that appear in the 100k consolidated set
#      → guarantees the holdout is 100% UNSEEN by the training model
#   5. Resolves label conflicts across sources (ClinVar highest priority)
#   6. Deduplicates leftover pool by (CHROM, POS, REF, ALT)
#   7. Samples 12,500 pathogenic + 12,500 benign (25k total)
#      — proportional source representation within each class
#   8. Runs zero-overlap verification against the training set
#   9. Writes 06_holdout_25k_unseen.csv
#
# Usage:
#   python data/build_holdout.py
#   python data/build_holdout.py --out_dir /kaggle/working
#   python data/build_holdout.py --n_per_class 12500
# =====================================================================

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter


# Label priority for conflict resolution (higher = more trusted)
LABEL_PRIORITY = {
    'ClinVar':       5,
    'dbSNP_ClinVar': 4,
    'gnomAD_v4.1':   3,
    'dbSNP_common':  2,
    'cBioPortal':    1,
}


def _get_priority(source: str) -> int:
    """Return label priority for a given source string."""
    for key, val in LABEL_PRIORITY.items():
        if key in str(source):
            return val
    return 0


def _find_data_dir() -> str:
    """Auto-discover the crct dataset directory."""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "crct dataset"),
        "/kaggle/input/dnai-dataset/crct dataset",
        "/kaggle/input/dnai-crct-dataset",
        "/kaggle/working/crct dataset",
        "./crct dataset",
        ".",
    ]
    for path in candidates:
        if os.path.isdir(path):
            probe = os.path.join(path, "01_clinvar_75k_P+B.csv")
            if os.path.exists(probe):
                return path

    # Walk /kaggle/input for any directory containing the file
    if os.path.exists("/kaggle/input"):
        import glob
        matches = glob.glob("/kaggle/input/**/01_clinvar_75k_P+B.csv", recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    raise FileNotFoundError(
        "Cannot find 'crct dataset' directory with the 4 source CSVs."
    )


def _make_variant_key(df: pd.DataFrame) -> pd.Series:
    """Build variant key strings from a DataFrame."""
    return (
        df['CHROM'].astype(str) + '_' +
        df['POS'].astype(str) + '_' +
        df['REF'].astype(str).str.upper() + '_' +
        df['ALT'].astype(str).str.upper()
    )


def load_all_sources(data_dir: str) -> pd.DataFrame:
    """
    Load all 4 source CSVs, standardise columns, tag with SOURCE_TAG,
    and concatenate into one DataFrame.
    """
    KEEP = ['CHROM', 'POS', 'REF', 'ALT', 'INT_LABEL', 'SOURCE', 'CONSEQUENCE']

    sources = [
        ('01_clinvar_75k_P+B.csv',    'ClinVar'),
        ('02_gnomad_55k_B.csv',       'gnomAD'),
        ('03_dbsnp_62k_P+B.csv',      'dbSNP'),
        ('04_cbioportal_63k_P.csv',   'cBioPortal'),
    ]

    frames = []
    for filename, tag in sources:
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path)
        available = [c for c in KEEP if c in df.columns]
        df = df[available].copy()
        df['SOURCE_TAG'] = tag
        print(f"   Loaded {filename}: {len(df):,} rows | "
              f"P={int((df['INT_LABEL']==1).sum()):,} "
              f"B={int((df['INT_LABEL']==0).sum()):,}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined['VARIANT_KEY'] = _make_variant_key(combined)
    print(f"\n   Total across all sources: {len(combined):,}")
    return combined


def build_holdout(data_dir: str, out_dir: str, n_per_class: int = 12_500,
                  seed: int = 42) -> str:
    """
    Build the 25k unseen holdout test set.

    Steps:
        1. Load all 4 source CSVs
        2. Load the 100k consolidated training set
        3. Remove all variants present in the training set
        4. Resolve label conflicts + deduplicate the leftover pool
        5. Sample n_per_class P + n_per_class B with proportional source mix
        6. Verify zero overlap with training set
        7. Save CSV

    Args:
        data_dir:     Directory containing the source CSVs
        out_dir:      Output directory for the holdout CSV
        n_per_class:  Number of samples per class (default 12,500 → 25k total)
        seed:         Random seed

    Returns:
        Path to the saved holdout CSV.
    """
    # ---- 1. Load all source CSVs ----
    print("\n" + "-" * 70)
    print("1. Loading all 4 source CSVs")
    print("-" * 70)
    df_all = load_all_sources(data_dir)

    # ---- 2. Load training set (consolidated 100k) ----
    print("\n" + "-" * 70)
    print("2. Loading 100k consolidated training set")
    print("-" * 70)
    cons_path = os.path.join(data_dir, '05_consolidated_balanced.csv')
    if not os.path.exists(cons_path):
        raise FileNotFoundError(
            f"Cannot find {cons_path}. "
            f"Run 'python data/build_consolidated.py' first."
        )
    df_cons = pd.read_csv(cons_path)
    if 'VARIANT_KEY' not in df_cons.columns:
        df_cons['VARIANT_KEY'] = _make_variant_key(df_cons)
    train_keys = set(df_cons['VARIANT_KEY'])
    print(f"   Training set: {len(df_cons):,} rows, {len(train_keys):,} unique variant keys")

    # ---- 3. Filter out training variants ----
    print("\n" + "-" * 70)
    print("3. Filtering out training variants (zero leakage)")
    print("-" * 70)
    df_leftover = df_all[~df_all['VARIANT_KEY'].isin(train_keys)].copy()
    print(f"   Leftover pool: {len(df_leftover):,} rows")
    print(f"   P={int((df_leftover['INT_LABEL']==1).sum()):,} "
          f"B={int((df_leftover['INT_LABEL']==0).sum()):,}")

    # Verify the filter worked
    leak_check = set(df_leftover['VARIANT_KEY']) & train_keys
    assert len(leak_check) == 0, f"FATAL: {len(leak_check)} variants leaked through filter!"
    print(f"Zero overlap with training set confirmed")

    # ---- 4. Resolve label conflicts + deduplicate ----
    print("\n" + "-" * 70)
    print("4. Deduplicating + resolving label conflicts in leftover pool")
    print("-" * 70)

    df_leftover['_priority'] = df_leftover['SOURCE'].apply(_get_priority)
    df_leftover = df_leftover.sort_values('_priority', ascending=False)

    # Detect conflicts
    key_labels = df_leftover.groupby('VARIANT_KEY')['INT_LABEL'].nunique()
    conflict_keys = key_labels[key_labels > 1].index
    print(f"   Label conflicts in leftover: {len(conflict_keys):,}")
    if len(conflict_keys) > 0:
        print(f"   → Resolving by ClinVar > gnomAD > dbSNP > cBioPortal priority")

    # Deduplicate — keep highest priority row per variant
    df_dedup = df_leftover.drop_duplicates(subset='VARIANT_KEY', keep='first').copy()
    df_dedup = df_dedup.drop(columns=['_priority'])
    df_dedup = df_dedup.reset_index(drop=True)

    print(f"   After dedup: {len(df_dedup):,} unique variants")
    print(f"   P={int((df_dedup['INT_LABEL']==1).sum()):,} "
          f"B={int((df_dedup['INT_LABEL']==0).sum()):,}")

    # ---- 5. Sample balanced holdout with proportional source mix ----
    print("\n" + "-" * 70)
    print(f"5. Sampling {n_per_class:,} per class ({2*n_per_class:,} total)")
    print("-" * 70)

    rng = np.random.RandomState(seed)

    def stratified_sample(subset, n_total, label_name):
        """Sample n_total rows, preserving SOURCE_TAG proportions."""
        fracs = subset['SOURCE_TAG'].value_counts(normalize=True)
        parts = []
        remaining = n_total
        tags = list(fracs.index)

        print(f"\n   {label_name} source distribution:")
        for i, tag in enumerate(tags):
            if i == len(tags) - 1:
                n_group = remaining
            else:
                n_group = int(fracs[tag] * n_total)
            grp = subset[subset['SOURCE_TAG'] == tag]
            n_group = min(n_group, len(grp), remaining)
            if n_group > 0:
                parts.append(grp.sample(n=n_group, random_state=seed))
                remaining -= n_group
                print(f"     {tag}: {n_group:,} samples")

        result = pd.concat(parts, ignore_index=True)
        # If we still need more (rounding), sample from largest source
        if len(result) < n_total:
            shortfall = n_total - len(result)
            remaining_pool = subset[~subset.index.isin(result.index)]
            if len(remaining_pool) >= shortfall:
                extra = remaining_pool.sample(n=shortfall, random_state=seed + 1)
                result = pd.concat([result, extra], ignore_index=True)
                print(f"     (extra {shortfall} to reach target)")

        return result

    df_p = df_dedup[df_dedup['INT_LABEL'] == 1]
    df_b = df_dedup[df_dedup['INT_LABEL'] == 0]

    n_p_avail = len(df_p)
    n_b_avail = len(df_b)
    print(f"   Available — P: {n_p_avail:,}  B: {n_b_avail:,}")

    if n_p_avail < n_per_class:
        raise ValueError(
            f"Not enough pathogenic variants: need {n_per_class:,}, "
            f"have {n_p_avail:,}. Reduce --n_per_class."
        )
    if n_b_avail < n_per_class:
        raise ValueError(
            f"Not enough benign variants: need {n_per_class:,}, "
            f"have {n_b_avail:,}. Reduce --n_per_class."
        )

    sampled_p = stratified_sample(df_p, n_per_class, "Pathogenic")
    sampled_b = stratified_sample(df_b, n_per_class, "Benign")

    holdout = pd.concat([sampled_p, sampled_b], ignore_index=True)
    holdout = holdout.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n   Holdout: {len(holdout):,} total | "
          f"P={int((holdout['INT_LABEL']==1).sum()):,} "
          f"B={int((holdout['INT_LABEL']==0).sum()):,}")

    # ---- 6. Final leakage verification ----
    print("\n" + "-" * 70)
    print("6. Final leakage verification")
    print("-" * 70)

    holdout_keys = set(holdout['VARIANT_KEY'])

    # Check 1: Zero overlap with training set
    overlap = holdout_keys & train_keys
    assert len(overlap) == 0, \
        f"FATAL: {len(overlap)} holdout variants overlap with training set!"
    print(f"Holdout–Training overlap: 0 variants")

    # Check 2: No duplicates within holdout
    dup_count = holdout.duplicated(subset='VARIANT_KEY').sum()
    assert dup_count == 0, \
        f"FATAL: {dup_count} duplicate variant keys in holdout!"
    print(f"Holdout internal duplicates: 0")

    # Check 3: Label balance
    n_p_final = int((holdout['INT_LABEL'] == 1).sum())
    n_b_final = int((holdout['INT_LABEL'] == 0).sum())
    pct_p = 100 * n_p_final / len(holdout)
    assert 45.0 <= pct_p <= 55.0, \
        f"FATAL: Holdout label imbalance: {pct_p:.1f}% pathogenic"
    print(f"Label balance: {pct_p:.1f}% pathogenic "
          f"(P={n_p_final:,} B={n_b_final:,})")

    # Check 4: Source diversity
    holdout_sources = sorted(holdout['SOURCE_TAG'].unique())
    print(f"Source diversity: {holdout_sources}")
    for tag in holdout_sources:
        tag_df = holdout[holdout['SOURCE_TAG'] == tag]
        labels = tag_df['INT_LABEL'].value_counts().to_dict()
        print(f"         {tag}: {len(tag_df):,} "
              f"(P={labels.get(1,0):,} B={labels.get(0,0):,})")

    print(f"\nALL HOLDOUT INTEGRITY CHECKS PASSED")

    # ---- 7. Save ----
    print("\n" + "-" * 70)
    print("7. Saving holdout dataset")
    print("-" * 70)

    out_cols = ['CHROM', 'POS', 'REF', 'ALT', 'INT_LABEL',
                'SOURCE', 'SOURCE_TAG', 'CONSEQUENCE', 'VARIANT_KEY']
    out_cols = [c for c in out_cols if c in holdout.columns]
    df_out = holdout[out_cols].copy()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '06_holdout_25k_unseen.csv')
    df_out.to_csv(out_path, index=False)

    print(f"   Saved: {out_path}")
    print(f"   Rows:    {len(df_out):,}")
    print(f"   Cols:    {list(df_out.columns)}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build 25k unseen holdout test set from leftover variants"
    )
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: same as source CSVs)')
    parser.add_argument('--n_per_class', type=int, default=12_500,
                        help='Samples per class (default: 12500 → 25k total)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("=" * 70)
    print("   DNAi — 25k UNSEEN HOLDOUT BUILDER")
    print("=" * 70)
    print(f"   Config: n_per_class={args.n_per_class:,} | seed={args.seed}")

    data_dir = _find_data_dir()
    print(f"   Data dir: {data_dir}")

    out_dir = args.out_dir or data_dir
    out_path = build_holdout(data_dir, out_dir, args.n_per_class, args.seed)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("   BUILD COMPLETE")
    print("=" * 70)
    print(f"   Output:       {out_path}")
    print(f"   Total:        {2 * args.n_per_class:,} variants")
    print(f"   Pathogenic:   {args.n_per_class:,}")
    print(f"   Benign:       {args.n_per_class:,}")
    print(f"   Leakage:      ZERO — verified against training set")
    print(f"   Duplicates:   ZERO — fully deduplicated")
    print(f"\n   Next step: python train.py --dataset consolidated_full")
    print("=" * 70)


if __name__ == '__main__':
    main()
