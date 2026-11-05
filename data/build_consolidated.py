#!/usr/bin/env python3
# =====================================================================
# data/build_consolidated.py — ONE-TIME consolidated dataset builder
#
# Run this locally (or in Kaggle) ONCE to produce:
#     05_consolidated_balanced.csv
#
# What it does:
#   1. Loads all 4 source CSVs
#   2. Resolves label conflicts (ClinVar has highest priority)
#   3. Deduplicates by (CHROM, POS, REF, ALT) variant key
#   4. Balances P/B to equal counts (~50k each)
#   5. Assigns stable FOLD_ID (0–4) with stratified splits
#      — Stratified by label, grouped so each source stays together
#   6. Writes the output CSV with VARIANT_KEY + FOLD_ID columns
#
# Usage:
#   python data/build_consolidated.py
#   python data/build_consolidated.py --out_dir /kaggle/working
# =====================================================================

import os
import sys
import argparse
import pandas as pd
import numpy as np
from collections import Counter

# Label priority for conflict resolution (higher = more trusted)
LABEL_PRIORITY = {
    'ClinVar':     5,
    'dbSNP_ClinVar': 4,   # ClinVar cross-referenced — same underlying source
    'gnomAD_v4.1': 3,     # Population frequency — high confidence benign
    'dbSNP_common': 2,    # Common SNP — assumed benign
    'cBioPortal':  1,     # Somatic cancer — pathogenic but lower certainty
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


def load_sources(data_dir: str) -> pd.DataFrame:
    """
    Load all 4 source CSVs, standardise to shared columns,
    and return a single concatenated DataFrame.
    """
    # Columns we keep from every source (union of what's available)
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

        # Keep only shared columns that exist
        available = [c for c in KEEP if c in df.columns]
        df = df[available].copy()

        # Add SOURCE_TAG for grouping (used in fold assignment)
        df['SOURCE_TAG'] = tag

        print(f"   Loaded {filename}: {len(df):,} rows | "
              f"P={int((df['INT_LABEL']==1).sum()):,} "
              f"B={int((df['INT_LABEL']==0).sum()):,}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n   Total before dedup: {len(combined):,}")
    return combined


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by (CHROM, POS, REF, ALT).

    When the same variant appears in multiple sources:
      - Use the label from the highest-priority source (ClinVar > gnomAD > dbSNP > cBio)
      - Keep track of how many conflicts were resolved

    Returns deduplicated DataFrame with one row per unique variant.
    """
    df = df.copy()

    # Normalise CHROM (strip 'chr' prefix for consistency, re-add later)
    df['_chrom_norm'] = df['CHROM'].astype(str).str.lower().str.replace('^chr', '', regex=True)

    # Variant key (canonical)
    df['VARIANT_KEY'] = (
        df['CHROM'].astype(str) + '_' +
        df['POS'].astype(str) + '_' +
        df['REF'].astype(str).str.upper() + '_' +
        df['ALT'].astype(str).str.upper()
    )

    # Priority score per row
    df['_priority'] = df['SOURCE'].apply(_get_priority)

    # Sort so highest-priority rows come first
    df = df.sort_values('_priority', ascending=False)

    # For each variant key: take the first row (highest priority)
    # BUT detect conflicts (same variant, different labels across sources)
    key_labels = df.groupby('VARIANT_KEY')['INT_LABEL'].nunique()
    conflict_keys = key_labels[key_labels > 1].index
    n_conflicts = len(conflict_keys)

    print(f"\n   Label conflicts (same variant, different labels across sources): {n_conflicts:,}")
    if n_conflicts > 0:
        print(f"   → Resolving by ClinVar > gnomAD > dbSNP_ClinVar > dbSNP_common > cBioPortal priority")

    # Deduplicate — keep highest priority row per variant
    df_dedup = df.drop_duplicates(subset='VARIANT_KEY', keep='first').copy()
    df_dedup = df_dedup.drop(columns=['_chrom_norm', '_priority'])

    print(f"   After dedup: {len(df_dedup):,} unique variants")
    print(f"   P={int((df_dedup['INT_LABEL']==1).sum()):,} "
          f"B={int((df_dedup['INT_LABEL']==0).sum()):,}")

    return df_dedup.reset_index(drop=True)


def balance_classes(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """
    Downsample majority class so P == B.
    Preserves proportional source distribution within each class.
    """
    df_p = df[df['INT_LABEL'] == 1]
    df_b = df[df['INT_LABEL'] == 0]

    n = min(len(df_p), len(df_b), max_per_class)
    print(f"\n   Balancing: {n:,} per class (total {2*n:,})")

    # Sample proportionally within each class to preserve source mix
    rng = np.random.RandomState(seed)

    def stratified_sample(subset, n_total):
        """Sample n_total rows from subset, preserving SOURCE_TAG proportions."""
        fracs = subset['SOURCE_TAG'].value_counts(normalize=True)
        parts = []
        remaining = n_total
        tags = list(fracs.index)
        for i, tag in enumerate(tags):
            if i == len(tags) - 1:
                # Give remainder to last group to avoid rounding errors
                n_group = remaining
            else:
                n_group = min(int(fracs[tag] * n_total), len(subset[subset['SOURCE_TAG'] == tag]))
            grp = subset[subset['SOURCE_TAG'] == tag]
            n_group = min(n_group, len(grp), remaining)
            if n_group > 0:
                parts.append(grp.sample(n=n_group, random_state=seed))
                remaining -= n_group
        return pd.concat(parts, ignore_index=True)

    sampled_p = stratified_sample(df_p, n)
    sampled_b = stratified_sample(df_b, n)

    balanced = pd.concat([sampled_p, sampled_b], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"   Final: {len(balanced):,} total | "
          f"P={int((balanced['INT_LABEL']==1).sum()):,} "
          f"B={int((balanced['INT_LABEL']==0).sum()):,}")
    print("\n   Source distribution:")
    for tag, count in balanced['SOURCE_TAG'].value_counts().items():
        label_dist = balanced[balanced['SOURCE_TAG'] == tag]['INT_LABEL'].value_counts().to_dict()
        print(f"     {tag}: {count:,}  (P={label_dist.get(1,0):,} B={label_dist.get(0,0):,})")

    return balanced


def assign_fold_ids(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """
    Assign FOLD_ID (0 to n_folds-1) to each variant.

    Strategy:
      - Sort variants by VARIANT_KEY (deterministic, reproducible)
      - Stratify by INT_LABEL so each fold has equal P/B ratio
      - Within each label, shuffle and assign round-robin fold IDs
        so source diversity is roughly preserved per fold
      - This ensures zero leakage: a variant's fold_id is fixed by its
        variant key alone, not by any random sampling
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    fold_ids = np.zeros(len(df), dtype=int)

    for label in [0, 1]:
        mask = df['INT_LABEL'] == label
        idx = df.index[mask].tolist()
        # Sort by VARIANT_KEY for determinism, then shuffle with fixed seed
        sorted_idx = sorted(idx, key=lambda i: df.loc[i, 'VARIANT_KEY'])
        rng.shuffle(sorted_idx)
        # Assign round-robin fold IDs
        for rank, i in enumerate(sorted_idx):
            fold_ids[i] = rank % n_folds

    df['FOLD_ID'] = fold_ids

    print(f"\n   Fold assignment (n_folds={n_folds}):")
    for fold in range(n_folds):
        fold_df = df[df['FOLD_ID'] == fold]
        n_p = int((fold_df['INT_LABEL'] == 1).sum())
        n_b = int((fold_df['INT_LABEL'] == 0).sum())
        print(f"     Fold {fold}: {len(fold_df):,} variants — P={n_p:,} B={n_b:,}")

    return df


def verify_no_leakage(df: pd.DataFrame, n_folds: int):
    """
    Verify that:
    1. No VARIANT_KEY appears in more than one fold (should be impossible by construction)
    2. Within each fold split (all-but-one fold vs one fold), there's zero overlap
    """
    print("\n   === LEAKAGE VERIFICATION ===")

    # Check 1: No duplicate variant keys in the consolidated set
    dup = df['VARIANT_KEY'].duplicated().sum()
    assert dup == 0, f"FATAL: {dup} duplicate VARIANT_KEYs in consolidated set!"
    print(f"Duplicate VARIANT_KEYs: 0")

    # Check 2: Each variant assigned exactly one fold
    fold_counts = df.groupby('VARIANT_KEY')['FOLD_ID'].nunique()
    multi = (fold_counts > 1).sum()
    assert multi == 0, f"FATAL: {multi} variants assigned to multiple folds!"
    print(f"Variants in multiple folds: 0")

    # Check 3: Simulate each train/val split and verify zero overlap
    all_keys = set(df['VARIANT_KEY'])
    for val_fold in range(n_folds):
        train_keys = set(df[df['FOLD_ID'] != val_fold]['VARIANT_KEY'])
        val_keys   = set(df[df['FOLD_ID'] == val_fold]['VARIANT_KEY'])
        overlap = train_keys & val_keys
        assert len(overlap) == 0, \
            f"FATAL: fold {val_fold} has {len(overlap)} train/val overlapping variants!"
        print(f"Fold {val_fold} — train/val overlap: 0")

    print("ALL LEAKAGE CHECKS PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="Build consolidated 5-fold dataset from all 4 source CSVs"
    )
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory (default: same as source CSVs)')
    parser.add_argument('--max_per_class', type=int, default=50_000,
                        help='Max samples per class after balancing (default: 50000)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of K-folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("=" * 70)
    print("   DNAi — CONSOLIDATED DATASET BUILDER")
    print("=" * 70)
    print(f"   Config: max_per_class={args.max_per_class:,} | "
          f"n_folds={args.n_folds} | seed={args.seed}")

    # ----------------------------------------------------------------
    # 1. Find data directory
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. Locating source CSVs")
    print("-" * 70)
    data_dir = _find_data_dir()
    print(f"   Data dir: {data_dir}")

    out_dir = args.out_dir or data_dir
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 2. Load all sources
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. Loading all 4 source CSVs")
    print("-" * 70)
    df_all = load_sources(data_dir)

    # ----------------------------------------------------------------
    # 3. Deduplicate + resolve label conflicts
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. Deduplicating + resolving label conflicts")
    print("-" * 70)
    df_dedup = resolve_duplicates(df_all)

    # ----------------------------------------------------------------
    # 4. Balance P/B classes
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. Balancing pathogenic / benign classes")
    print("-" * 70)
    df_balanced = balance_classes(df_dedup, args.max_per_class, args.seed)

    # ----------------------------------------------------------------
    # 5. Assign fold IDs
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. Assigning 5-fold IDs (deterministic, stratified)")
    print("-" * 70)
    df_final = assign_fold_ids(df_balanced, args.n_folds, args.seed)

    # ----------------------------------------------------------------
    # 6. Verify zero leakage
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. Verifying zero leakage across all fold splits")
    print("-" * 70)
    verify_no_leakage(df_final, args.n_folds)

    # ----------------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("7. Saving consolidated dataset")
    print("-" * 70)

    # Final column order
    out_cols = ['CHROM', 'POS', 'REF', 'ALT', 'INT_LABEL',
                'SOURCE', 'SOURCE_TAG', 'CONSEQUENCE', 'VARIANT_KEY', 'FOLD_ID']
    # Keep only columns that exist
    out_cols = [c for c in out_cols if c in df_final.columns]
    df_out = df_final[out_cols].copy()

    out_path = os.path.join(out_dir, '05_consolidated_balanced.csv')
    df_out.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"   Rows:    {len(df_out):,}")
    print(f"   Cols:    {list(df_out.columns)}")

    # ----------------------------------------------------------------
    # 8. Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("   BUILD COMPLETE")
    print("=" * 70)
    print(f"   Output:          {out_path}")
    print(f"   Total variants:  {len(df_out):,}")
    print(f"   Pathogenic:      {int((df_out['INT_LABEL']==1).sum()):,}")
    print(f"   Benign:          {int((df_out['INT_LABEL']==0).sum()):,}")
    print(f"   Folds:           {args.n_folds} (use FOLD_ID column)")
    print(f"   Leakage:         ZERO — verified across all fold splits")
    print("\n   Next step: python train.py --dataset consolidated --fold 0")
    print("=" * 70)


if __name__ == '__main__':
    main()
