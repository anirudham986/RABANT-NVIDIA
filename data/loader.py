# =====================================================================
# data/loader.py — Unified dataset loading, balancing, and splitting
#
# Handles all datasets through a single interface:
#   - 'single':     Single-file datasets (ClinVar, dbSNP)
#   - 'combine':    Two-file datasets (cBioPortal + gnomAD)
#   - 'kfold':      Consolidated dataset — splits by FOLD_ID, no random sampling
#
# KEY FIX vs old version:
#   The old code sampled rows BEFORE checking the SPLIT column, which meant
#   the pre-defined SPLIT column was always ignored (because after sampling,
#   the test split fell below the 5% fallback threshold). This caused
#   non-reproducible random splits on every run.
#
#   New behaviour:
#   - For 'kfold' (consolidated): split is 100% deterministic by FOLD_ID.
#     No random sampling before splitting. Zero leakage guaranteed.
#   - For 'single'/'combine': respects the SPLIT column if it exists and
#     gives adequate test coverage; only falls back if truly unavailable.
# =====================================================================

import os
import glob
import pandas as pd
import numpy as np
from .registry import get_dataset_info


def _find_data_dir(filenames: list) -> str:
    """
    Auto-discover the data directory by searching Kaggle input paths
    and the local 'crct dataset' folder.

    Args:
        filenames: List of CSV filenames to search for.

    Returns:
        Path to the directory containing the dataset files.

    Raises:
        FileNotFoundError if no valid directory is found.
    """
    search_roots = [
        "/kaggle/input/",
        "/kaggle/working/",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "crct dataset"),
        os.path.dirname(os.path.dirname(__file__)),
        "./crct dataset",
        ".",
    ]

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            if filenames[0] in files:
                if all(f in files for f in filenames):
                    return dirpath

    for pattern in ["/kaggle/input/**/" + filenames[0]]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    raise FileNotFoundError(
        f"Could not find dataset files {filenames} in any known location. "
        f"Searched: {search_roots}"
    )


def _load_kfold(data_dir: str, info: dict, val_fold: int) -> tuple:
    """
    Load the consolidated dataset and split by FOLD_ID.

    This is the ONLY correct way to split the consolidated dataset.
    No random sampling. No SPLIT column. Pure fold-ID arithmetic.

    Args:
        data_dir: Directory containing the CSV
        info:     Registry info for 'consolidated'
        val_fold: Fold ID (0–4) to use as validation

    Returns:
        (train_df, val_df) — each with columns [CHROM, POS, REF, ALT, INT_LABEL, ...]
    """
    filepath = os.path.join(data_dir, info['files'][0])
    df = pd.read_csv(filepath)

    print(f"   Loaded: {info['files'][0]} — {len(df):,} rows")

    fold_col = info.get('fold_id_column', 'FOLD_ID')
    n_folds  = info.get('n_folds', 5)

    if fold_col not in df.columns:
        raise ValueError(
            f"Consolidated CSV missing '{fold_col}' column. "
            f"Run 'python data/build_consolidated.py' first to generate it."
        )

    # Validate fold range
    actual_folds = sorted(df[fold_col].unique())
    if val_fold not in actual_folds:
        raise ValueError(
            f"val_fold={val_fold} not in dataset folds {actual_folds}."
        )

    label_col = info['label_column']

    train_df = df[df[fold_col] != val_fold].copy().reset_index(drop=True)
    val_df   = df[df[fold_col] == val_fold].copy().reset_index(drop=True)

    print(f"\n   K-Fold split (val_fold={val_fold}):")
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")
    print(f"   Train — P: {int((train_df[label_col]==1).sum()):,}  "
          f"B: {int((train_df[label_col]==0).sum()):,}")
    print(f"   Val   — P: {int((val_df[label_col]==1).sum()):,}  "
          f"B: {int((val_df[label_col]==0).sum()):,}")

    if 'SOURCE_TAG' in train_df.columns:
        print(f"\n   Train source distribution:")
        for tag, cnt in train_df['SOURCE_TAG'].value_counts().items():
            print(f"     {tag}: {cnt:,}")
        print(f"\n   Val source distribution:")
        for tag, cnt in val_df['SOURCE_TAG'].value_counts().items():
            print(f"     {tag}: {cnt:,}")

    train_df = train_df.rename(columns={label_col: 'LABEL'})
    val_df = val_df.rename(columns={label_col: 'LABEL'})

    return train_df, val_df


def _load_single(data_dir: str, info: dict, max_per_class: int,
                  seed: int) -> tuple:
    """
    Load a single-file dataset (ClinVar or dbSNP).
    Both classes exist in the same file.

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    filepath = os.path.join(data_dir, info['files'][0])
    df = pd.read_csv(filepath)
    print(f"   Loaded: {info['files'][0]} — {len(df):,} rows")
    print(f"   Benign (0):     {int((df[info['label_column']] == 0).sum()):,}")
    print(f"   Pathogenic (1): {int((df[info['label_column']] == 1).sum()):,}")

    return _balance_and_split(df, info['label_column'], max_per_class, seed)


def _load_combined(data_dir: str, info: dict, max_per_class: int,
                   seed: int) -> tuple:
    """
    Load and combine two single-class files (cBioPortal + gnomAD).
    cBioPortal provides pathogenic variants, gnomAD provides benign.

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    path_file  = info['files'][info['pathogenic_file_idx']]
    benign_file = info['files'][info['benign_file_idx']]

    df_path   = pd.read_csv(os.path.join(data_dir, path_file))
    df_benign = pd.read_csv(os.path.join(data_dir, benign_file))

    print(f"   Loaded: {path_file} — {len(df_path):,} pathogenic")
    print(f"   Loaded: {benign_file} — {len(df_benign):,} benign")

    label_col = info['label_column']

    # Standardize columns before concat
    base_cols = ['CHROM', 'POS', 'REF', 'ALT', label_col]
    optional  = ['SPLIT']
    keep_cols = base_cols + [c for c in optional
                             if c in df_path.columns and c in df_benign.columns]

    df_path   = df_path[keep_cols]
    df_benign = df_benign[keep_cols]

    df = pd.concat([df_path, df_benign], ignore_index=True)
    print(f"   Combined: {len(df):,} total")
    print(f"   Benign (0):     {int((df[label_col] == 0).sum()):,}")
    print(f"   Pathogenic (1): {int((df[label_col] == 1).sum()):,}")

    return _balance_and_split(df, label_col, max_per_class, seed)


def _balance_and_split(df: pd.DataFrame, label_col: str,
                       max_per_class: int, seed: int) -> tuple:
    """
    Balance class counts and split into train/test DataFrames.

    CORRECTED LOGIC vs. old version:
    - Old: sampled rows first → SPLIT column became useless (always fell through)
    - New: respects SPLIT column if it gives ≥10% test coverage;
           uses a PROPER stratified random split (by label) as fallback

    Returns:
        (train_df, test_df) with columns [CHROM, POS, REF, ALT, LABEL]
    """
    label_col_vals = df[label_col]
    n_benign      = int((label_col_vals == 0).sum())
    n_pathogenic  = int((label_col_vals == 1).sum())

    # ------------------------------------------------------------------
    # STEP 1: Use pre-defined SPLIT column if it gives adequate coverage
    # ------------------------------------------------------------------
    if 'SPLIT' in df.columns:
        test_size_from_split = int((df['SPLIT'] == 'test').sum())
        total = len(df)
        pct_test = test_size_from_split / total

        if pct_test >= 0.10:
            # The SPLIT column gives ≥10% test data — use it directly
            train_raw = df[df['SPLIT'] == 'train'].copy()
            test_raw  = df[df['SPLIT'] == 'test'].copy()

            # Balance each split independently
            train_df = _balance_within_split(train_raw, label_col, max_per_class, seed, 'train')
            test_df  = _balance_within_split(test_raw,  label_col, max_per_class, seed + 1, 'test')

            train_out = _standardise_cols(train_df, label_col)
            test_out  = _standardise_cols(test_df, label_col)

            print(f"\n   Using pre-defined SPLIT column ({pct_test*100:.1f}% test)")
            _print_split_stats(train_out, test_out)
            return train_out, test_out
        else:
            print(f"SPLIT column gives only {pct_test*100:.1f}% test — "
                  f"using stratified random 85/15 split instead")

    # ------------------------------------------------------------------
    # STEP 2: Stratified 85/15 split (preserves P/B ratio in both splits)
    # ------------------------------------------------------------------
    n_per_class = min(n_benign, n_pathogenic, max_per_class)

    df_b = df[df[label_col] == 0].sample(n=n_per_class, random_state=seed)
    df_p = df[df[label_col] == 1].sample(n=n_per_class, random_state=seed)

    # Split each class separately → guarantees label balance in both splits
    TEST_FRAC = 0.15
    n_test_per_class = max(1, int(n_per_class * TEST_FRAC))

    test_b  = df_b.sample(n=n_test_per_class, random_state=seed)
    train_b = df_b.drop(test_b.index)

    test_p  = df_p.sample(n=n_test_per_class, random_state=seed)
    train_p = df_p.drop(test_p.index)

    train_df = pd.concat([train_b, train_p]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat([test_b, test_p]).sample(frac=1, random_state=seed + 1).reset_index(drop=True)

    train_out = _standardise_cols(train_df, label_col)
    test_out  = _standardise_cols(test_df, label_col)

    print(f"\n   Balanced: {n_per_class:,} per class | Stratified 85/15 split")
    _print_split_stats(train_out, test_out)
    return train_out, test_out


def _balance_within_split(df: pd.DataFrame, label_col: str,
                          max_per_class: int, seed: int,
                          split_name: str) -> pd.DataFrame:
    """Balance a single split to equal P/B counts."""
    n_b = int((df[label_col] == 0).sum())
    n_p = int((df[label_col] == 1).sum())
    n   = min(n_b, n_p, max_per_class)

    df_b = df[df[label_col] == 0].sample(n=n, random_state=seed)
    df_p = df[df[label_col] == 1].sample(n=n, random_state=seed)
    return pd.concat([df_b, df_p]).sample(frac=1, random_state=seed).reset_index(drop=True)


def _standardise_cols(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Rename label column to 'LABEL' and keep only needed columns."""
    rename_map = {label_col: 'LABEL'}
    keep = ['CHROM', 'POS', 'REF', 'ALT', label_col]
    keep = [c for c in keep if c in df.columns]
    out  = df[keep].rename(columns=rename_map)
    return out.reset_index(drop=True)


def _print_split_stats(train_out: pd.DataFrame, test_out: pd.DataFrame):
    """Print train/test statistics."""
    print(f"   Train: {len(train_out):,} | Test: {len(test_out):,}")
    print(f"   Train — Benign: {int((train_out['LABEL']==0).sum()):,}, "
          f"Pathogenic: {int((train_out['LABEL']==1).sum()):,}")
    print(f"   Test  — Benign: {int((test_out['LABEL']==0).sum()):,}, "
          f"Pathogenic: {int((test_out['LABEL']==1).sum()):,}")


def load_dataset(dataset_name: str, max_per_class: int = 50_000,
                 seed: int = 42, val_fold: int = 0) -> tuple:
    """
    Main entry point: load any registered dataset by name.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal', 'consolidated'
        max_per_class: Maximum samples per class (for single/combine strategies)
        seed:          Random seed for reproducibility
        val_fold:      Validation fold (0–4) for 'consolidated' strategy only

    Returns:
        (train_df, val_df) — each with columns [CHROM, POS, REF, ALT, LABEL]
        For 'consolidated', also includes [SOURCE_TAG, VARIANT_KEY, FOLD_ID]
    """
    info = get_dataset_info(dataset_name)
    print(f"\n   Dataset: {info['description']}")
    print(f"   Strategy: {info['strategy']}")

    data_dir = _find_data_dir(info['files'])
    print(f"   Data dir: {data_dir}")

    if info['strategy'] == 'single':
        return _load_single(data_dir, info, max_per_class, seed)
    elif info['strategy'] == 'combine':
        return _load_combined(data_dir, info, max_per_class, seed)
    elif info['strategy'] == 'kfold':
        return _load_kfold(data_dir, info, val_fold)
    elif info['strategy'] == 'full_train':
        return _load_full_train(data_dir, info)
    else:
        raise ValueError(f"Unknown loading strategy: {info['strategy']}")


def _load_full_train(data_dir: str, info: dict) -> tuple:
    """
    Load the consolidated dataset as FULL training set (all 100k rows)
    and the 25k holdout CSV as the test set.

    This mode is for the final model that will be used for compression.
    No K-fold splitting — the entire 100k is used for training, and the
    model is evaluated on 25k completely unseen variants.

    Args:
        data_dir: Directory containing both CSV files
        info:     Registry info for 'consolidated_full'

    Returns:
        (train_df, test_df) — each with columns [CHROM, POS, REF, ALT, LABEL, ...]
    """
    train_file = info['files'][0]  # 05_consolidated_balanced.csv
    test_file  = info['files'][1]  # 06_holdout_25k_unseen.csv

    train_path = os.path.join(data_dir, train_file)
    test_path  = os.path.join(data_dir, test_file)

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Holdout test set not found: {test_path}\n"
            f"Run 'python data/build_holdout.py' first to generate it."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    label_col = info['label_column']

    print(f"   Loaded TRAIN: {train_file} — {len(train_df):,} rows")
    print(f"   Loaded TEST:  {test_file} — {len(test_df):,} rows")

    # ---- Cross-set leakage check ----
    def _make_key(df):
        if 'VARIANT_KEY' in df.columns:
            return set(df['VARIANT_KEY'].astype(str))
        return set(
            df['CHROM'].astype(str) + '_' +
            df['POS'].astype(str) + '_' +
            df['REF'].astype(str).str.upper() + '_' +
            df['ALT'].astype(str).str.upper()
        )

    train_keys = _make_key(train_df)
    test_keys  = _make_key(test_df)
    overlap    = train_keys & test_keys

    if len(overlap) > 0:
        raise ValueError(
            f"FATAL: {len(overlap)} variants overlap between training set and "
            f"holdout test set! Data leakage detected. "
            f"Regenerate holdout with 'python data/build_holdout.py'."
        )
    print(f"Train–Test overlap: 0 variants (zero leakage)")

    # Print statistics
    print(f"\n   Full-train mode (all samples for training):")
    print(f"   Train: {len(train_df):,} | "
          f"P={int((train_df[label_col]==1).sum()):,} "
          f"B={int((train_df[label_col]==0).sum()):,}")
    print(f"   Test:  {len(test_df):,} | "
          f"P={int((test_df[label_col]==1).sum()):,} "
          f"B={int((test_df[label_col]==0).sum()):,}")

    if 'SOURCE_TAG' in test_df.columns:
        print(f"\n   Holdout source distribution:")
        for tag, cnt in test_df['SOURCE_TAG'].value_counts().items():
            labels = test_df[test_df['SOURCE_TAG']==tag][label_col].value_counts().to_dict()
            print(f"     {tag}: {cnt:,} (P={labels.get(1,0):,} B={labels.get(0,0):,})")

    # Standardise column names
    train_df = train_df.rename(columns={label_col: 'LABEL'})
    test_df  = test_df.rename(columns={label_col: 'LABEL'})

    return train_df, test_df

