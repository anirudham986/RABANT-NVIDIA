#!/usr/bin/env python3
"""
build_unseen_tests.py
=====================
Generate 3 balanced, unseen holdout test sets from the original source CSVs.

  07_clinvar_test_unseen.csv   — ClinVar only   (equal P / B)
  08_dbsnp_test_unseen.csv     — dbSNP only     (equal P / B)
  09_cbio_gnomad_test_unseen.csv — cBioPortal P + gnomAD B  (equal P / B,
                                    equal cBioPortal and gnomAD counts)

Constraints
-----------
1. Every sample must be UNSEEN in 05_consolidated_balanced.csv AND
   06_holdout_25k_unseen.csv.
2. All 3 files have the SAME total number of samples.
3. Perfect P / B balance in every file.
4. File 09 has equal cBioPortal (P) and gnomAD (B) contributions.
5. Maximize sample count while satisfying all above constraints.

Usage
-----
    python data/build_unseen_tests.py [--data_dir "crct dataset"]
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_variant_key(df: pd.DataFrame) -> pd.Series:
    """Create a canonical variant key: CHROM_POS_REF_ALT."""
    return (
        df["CHROM"].astype(str) + "_" +
        df["POS"].astype(str) + "_" +
        df["REF"].astype(str) + "_" +
        df["ALT"].astype(str)
    )


def standardize_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have an INT_LABEL column (0 = Benign, 1 = Pathogenic)."""
    if "INT_LABEL" in df.columns:
        return df
    if "LABEL" in df.columns:
        df["INT_LABEL"] = df["LABEL"].map(
            {"Benign": 0, "Pathogenic": 1}
        ).astype(int)
    else:
        raise KeyError("No LABEL or INT_LABEL column found")
    return df


def add_source_tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Add a SOURCE_TAG column for provenance tracking."""
    df["SOURCE_TAG"] = tag
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build 3 balanced unseen test sets."
    )
    parser.add_argument(
        "--data_dir", type=str, default="crct dataset",
        help="Path to the directory containing CSV files 01–06."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    seed = args.seed
    rng = np.random.RandomState(seed)

    print("=" * 70)
    print("  BUILD UNSEEN TEST SETS")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load the "seen" variant keys from 05 + 06
    # ------------------------------------------------------------------
    print("\n[1] Loading seen variant keys from 05 + 06 ...")

    df_05 = pd.read_csv(os.path.join(data_dir, "05_consolidated_balanced.csv"))
    df_06 = pd.read_csv(os.path.join(data_dir, "06_holdout_25k_unseen.csv"))

    seen_keys = set()

    # 05 has VARIANT_KEY column
    if "VARIANT_KEY" in df_05.columns:
        seen_keys.update(df_05["VARIANT_KEY"].values)
    else:
        seen_keys.update(make_variant_key(df_05).values)

    # 06 has VARIANT_KEY column
    if "VARIANT_KEY" in df_06.columns:
        seen_keys.update(df_06["VARIANT_KEY"].values)
    else:
        seen_keys.update(make_variant_key(df_06).values)

    print(f"    Total seen variant keys: {len(seen_keys):,}")
    print(f"      05 consolidated: {len(df_05):,}")
    print(f"      06 holdout:      {len(df_06):,}")

    # ------------------------------------------------------------------
    # 2. Load source CSVs and filter to unseen only
    # ------------------------------------------------------------------
    print("\n[2] Loading source CSVs and filtering to unseen ...")

    # --- ClinVar (01) ---
    df_clinvar = pd.read_csv(os.path.join(data_dir, "01_clinvar_75k_P+B.csv"))
    df_clinvar = standardize_label(df_clinvar)
    df_clinvar["VARIANT_KEY"] = make_variant_key(df_clinvar)
    df_clinvar = add_source_tag(df_clinvar, "ClinVar")
    df_clinvar_unseen = df_clinvar[~df_clinvar["VARIANT_KEY"].isin(seen_keys)].copy()

    clinvar_P = df_clinvar_unseen[df_clinvar_unseen["INT_LABEL"] == 1]
    clinvar_B = df_clinvar_unseen[df_clinvar_unseen["INT_LABEL"] == 0]
    print(f"    ClinVar  — Total: {len(df_clinvar):,} → "
          f"Unseen: {len(df_clinvar_unseen):,} "
          f"(P={len(clinvar_P):,}, B={len(clinvar_B):,})")

    # --- gnomAD (02) — Benign only ---
    df_gnomad = pd.read_csv(os.path.join(data_dir, "02_gnomad_55k_B.csv"))
    df_gnomad = standardize_label(df_gnomad)
    df_gnomad["VARIANT_KEY"] = make_variant_key(df_gnomad)
    df_gnomad = add_source_tag(df_gnomad, "gnomAD")
    df_gnomad_unseen = df_gnomad[~df_gnomad["VARIANT_KEY"].isin(seen_keys)].copy()

    gnomad_B = df_gnomad_unseen[df_gnomad_unseen["INT_LABEL"] == 0]
    gnomad_P = df_gnomad_unseen[df_gnomad_unseen["INT_LABEL"] == 1]  # should be 0
    print(f"    gnomAD   — Total: {len(df_gnomad):,} → "
          f"Unseen: {len(df_gnomad_unseen):,} "
          f"(P={len(gnomad_P):,}, B={len(gnomad_B):,})")

    # --- dbSNP (03) ---
    df_dbsnp = pd.read_csv(os.path.join(data_dir, "03_dbsnp_62k_P+B.csv"))
    df_dbsnp = standardize_label(df_dbsnp)
    df_dbsnp["VARIANT_KEY"] = make_variant_key(df_dbsnp)
    df_dbsnp = add_source_tag(df_dbsnp, "dbSNP")
    df_dbsnp_unseen = df_dbsnp[~df_dbsnp["VARIANT_KEY"].isin(seen_keys)].copy()

    dbsnp_P = df_dbsnp_unseen[df_dbsnp_unseen["INT_LABEL"] == 1]
    dbsnp_B = df_dbsnp_unseen[df_dbsnp_unseen["INT_LABEL"] == 0]
    print(f"    dbSNP    — Total: {len(df_dbsnp):,} → "
          f"Unseen: {len(df_dbsnp_unseen):,} "
          f"(P={len(dbsnp_P):,}, B={len(dbsnp_B):,})")

    # --- cBioPortal (04) — Pathogenic only ---
    df_cbio = pd.read_csv(os.path.join(data_dir, "04_cbioportal_63k_P.csv"))
    df_cbio = standardize_label(df_cbio)
    df_cbio["VARIANT_KEY"] = make_variant_key(df_cbio)
    df_cbio = add_source_tag(df_cbio, "cBioPortal")
    df_cbio_unseen = df_cbio[~df_cbio["VARIANT_KEY"].isin(seen_keys)].copy()

    cbio_P = df_cbio_unseen[df_cbio_unseen["INT_LABEL"] == 1]
    cbio_B = df_cbio_unseen[df_cbio_unseen["INT_LABEL"] == 0]  # should be 0
    print(f"    cBioPort — Total: {len(df_cbio):,} → "
          f"Unseen: {len(df_cbio_unseen):,} "
          f"(P={len(cbio_P):,}, B={len(cbio_B):,})")

    # ------------------------------------------------------------------
    # 3. Calculate maximum balanced counts per test set
    # ------------------------------------------------------------------
    print("\n[3] Calculating balanced split sizes ...")

    # 07 ClinVar: balanced P/B → limited by min(P, B) per class
    max_07_per_class = min(len(clinvar_P), len(clinvar_B))

    # 08 dbSNP: balanced P/B → limited by min(P, B) per class
    max_08_per_class = min(len(dbsnp_P), len(dbsnp_B))

    # 09 cBioPortal(P) + gnomAD(B): balanced → limited by min(cbio_P, gnomad_B)
    max_09_per_class = min(len(cbio_P), len(gnomad_B))

    print(f"    07 ClinVar  max per class: {max_07_per_class:,}")
    print(f"    08 dbSNP    max per class: {max_08_per_class:,}")
    print(f"    09 cBio+gno max per class: {max_09_per_class:,}")

    # All 3 files must have EQUAL total samples → equal per-class count
    per_class = min(max_07_per_class, max_08_per_class, max_09_per_class)
    total_per_file = per_class * 2

    print(f"\n  Final per-class count:  {per_class:,}")
    print(f"    Total samples per file: {total_per_file:,}")
    print(f"    Grand total (3 files):  {total_per_file * 3:,}")

    if per_class == 0:
        print("\nERROR: No unseen samples available! Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Sample balanced subsets
    # ------------------------------------------------------------------
    print("\n[4] Sampling balanced subsets ...")

    # 07: ClinVar
    clinvar_P_sampled = clinvar_P.sample(n=per_class, random_state=rng)
    clinvar_B_sampled = clinvar_B.sample(n=per_class, random_state=rng)
    df_07 = pd.concat([clinvar_P_sampled, clinvar_B_sampled], ignore_index=True)
    df_07 = df_07.sample(frac=1, random_state=rng).reset_index(drop=True)

    # 08: dbSNP
    dbsnp_P_sampled = dbsnp_P.sample(n=per_class, random_state=rng)
    dbsnp_B_sampled = dbsnp_B.sample(n=per_class, random_state=rng)
    df_08 = pd.concat([dbsnp_P_sampled, dbsnp_B_sampled], ignore_index=True)
    df_08 = df_08.sample(frac=1, random_state=rng).reset_index(drop=True)

    # 09: cBioPortal (P) + gnomAD (B)
    cbio_P_sampled = cbio_P.sample(n=per_class, random_state=rng)
    gnomad_B_sampled = gnomad_B.sample(n=per_class, random_state=rng)
    df_09 = pd.concat([cbio_P_sampled, gnomad_B_sampled], ignore_index=True)
    df_09 = df_09.sample(frac=1, random_state=rng).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Standardize output columns
    # ------------------------------------------------------------------
    # Keep a clean, consistent schema across all 3 test files
    output_cols = [
        "CHROM", "POS", "REF", "ALT", "INT_LABEL",
        "SOURCE", "SOURCE_TAG", "CONSEQUENCE", "VARIANT_KEY"
    ]

    for df in [df_07, df_08, df_09]:
        # Ensure all output cols exist
        for col in output_cols:
            if col not in df.columns:
                df[col] = ""

    df_07_out = df_07[output_cols].copy()
    df_08_out = df_08[output_cols].copy()
    df_09_out = df_09[output_cols].copy()

    # ------------------------------------------------------------------
    # 6. Cross-verify: zero leakage
    # ------------------------------------------------------------------
    print("\n[5] Cross-verifying zero leakage ...")

    for name, df_test in [("07", df_07_out), ("08", df_08_out), ("09", df_09_out)]:
        test_keys = set(df_test["VARIANT_KEY"].values)
        overlap_05 = test_keys & set(df_05["VARIANT_KEY"].values) if "VARIANT_KEY" in df_05.columns else set()
        overlap_06 = test_keys & set(df_06["VARIANT_KEY"].values) if "VARIANT_KEY" in df_06.columns else set()

        if overlap_05 or overlap_06:
            print(f"{name} LEAKAGE DETECTED! "
                  f"Overlap with 05: {len(overlap_05)}, 06: {len(overlap_06)}")
            sys.exit(1)
        else:
            print(f"{name}: Zero overlap with 05 and 06")

    # Cross-verify between test sets themselves
    keys_07 = set(df_07_out["VARIANT_KEY"].values)
    keys_08 = set(df_08_out["VARIANT_KEY"].values)
    keys_09 = set(df_09_out["VARIANT_KEY"].values)

    overlap_07_08 = keys_07 & keys_08
    overlap_07_09 = keys_07 & keys_09
    overlap_08_09 = keys_08 & keys_09

    if overlap_07_08 or overlap_07_09 or overlap_08_09:
        print(f"Inter-test overlap: 07∩08={len(overlap_07_08)}, "
              f"07∩09={len(overlap_07_09)}, 08∩09={len(overlap_08_09)}")
        # Note: inter-test overlap is expected=0 since they come from
        # different sources, but we verify anyway
    else:
        print("Zero inter-test overlap (07, 08, 09 are mutually exclusive)")

    # ------------------------------------------------------------------
    # 7. Save files
    # ------------------------------------------------------------------
    print("\n[6] Saving test files ...")

    out_07 = os.path.join(data_dir, "07_clinvar_test_unseen.csv")
    out_08 = os.path.join(data_dir, "08_dbsnp_test_unseen.csv")
    out_09 = os.path.join(data_dir, "09_cbio_gnomad_test_unseen.csv")

    df_07_out.to_csv(out_07, index=False)
    df_08_out.to_csv(out_08, index=False)
    df_09_out.to_csv(out_09, index=False)

    print(f"Yes {out_07}")
    print(f"Yes {out_08}")
    print(f"Yes {out_09}")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    for name, path, df_test in [
        ("07 ClinVar Test", out_07, df_07_out),
        ("08 dbSNP Test", out_08, df_08_out),
        ("09 cBio+gnomAD Test", out_09, df_09_out),
    ]:
        n_p = (df_test["INT_LABEL"] == 1).sum()
        n_b = (df_test["INT_LABEL"] == 0).sum()
        sources = df_test["SOURCE_TAG"].value_counts().to_dict()
        print(f"\n  {name}:")
        print(f"    Total:      {len(df_test):,}")
        print(f"    Pathogenic: {n_p:,}")
        print(f"    Benign:     {n_b:,}")
        print(f"    Sources:    {sources}")

    print(f"\n  Grand Total:  {len(df_07_out) + len(df_08_out) + len(df_09_out):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
