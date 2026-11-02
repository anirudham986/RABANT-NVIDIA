#!/usr/bin/env python3
# =============================================================================
# kfold_unseen_test_50m_v2.py - 5-fold unseen test evaluation for NTv2 50M v2
# =============================================================================
#
# Usage:
#   python3 kfold_unseen_test_50m_v2.py
#
# Evaluates each of the 5 fold checkpoints from kfold_50m_v2.py on the
# three source-specific unseen test sets:
#   07_clinvar_test_unseen.csv
#   08_dbsnp_test_unseen.csv
#   09_cbio_gnomad_test_unseen.csv
#
# This is the unseen-test analogue of kfold_50m_v2.py.
# Every fold model is tested independently, then metrics are aggregated
# across the five fold checkpoints.
#
# Model: NTv2-50M (12 layers, 16 heads, 512 hidden, all unfrozen)
# Weights: output/kfold_50m_v2/fold_{0..4}_best.pth
# Output:  output/kfold_50m_v2_unseen/
# =============================================================================

import argparse
import datetime
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data import DualSeqDataset
from engine import evaluate, print_metrics
from model import NTv2DualSeqClassifier
from utils import get_device, load_hg38, set_seed, supports_amp


TEST_FILES = {
    'clinvar': {
        'file': '07_clinvar_test_unseen.csv',
        'name': 'ClinVar (Unseen)',
        'description': 'Clinical variant significance - single-source test',
    },
    'dbsnp': {
        'file': '08_dbsnp_test_unseen.csv',
        'name': 'dbSNP (Unseen)',
        'description': 'Common / ClinVar cross-referenced variants',
    },
    'cbio_gnomad': {
        'file': '09_cbio_gnomad_test_unseen.csv',
        'name': 'cBioPortal + gnomAD (Unseen)',
        'description': 'Cancer somatic pathogenic + population benign',
    },
}

# NTv2-50M with ALL 12 layers unfrozen
MODEL_DEFAULTS = {
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
    'num_layers_to_unfreeze': 12,     # ALL 12 layers
    'dropout': 0.2,
    'seq_length': 1000,
    'batch_size': 32,
    'num_workers': 4,
    'seed': 42,
    'n_folds': 5,
}

WEIGHTS_DIR = os.path.join('.', 'output', 'kfold_50m_v2')
WEIGHTS_PATTERN = 'fold_{fold}_best.pth'
OUTPUT_DIR = os.path.join('.', 'output', 'kfold_50m_v2_unseen')
MODEL_LABEL = 'NTv2 50M (All 12 Layers Unfrozen)'

METRIC_KEYS = [
    ('auroc', 'AUROC (%)', 4),
    ('accuracy', 'Accuracy (%)', 4),
    ('f1', 'F1-Score (%)', 4),
    ('mcc', 'MCC', 6),
    ('precision', 'Precision (%)', 4),
    ('recall', 'Recall / Sensitivity (%)', 4),
    ('specificity', 'Specificity (%)', 4),
]


def find_data_dir(data_dir_hint=None):
    candidates = []
    if data_dir_hint:
        candidates.append(data_dir_hint)
    candidates.extend([
        os.path.join(_HERE, 'crct dataset'),
        '/kaggle/input',
        '/kaggle/working',
        'crct dataset',
        '.',
    ])

    required = [info['file'] for info in TEST_FILES.values()]
    for root in candidates:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            if all(name in files for name in required):
                return dirpath

    raise FileNotFoundError(
        f"Could not find unseen test CSVs {required}. Use --data_dir."
    )


def load_test_csv(data_dir, key):
    info = TEST_FILES[key]
    path = os.path.join(data_dir, info['file'])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)
    if 'LABEL' not in df.columns:
        if 'INT_LABEL' in df.columns:
            df = df.rename(columns={'INT_LABEL': 'LABEL'})
        else:
            raise KeyError(f"No LABEL or INT_LABEL column in {path}")

    required = ['CHROM', 'POS', 'REF', 'ALT', 'LABEL']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    return df


def build_test_loaders(data_dir, genome, tokenizer, has_chr, cfg, max_tokens, device_type):
    loaders = {}
    pin_memory = (device_type == 'cuda')

    for i, (key, info) in enumerate(TEST_FILES.items(), 1):
        test_df = load_test_csv(data_dir, key)
        n_p = int((test_df['LABEL'] == 1).sum())
        n_b = int((test_df['LABEL'] == 0).sum())

        print(f"\n  Test set {i}/3: {info['name']}")
        print(f"    File: {info['file']}")
        print(f"    Samples: {len(test_df):,} | P={n_p:,} B={n_b:,}")
        if 'SOURCE_TAG' in test_df.columns:
            for src, count in test_df['SOURCE_TAG'].value_counts().items():
                print(f"    {src}: {count:,}")

        dataset = DualSeqDataset(
            test_df,
            genome,
            tokenizer,
            has_chr,
            seq_len=cfg['seq_length'],
            max_tokens=max_tokens,
            seed=cfg['seed'] + 10_000 + i,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            pin_memory=pin_memory,
        )

        loaders[key] = {
            'info': info,
            'loader': loader,
            'csv_samples': len(test_df),
            'dataset_samples': len(dataset),
            'pathogenic': n_p,
            'benign': n_b,
        }

    return loaders


def resolve_weight_paths(weights_dir, pattern, n_folds):
    paths = []
    for fold_id in range(n_folds):
        path = os.path.join(weights_dir, pattern.format(fold=fold_id))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing fold {fold_id} weights: {path}\n"
                f"Make sure kfold_50m_v2.py has completed successfully."
            )
        paths.append(path)
    return paths


def load_model(weights_path, cfg, device):
    model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def summarize_metric_dict(metrics_list):
    summary = {}
    for key, _, _ in METRIC_KEYS:
        vals = np.array([m[key] for m in metrics_list], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ci_half = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        summary[key] = {
            'mean': mean,
            'std': std,
            'min': float(vals.min()),
            'max': float(vals.max()),
            'ci_95_lower': mean - ci_half,
            'ci_95_upper': mean + ci_half,
            'values': vals.tolist(),
        }
    return summary


def print_dataset_summary(dataset_key, fold_metrics):
    info = TEST_FILES[dataset_key]
    print(f"\n  {'-' * 78}")
    print(f"  {info['name']} - 5-fold unseen summary")
    print(f"  {'-' * 78}")
    print(f"  {'Metric':<28} {'Mean +/- Std':>18} {'Min':>10} {'Max':>10} {'95% CI':>22}")
    print(f"  {'-' * 92}")

    for key, label, decimals in METRIC_KEYS:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ci_half = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        fmt = f".{decimals}f"
        mean_std = f"{mean:{fmt}} +/- {std:{fmt}}"
        ci = f"[{mean - ci_half:{fmt}}, {mean + ci_half:{fmt}}]"
        print(
            f"  {label:<28} {mean_std:>18} "
            f"{vals.min():>10{fmt}} {vals.max():>10{fmt}} {ci:>22}"
        )


def print_overall_summary(overall_by_fold):
    print(f"\n  {'=' * 78}")
    print("  ALL UNSEEN TEST SETS COMBINED - macro-average over the 3 unseen sets")
    print(f"  {'=' * 78}")
    print(f"  {'Metric':<28} {'Mean +/- Std':>18} {'Min':>10} {'Max':>10} {'95% CI':>22}")
    print(f"  {'-' * 92}")

    for key, label, decimals in METRIC_KEYS:
        vals = np.array([m[key] for m in overall_by_fold], dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ci_half = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        fmt = f".{decimals}f"
        mean_std = f"{mean:{fmt}} +/- {std:{fmt}}"
        ci = f"[{mean - ci_half:{fmt}}, {mean + ci_half:{fmt}}]"
        print(
            f"  {label:<28} {mean_std:>18} "
            f"{vals.min():>10{fmt}} {vals.max():>10{fmt}} {ci:>22}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate five k-fold NTv2-50M checkpoints on unseen test CSVs.'
    )
    parser.add_argument('--weights_dir', type=str, default=WEIGHTS_DIR)
    parser.add_argument('--pattern', type=str, default=WEIGHTS_PATTERN)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--n_folds', type=int, default=MODEL_DEFAULTS['n_folds'])
    parser.add_argument('--batch_size', type=int, default=MODEL_DEFAULTS['batch_size'])
    parser.add_argument('--num_workers', type=int, default=MODEL_DEFAULTS['num_workers'])
    parser.add_argument('--seed', type=int, default=MODEL_DEFAULTS['seed'])
    args = parser.parse_args()

    start = time.time()
    set_seed(args.seed)

    cfg = {**MODEL_DEFAULTS}
    cfg['batch_size'] = args.batch_size
    cfg['num_workers'] = args.num_workers
    cfg['seed'] = args.seed

    weight_paths = resolve_weight_paths(args.weights_dir, args.pattern, args.n_folds)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 78)
    print("  5-FOLD UNSEEN TEST EVALUATION — NTv2 50M (ALL 12 LAYERS)")
    print("=" * 78)
    print(f"  Model:       {MODEL_LABEL}")
    print(f"  Model name:  {cfg['model_name']}")
    print(f"  Layers:      12 (all unfrozen)")
    print(f"  Heads:       16")
    print(f"  Hidden:      512")
    print(f"  Weights dir: {args.weights_dir}")
    print(f"  Pattern:     {args.pattern}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Folds:       {args.n_folds}")
    print("=" * 78)

    device = get_device()
    device_type = device.type
    use_amp = supports_amp()

    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:    {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")

    print(f"\n  {'-' * 78}")
    print("  STEP 0: Loading shared resources")
    print(f"  {'-' * 78}")
    genome, has_chr = load_hg38()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"  Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    print(f"\n  {'-' * 78}")
    print("  STEP 1: Loading unseen test datasets")
    print(f"  {'-' * 78}")
    data_dir = find_data_dir(args.data_dir)
    print(f"  Data dir: {data_dir}")
    test_loaders = build_test_loaders(data_dir, genome, tokenizer, has_chr, cfg, max_tokens, device_type)

    per_fold = {}
    per_dataset_metrics = {key: [] for key in TEST_FILES}
    overall_by_fold = []
    fold_times = []

    for fold_id, weights_path in enumerate(weight_paths):
        fold_start = time.time()
        print(f"\n\n{'=' * 78}")
        print(f"  FOLD {fold_id + 1} / {args.n_folds}")
        print(f"  Weights: {weights_path}")
        print(f"{'=' * 78}")

        set_seed(cfg['seed'] + fold_id * 1000)
        model = load_model(weights_path, cfg, device)

        fold_results = {}
        fold_metric_rows = []
        for key, payload in test_loaders.items():
            info = payload['info']
            print(f"\n  Evaluating fold {fold_id} on {info['name']}...")
            metrics = evaluate(
                model,
                payload['loader'],
                device,
                use_amp=use_amp,
                desc=f"Fold {fold_id} {key}",
            )
            print_metrics(metrics, prefix="  ")

            result = {
                'name': info['name'],
                'file': info['file'],
                'description': info['description'],
                'total_csv_samples': payload['csv_samples'],
                'extracted_samples': payload['dataset_samples'],
                'pathogenic': payload['pathogenic'],
                'benign': payload['benign'],
                'metrics': metrics,
            }
            fold_results[key] = result
            per_dataset_metrics[key].append(metrics)
            fold_metric_rows.append(metrics)

        fold_macro = {
            metric: float(np.mean([row[metric] for row in fold_metric_rows]))
            for metric, _, _ in METRIC_KEYS
        }
        fold_macro['n_samples'] = int(sum(row['n_samples'] for row in fold_metric_rows))
        overall_by_fold.append(fold_macro)

        elapsed = time.time() - fold_start
        fold_times.append(elapsed)
        per_fold[f'fold_{fold_id}'] = {
            'weights_file': os.path.abspath(weights_path),
            'test_sets': fold_results,
            'macro_average': fold_macro,
            'elapsed_seconds': elapsed,
        }

        fold_path = os.path.join(output_dir, f'fold_{fold_id}_unseen_results.json')
        with open(fold_path, 'w') as f:
            json.dump(per_fold[f'fold_{fold_id}'], f, indent=2, default=str)
        print(f"\n  Fold {fold_id} unseen results saved: {fold_path}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n\n{'=' * 78}")
    print("  PUBLICATION-READY UNSEEN TEST SUMMARY — NTv2 50M (ALL 12 LAYERS)")
    print(f"{'=' * 78}")

    aggregated_by_dataset = {}
    for key, metrics_list in per_dataset_metrics.items():
        print_dataset_summary(key, metrics_list)
        aggregated_by_dataset[key] = {
            'name': TEST_FILES[key]['name'],
            'file': TEST_FILES[key]['file'],
            'aggregated': summarize_metric_dict(metrics_list),
        }

    print_overall_summary(overall_by_fold)
    aggregated_overall = summarize_metric_dict(overall_by_fold)

    elapsed_total = time.time() - start
    total_time = str(datetime.timedelta(seconds=int(elapsed_total)))

    summary = {
        'experiment': f'5-Fold Unseen Test - {MODEL_LABEL}',
        'timestamp': datetime.datetime.now().isoformat(),
        'model_name': cfg['model_name'],
        'model_label': MODEL_LABEL,
        'model_config': {
            'num_hidden_layers': 12,
            'num_attention_heads': 16,
            'hidden_size': 512,
            'num_layers_to_unfreeze': cfg['num_layers_to_unfreeze'],
        },
        'weights_dir': os.path.abspath(args.weights_dir),
        'weights_pattern': args.pattern,
        'data_dir': os.path.abspath(data_dir),
        'test_files': TEST_FILES,
        'n_folds': args.n_folds,
        'seq_length': cfg['seq_length'],
        'batch_size': cfg['batch_size'],
        'num_workers': cfg['num_workers'],
        'total_time_seconds': elapsed_total,
        'total_time_human': total_time,
        'fold_times_seconds': fold_times,
        'per_fold': per_fold,
        'aggregated_by_dataset': aggregated_by_dataset,
        'aggregated_macro_average': aggregated_overall,
    }

    summary_path = os.path.join(output_dir, 'kfold_unseen_50m_v2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  ✅ Summary saved: {summary_path}")
    print(f"  Total time: {total_time}")
    print("=" * 78)


if __name__ == '__main__':
    main()
