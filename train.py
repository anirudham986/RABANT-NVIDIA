#!/usr/bin/env python3
# =====================================================================
# train.py — Main Entry Point for NTv2 Multi-Dataset Training
# =====================================================================
#
# Usage:
#   python train.py                         # Interactive dataset selection
#   python train.py --dataset clinvar       # ClinVar (direct)
#   python train.py --dataset dbsnp         # dbSNP (direct)
#   python train.py --dataset cbioportal    # cBioPortal + gnomAD
#   python train.py --dataset consolidated  # All 4 merged (RECOMMENDED)
#   python train.py --dataset consolidated_full   # Full 100k train + 25k test
#   python train.py --dataset consolidated --fold 2   # Specific validation fold
#
# Environment:
#   HG38_PATH=/path/to/hg38.fa   (set to skip genome search/download)
#
# GPU:   Designed for NVIDIA Quadro GV100 / V100 / A100 (32GB VRAM)
# Model: InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
# =====================================================================

import argparse
import os
import sys
import json
import time
import datetime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports
from config import get_config, DATASET_CHOICES
from data import load_dataset, DualSeqDataset, run_audit, LeakageAuditError
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate, print_metrics
from utils import load_hg38, set_seed, get_device, supports_amp


# =====================================================================
# 1. DATASET SELECTION (interactive or CLI argument)
# =====================================================================
def select_dataset() -> str:
    """Interactive dataset selection prompt."""
    print("\n" + "=" * 70)
    print("   NTv2 MULTI-DATASET TRAINING PIPELINE")
    print("=" * 70)
    print("\n   Available datasets:\n")
    print("     [1] clinvar      — ClinVar 75k (Pathogenic + Benign)")
    print("                        Clinical significance annotations")
    print()
    print("     [2] dbsnp        — dbSNP 62k (Pathogenic + Benign)")
    print("                        Common variants + ClinVar cross-ref")
    print()
    print("     [3] cbioportal   — cBioPortal 63k + gnomAD 55k (Combined)")
    print("                        Cancer somatic (P) + population (B)")
    print()
    print("     [4] consolidated — ALL 4 datasets merged & deduplicated ★ K-FOLD")
    print("                        Zero leakage • 5-fold stratified • source-diverse")
    print()
    print("     [5] consolidated_full — Full 100k train + 25k unseen holdout ★ FINAL")
    print("                        Train on ALL 100k • Test on 25k never-seen variants")
    print("                        Final model for compression pipeline")
    print()

    while True:
        choice = input("   Enter your choice [1/2/3/4/5 or name]: ").strip().lower()
        if choice in DATASET_CHOICES:
            dataset_name = DATASET_CHOICES[choice]
            print(f"\nSelected: {dataset_name}")
            return dataset_name
        print(f"Invalid choice '{choice}'. "
              f"Please enter 1-5 or a dataset name.")


def parse_args():
    """Parse CLI arguments for non-interactive usage."""
    parser = argparse.ArgumentParser(
        description="NTv2 Multi-Dataset Variant Effect Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                               # Interactive mode
  python train.py --dataset clinvar             # ClinVar dataset
  python train.py --dataset consolidated_full   # Full 100k+25k (RECOMMENDED)
  python train.py --dataset consolidated --fold 2  # Specific validation fold
  python train.py --resume /path/to/checkpoint.pth  # Resume training

Environment Variables:
  HG38_PATH=/path/to/hg38.fa    Path to hg38 reference genome
        """
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        choices=['clinvar', 'dbsnp', 'cbioportal', 'consolidated', 'consolidated_full'],
        help='Dataset to train on (skips interactive prompt)'
    )
    parser.add_argument(
        '--fold', type=int, default=None,
        help='Validation fold ID (0–4) for consolidated dataset'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Override number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--save_dir', type=str, default=None,
        help='Override checkpoint save directory'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to a checkpoint .pth file to resume training from'
    )
    return parser.parse_args()


# =====================================================================
# MAIN
# =====================================================================
def main():
    pipeline_start = time.time()
    args = parse_args()

    # --- Dataset selection ---
    if args.dataset:
        dataset_name = args.dataset
        print(f"\n   Dataset (CLI): {dataset_name}")
    else:
        dataset_name = select_dataset()

    # --- Configuration ---
    cfg = get_config(dataset_name, val_fold=args.fold)

    # Apply CLI overrides
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.save_dir:
        cfg['save_dir'] = args.save_dir

    # --- Resolve resume path ---
    resume_path = args.resume
    if resume_path is not None and not os.path.exists(resume_path):
        print(f"\nResume checkpoint not found: {resume_path}")
        sys.exit(1)

    # --- Banner ---
    print("\n" + "=" * 70)
    print("   NT v2 — VARIANT EFFECT PREDICTION PIPELINE")
    print("=" * 70)
    print(f"   Model:     {cfg['model_name']}")
    print(f"   Dataset:   {dataset_name}")
    if dataset_name == 'consolidated':
        print(f"   Val fold:  {cfg['val_fold']} / {cfg.get('n_folds', 5) - 1}")
    elif dataset_name == 'consolidated_full':
        print(f"   Mode:      Full 100k train + 25k unseen holdout test")
        print(f"   Purpose:   Final model for compression pipeline")
    print(f"   Approach:  Dual-sequence + Full FT + Focal loss")
    print(f"   Context:   {cfg['seq_length']}bp from hg38")
    print(f"   Epochs:    {cfg['epochs']} | Batch: {cfg['batch_size']} "
          f"(eff: {cfg['batch_size'] * cfg['grad_accum_steps']})")
    print(f"   Patience:  {cfg['patience']} epochs")
    print(f"   Resume:    {resume_path or 'None (fresh start)'}")
    print(f"   Output:    {cfg['save_dir']}")
    print("=" * 70)

    # --- Seed ---
    set_seed(cfg['seed'], benchmark=cfg.get('cudnn_benchmark', True))

    # --- Device ---
    print("\n" + "-" * 70)
    print("1. Device Setup")
    print("-" * 70)
    device = get_device()
    use_amp = supports_amp()

    # --- Reference Genome ---
    print("\n" + "-" * 70)
    print("2. Loading hg38 Reference Genome")
    print("-" * 70)
    genome, has_chr = load_hg38()

    # --- Tokenizer ---
    print("\n" + "-" * 70)
    print("3. Loading NT v2 Tokenizer")
    print("-" * 70)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'], trust_remote_code=True)
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Model ---
    print("\n" + "-" * 70)
    print("4. Building NTv2 Dual-Sequence Classifier")
    print("-" * 70)
    model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable:        {trainable_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"   Backbone (train): {backbone_params:,}")
    print(f"   Head:             {head_params:,}")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU memory:       {gpu_mem:.2f} GB allocated | "
              f"{gpu_reserved:.2f} GB reserved")

    # --- Data Loading ---
    print("\n" + "-" * 70)
    print(f"5. Loading Dataset: {dataset_name}")
    print("-" * 70)

    # Auto-generate holdout CSV if using consolidated_full and it doesn't exist
    if dataset_name == 'consolidated_full':
        from data.build_holdout import build_holdout, _find_data_dir
        try:
            data_dir_check = _find_data_dir()
            holdout_path = os.path.join(data_dir_check, '06_holdout_25k_unseen.csv')
            if not os.path.exists(holdout_path):
                print("\nHoldout CSV not found — generating automatically...")
                build_holdout(data_dir_check, data_dir_check)
                print()
        except Exception as e:
            print(f"Auto-generation failed: {e}")
            print("Run 'python data/build_holdout.py' manually.")
            sys.exit(1)

    train_df, val_df = load_dataset(
        dataset_name,
        max_per_class=cfg['max_per_class'],
        seed=cfg['seed'],
        val_fold=cfg.get('val_fold', 0),
    )

    # ================================================================
    # MANDATORY LEAKAGE AUDIT — Training will ABORT if this fails
    # ================================================================
    try:
        run_audit(
            train_df, val_df,
            dataset_name=dataset_name,
            val_fold=cfg.get('val_fold', -1)
        )
    except LeakageAuditError as e:
        print("\n" + "!" * 70)
        print("LEAKAGE AUDIT FAILED — TRAINING ABORTED")
        print("!" * 70)
        print(str(e))
        print("\n   Fix the data issue above before rerunning. "
              "Results on leaky data are scientifically invalid.")
        sys.exit(1)

    # --- Build DualSeqDatasets ---
    print("\n" + "-" * 70)
    print("6. Building Dual-Sequence Datasets")
    print("-" * 70)

    print("\n   --- Train Set ---")
    train_dataset = DualSeqDataset(
        train_df, genome, tokenizer, has_chr,
        seq_len=cfg['seq_length'],
        max_tokens=max_tokens,
        seed=cfg['seed'],
    )

    print("\n   --- Validation Set ---")
    val_dataset = DualSeqDataset(
        val_df, genome, tokenizer, has_chr,
        seq_len=cfg['seq_length'],
        max_tokens=max_tokens,
        seed=cfg['seed'] + 1,
    )

    # --- DataLoaders ---
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=pin_memory,
    )

    print(f"\n   Train: {len(train_dataset):,} samples | "
          f"Val: {len(val_dataset):,} samples")
    print(f"   Batches/epoch: {len(train_loader)} train | {len(val_loader)} val")
    print(f"   Effective batch: {cfg['batch_size'] * cfg['grad_accum_steps']}")

    # --- Training ---
    print("\n" + "-" * 70)
    print("7. Training (Full Fine-Tuning)")
    print("-" * 70)

    criterion = FocalLoss(
        gamma=cfg['focal_gamma'],
        label_smoothing=cfg['label_smoothing'],
    )

    best_acc, best_auroc, history = train(
        model, train_loader, val_loader, device, criterion, cfg,
        use_amp=use_amp,
        resume_from=resume_path,
    )

    # --- Final Evaluation ---
    print("\n" + "-" * 70)
    print("8. Final Evaluation on Best Model")
    print("-" * 70)

    final = evaluate(model, val_loader, device, use_amp=use_amp,
                     desc="Final Eval")

    print(f"\n   FINAL RESULTS (best model):")
    print_metrics(final, prefix="   ")

    # --- Save Weights-Only .pth ---
    print("\n" + "-" * 70)
    print("9. Saving Model Weights")
    print("-" * 70)

    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Single .pth file — ONLY model.state_dict()
    if dataset_name == 'consolidated_full':
        model_filename = f"ntv2_{dataset_name}_final.pth"
    elif dataset_name == 'consolidated':
        model_filename = f"ntv2_{dataset_name}_fold{cfg['val_fold']}_final.pth"
    else:
        model_filename = f"ntv2_{dataset_name}_final.pth"

    weights_path = os.path.join(save_dir, model_filename)

    # Save ONLY the state_dict — nothing else
    torch.save(model.state_dict(), weights_path)

    weights_size_mb = os.path.getsize(weights_path) / 1e6
    n_params = sum(1 for _ in model.state_dict().keys())
    print(f"Weights saved: {weights_path}")
    print(f"Size: {weights_size_mb:.1f} MB | {n_params} parameter tensors")
    print(f"Contains: model.state_dict() ONLY (no optimizer, no config, no history)")
    print(f"Use as input to compression pipeline")

    # --- Save training history JSON ---
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History: {history_path}")

    # --- Save final metrics JSON ---
    metrics_path = os.path.join(save_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"Metrics: {metrics_path}")

    # --- Save config JSON ---
    config_path = os.path.join(save_dir, 'training_config.json')
    # Convert config to JSON-serializable format
    config_save = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
            config_save[k] = v
        except (TypeError, ValueError):
            config_save[k] = str(v)
    config_save['final_accuracy'] = final['accuracy']
    config_save['final_auroc'] = final['auroc']
    config_save['final_f1'] = final['f1']
    config_save['final_mcc'] = final['mcc']
    config_save['model_architecture'] = {
        'model_name': cfg['model_name'],
        'hidden_size': model.hidden_size,
        'num_layers_to_unfreeze': cfg['num_layers_to_unfreeze'],
        'dropout': cfg['dropout'],
        'pooling': 'mean',
        'approach': 'dual_sequence_focal',
        'classifier_input_dim': model.hidden_size * 3,
    }
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"Config: {config_path}")

    # --- Verify saved weights ---
    print("\n" + "-" * 70)
    print("10. Verifying Saved Weights")
    print("-" * 70)

    loaded_sd = torch.load(weights_path, map_location=device, weights_only=True)

    # Verify it's a raw state_dict (not wrapped in another dict)
    assert isinstance(loaded_sd, dict), "Saved file is not a dict!"
    first_key = next(iter(loaded_sd))
    assert isinstance(loaded_sd[first_key], torch.Tensor), \
        f"First value is not a tensor: {type(loaded_sd[first_key])}"

    # Quick numerical check: load into fresh model and evaluate
    verify_model = NTv2DualSeqClassifier(
        model_name=cfg['model_name'],
        num_layers_to_unfreeze=cfg['num_layers_to_unfreeze'],
        dropout=cfg['dropout'],
    ).to(device)
    verify_model.load_state_dict(loaded_sd)
    verify_metrics = evaluate(verify_model, val_loader, device, use_amp=use_amp,
                              desc="Verify")
    match = abs(verify_metrics['accuracy'] - final['accuracy']) < 0.01
    print(f"   Reload check: Acc={verify_metrics['accuracy']:.2f}% | "
          f"AUROC={verify_metrics['auroc']:.2f}% — "
          f"{'PASS' if match else 'FAIL (mismatch!)'}")
    del verify_model

    # --- Summary ---
    total_pipeline_time = time.time() - pipeline_start
    total_str = str(datetime.timedelta(seconds=int(total_pipeline_time)))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Dataset:    {dataset_name}")
    if dataset_name == 'consolidated':
        print(f"   Val fold:   {cfg['val_fold']} / {cfg.get('n_folds', 5) - 1}")
    elif dataset_name == 'consolidated_full':
        print(f"   Mode:       Full 100k train + 25k unseen holdout")
    print(f"   Approach:   Dual-seq + Focal loss + Full fine-tuning")
    print(f"   Samples:    {len(train_dataset) + len(val_dataset):,} total "
          f"({len(train_dataset):,} train / {len(val_dataset):,} val)")
    print(f"   Context:    {cfg['seq_length']}bp from hg38")
    print(f"   Batch:      {cfg['batch_size']} × {cfg['grad_accum_steps']} "
          f"= {cfg['batch_size'] * cfg['grad_accum_steps']} effective")
    print(f"   ──────────────────────────────────────")
    print(f"   Accuracy:   {final['accuracy']:.2f}%")
    print(f"   AUROC:      {final['auroc']:.2f}%")
    print(f"   F1:         {final['f1']:.2f}%")
    print(f"   MCC:        {final['mcc']:.4f}")
    print(f"   Precision:  {final['precision']:.2f}%")
    print(f"   Recall:     {final['recall']:.2f}%")
    print(f"   Specificity:{final['specificity']:.2f}%")
    print(f"   ──────────────────────────────────────")
    print(f"   Leakage:    ZERO (audited before training)")
    print(f"   Time:       {total_str}")
    print(f"   ──────────────────────────────────────")
    print(f"   Output directory: {save_dir}/")
    print(f"{model_filename}          — weights only (for compression)")
    print(f"training_history.json      — per-epoch metrics")
    print(f"final_metrics.json          — final evaluation results")
    print(f"training_config.json       — full config + architecture info")
    print("=" * 70)


if __name__ == '__main__':
    main()
