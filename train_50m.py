import os
import sys
import json
import time
import datetime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Local imports
from data import load_dataset, DualSeqDataset, run_audit, LeakageAuditError
from model import NTv2DualSeqClassifier, FocalLoss
from engine import train, evaluate, print_metrics
from utils import load_hg38, set_seed, get_device, supports_amp


# =====================================================================
# CONFIGURATION — 50M model with FULL fine-tuning (all 12 layers)
# =====================================================================

CFG = {
    # Model — NTv2-50M with ALL 12 layers unfrozen
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-50m-multi-species',
    'num_layers_to_unfreeze': 12,       # ALL 12 layers (full fine-tuning)
    'seq_length': 1000,                 # Same 1000bp context window

    # Data — identical to 100M
    'max_per_class': 50_000,
    'dataset_name': 'consolidated_full',

    # Training — identical to 100M
    'epochs': 20,
    'batch_size': 32,
    'grad_accum_steps': 4,              # Effective batch = 128
    'backbone_lr': 5e-6,               # Same differential LR
    'head_lr': 5e-4,
    'weight_decay': 0.01,
    'warmup_fraction': 0.15,
    'label_smoothing': 0.05,
    'focal_gamma': 1.5,
    'max_grad_norm': 1.0,
    'dropout': 0.2,
    'patience': 4,
    'seed': 42,

    # Logging
    'log_every_n_steps': 50,
    'verbose': True,

    # Performance
    'cudnn_benchmark': True,
    'num_workers': 4,

    # Output
    'save_dir': os.path.join('.', 'output', 'ntv2_50m_baseline_v2'),
}


# =====================================================================
# MAIN
# =====================================================================

def main():
    pipeline_start = time.time()

    # --- Banner ---
    print("\n" + "=" * 70)
    print("   NTv2 50M BASELINE v2 — FULL FINE-TUNING (ALL 12 LAYERS)")
    print("=" * 70)
    print(f"   Model:     {CFG['model_name']}")
    print(f"   Layers:    12 (all unfrozen — full fine-tuning)")
    print(f"   Heads:     16 attention heads per layer")
    print(f"   Hidden:    512")
    print(f"   FFN:       2048 (4× hidden)")
    print(f"   Dataset:   consolidated_full (100k train + 25k val)")
    print(f"   Purpose:   FAIR baseline for LTH compressed 100M comparison")
    print(f"   Approach:  Dual-sequence + Full FT + Focal loss")
    print(f"   Context:   {CFG['seq_length']}bp from hg38")
    print(f"   Epochs:    {CFG['epochs']} | Batch: {CFG['batch_size']} "
          f"(eff: {CFG['batch_size'] * CFG['grad_accum_steps']})")
    print(f"   Patience:  {CFG['patience']} epochs")
    print(f"   Output:    {CFG['save_dir']}")
    print()
    print(f"   ⚠️  FIX: Original train_50m_baseline.py only unfroze 4/12")
    print(f"        layers. This v2 script unfreezes ALL 12 layers.")
    print("=" * 70)

    # --- Seed ---
    set_seed(CFG['seed'], benchmark=CFG.get('cudnn_benchmark', True))

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
    print("3. Loading NT v2 Tokenizer (50M)")
    print("-" * 70)
    tokenizer = AutoTokenizer.from_pretrained(
        CFG['model_name'], trust_remote_code=True
    )
    max_tokens = min(256, tokenizer.model_max_length)
    print(f"   Tokenizer: vocab={tokenizer.vocab_size}, max_tokens={max_tokens}")

    # --- Model ---
    print("\n" + "-" * 70)
    print("4. Building NTv2-50M Dual-Sequence Classifier (FULL FT)")
    print("-" * 70)
    model = NTv2DualSeqClassifier(
        model_name=CFG['model_name'],
        num_layers_to_unfreeze=CFG['num_layers_to_unfreeze'],
        dropout=CFG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.classifier.parameters())
    n_layers = model.backbone.config.num_hidden_layers
    n_heads = model.backbone.config.num_attention_heads

    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable:        {trainable_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")
    print(f"   Backbone (train): {backbone_params:,}")
    print(f"   Head:             {head_params:,}")
    print(f"   Hidden size:      {model.hidden_size}")
    print(f"   Num layers:       {n_layers}")
    print(f"   Attention heads:  {n_heads}")
    print(f"   Layers unfrozen:  {CFG['num_layers_to_unfreeze']} / {n_layers}")

    # Sanity check: all layers should be unfrozen
    if CFG['num_layers_to_unfreeze'] >= n_layers:
        print(f"   ✅ FULL fine-tuning confirmed (all {n_layers} layers trainable)")
    else:
        print(f"   ⚠️  PARTIAL fine-tuning: {CFG['num_layers_to_unfreeze']}/{n_layers} layers")

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU memory:       {gpu_mem:.2f} GB allocated | "
              f"{gpu_reserved:.2f} GB reserved")

    # --- Data Loading ---
    print("\n" + "-" * 70)
    print("5. Loading Dataset: consolidated_full")
    print("-" * 70)

    # Auto-generate holdout CSV if needed
    from data.build_holdout import build_holdout, _find_data_dir
    try:
        data_dir_check = _find_data_dir()
        holdout_path = os.path.join(data_dir_check, '06_holdout_25k_unseen.csv')
        if not os.path.exists(holdout_path):
            print("\n   ⚠️ Holdout CSV not found — generating automatically...")
            build_holdout(data_dir_check, data_dir_check)
            print()
    except Exception as e:
        print(f"   ⚠️ Auto-generation failed: {e}")
        print("   Run 'python data/build_holdout.py' manually.")
        sys.exit(1)

    train_df, val_df = load_dataset(
        'consolidated_full',
        max_per_class=CFG['max_per_class'],
        seed=CFG['seed'],
        val_fold=0,
    )

    # ================================================================
    # MANDATORY LEAKAGE AUDIT
    # ================================================================
    try:
        run_audit(
            train_df, val_df,
            dataset_name='consolidated_full',
            val_fold=-1
        )
    except LeakageAuditError as e:
        print("\n" + "!" * 70)
        print("   🚨 LEAKAGE AUDIT FAILED — TRAINING ABORTED")
        print("!" * 70)
        print(str(e))
        sys.exit(1)

    # --- Build DualSeqDatasets ---
    print("\n" + "-" * 70)
    print("6. Building Dual-Sequence Datasets")
    print("-" * 70)

    print("\n   --- Train Set ---")
    train_dataset = DualSeqDataset(
        train_df, genome, tokenizer, has_chr,
        seq_len=CFG['seq_length'],
        max_tokens=max_tokens,
        seed=CFG['seed'],
    )

    print("\n   --- Validation Set ---")
    val_dataset = DualSeqDataset(
        val_df, genome, tokenizer, has_chr,
        seq_len=CFG['seq_length'],
        max_tokens=max_tokens,
        seed=CFG['seed'] + 1,
    )

    # --- DataLoaders ---
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=pin_memory,
    )

    print(f"\n   Train: {len(train_dataset):,} samples | "
          f"Val: {len(val_dataset):,} samples")
    print(f"   Batches/epoch: {len(train_loader)} train | {len(val_loader)} val")
    print(f"   Effective batch: {CFG['batch_size'] * CFG['grad_accum_steps']}")

    # --- Training ---
    print("\n" + "-" * 70)
    print("7. Training (Full Fine-Tuning — 50M, ALL 12 LAYERS)")
    print("-" * 70)

    criterion = FocalLoss(
        gamma=CFG['focal_gamma'],
        label_smoothing=CFG['label_smoothing'],
    )

    best_acc, best_auroc, history = train(
        model, train_loader, val_loader, device, criterion, CFG,
        use_amp=use_amp,
        resume_from=None,
    )

    # --- Final Evaluation ---
    print("\n" + "-" * 70)
    print("8. Final Evaluation on Best 50M Model")
    print("-" * 70)

    final = evaluate(model, val_loader, device, use_amp=use_amp,
                     desc="Final Eval (50M v2)")

    print(f"\n   FINAL RESULTS (best 50M model — all 12 layers fine-tuned):")
    print_metrics(final, prefix="   ")

    # --- Save Weights-Only .pth ---
    print("\n" + "-" * 70)
    print("9. Saving 50M Model Weights")
    print("-" * 70)

    save_dir = CFG['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    model_filename = "ntv2_50m_baseline_v2_final.pth"
    weights_path = os.path.join(save_dir, model_filename)

    # Save ONLY the state_dict
    torch.save(model.state_dict(), weights_path)

    weights_size_mb = os.path.getsize(weights_path) / 1e6
    n_params = sum(1 for _ in model.state_dict().keys())
    print(f"   ✅ Weights saved: {weights_path}")
    print(f"   📦 Size: {weights_size_mb:.1f} MB | {n_params} parameter tensors")
    print(f"   📦 Contains: model.state_dict() ONLY")

    # --- Save training history JSON ---
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   ✅ History: {history_path}")

    # --- Save final metrics JSON ---
    metrics_path = os.path.join(save_dir, 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"   ✅ Metrics: {metrics_path}")

    # --- Save config JSON ---
    config_path = os.path.join(save_dir, 'training_config.json')
    config_save = {}
    for k, v in CFG.items():
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
        'model_name': CFG['model_name'],
        'hidden_size': model.hidden_size,
        'num_hidden_layers': n_layers,
        'num_attention_heads': n_heads,
        'num_layers_to_unfreeze': CFG['num_layers_to_unfreeze'],
        'dropout': CFG['dropout'],
        'pooling': 'mean',
        'approach': 'dual_sequence_focal',
        'classifier_input_dim': model.hidden_size * 3,
    }
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"   ✅ Config: {config_path}")

    # --- Verify saved weights ---
    print("\n" + "-" * 70)
    print("10. Verifying Saved Weights")
    print("-" * 70)

    loaded_sd = torch.load(weights_path, map_location=device, weights_only=True)
    assert isinstance(loaded_sd, dict), "Saved file is not a dict!"

    verify_model = NTv2DualSeqClassifier(
        model_name=CFG['model_name'],
        num_layers_to_unfreeze=CFG['num_layers_to_unfreeze'],
        dropout=CFG['dropout'],
    ).to(device)
    verify_model.load_state_dict(loaded_sd)
    verify_metrics = evaluate(verify_model, val_loader, device, use_amp=use_amp,
                              desc="Verify (50M v2)")
    match = abs(verify_metrics['accuracy'] - final['accuracy']) < 0.01
    print(f"   Reload check: Acc={verify_metrics['accuracy']:.2f}% | "
          f"AUROC={verify_metrics['auroc']:.2f}% — "
          f"{'✅ PASS' if match else '❌ FAIL (mismatch!)'}")
    del verify_model

    # --- Summary ---
    total_pipeline_time = time.time() - pipeline_start
    total_str = str(datetime.timedelta(seconds=int(total_pipeline_time)))

    print("\n" + "=" * 70)
    print("   🎯 50M BASELINE v2 TRAINING COMPLETE (ALL 12 LAYERS FINE-TUNED)")
    print("=" * 70)
    print(f"   Model:      NTv2-50M (12 layers, 512 hidden, 16 heads)")
    print(f"   Fine-tuned: ALL 12/12 layers + embeddings")
    print(f"   Dataset:    consolidated_full (100k train + 25k val)")
    print(f"   Approach:   Dual-seq + Focal loss + Full fine-tuning")
    print(f"   Samples:    {len(train_dataset) + len(val_dataset):,} total "
          f"({len(train_dataset):,} train / {len(val_dataset):,} val)")
    print(f"   ──────────────────────────────────────")
    print(f"   Accuracy:   {final['accuracy']:.2f}%")
    print(f"   AUROC:      {final['auroc']:.2f}%")
    print(f"   F1:         {final['f1']:.2f}%")
    print(f"   MCC:        {final['mcc']:.4f}")
    print(f"   Precision:  {final['precision']:.2f}%")
    print(f"   Recall:     {final['recall']:.2f}%")
    print(f"   Specificity:{final['specificity']:.2f}%")
    print(f"   ──────────────────────────────────────")
    print(f"   Leakage:    ✅ ZERO (audited before training)")
    print(f"   Time:       {total_str}")
    print(f"   ──────────────────────────────────────")
    print(f"   Output: {save_dir}/")
    print(f"     📦 {model_filename}          — weights only")
    print(f"     📊 training_history.json      — per-epoch metrics")
    print(f"     📊 final_metrics.json          — final evaluation results")
    print(f"     ⚙️  training_config.json       — full config + architecture")
    print(f"   ──────────────────────────────────────")
    print(f"   NEXT STEP: Test on unseen data with:")
    print(f"     python3 test_unseen.py --weights {weights_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
