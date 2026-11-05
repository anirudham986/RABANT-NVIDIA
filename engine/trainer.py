# =====================================================================
# engine/trainer.py — Training loop with:
#   - Detailed per-batch logging (loss, acc, grad norm, GPU mem, LR)
#   - Full per-epoch validation metrics (acc, auroc, f1, mcc, confusion)
#   - Best-model tracking in memory (no intermediate checkpoint files)
#   - Mixed precision, gradient accumulation, cosine LR, early stopping
#   - Epoch-over-epoch deltas for monitoring learning progress
# =====================================================================

import os
import time
import json
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from copy import deepcopy

from .evaluator import evaluate, print_metrics


# =====================================================================
# Main training function
# =====================================================================

def train(model, train_loader, val_loader, device, criterion, cfg: dict,
          use_amp: bool = True, resume_from: str = None):
    """
    Full training loop for the NTv2 dual-sequence classifier.

    Args:
        model:        NTv2DualSeqClassifier (already on device)
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        device:       torch.device
        criterion:    Loss function (FocalLoss)
        cfg:          Configuration dictionary
        use_amp:      Whether to use mixed precision (GPU only)
        resume_from:  Optional path to a checkpoint to resume from

    Returns:
        (best_accuracy, best_auroc, history_dict)
    """
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    log_every = cfg.get('log_every_n_steps', 50)
    verbose = cfg.get('verbose', True)

    print(f"\n   {'='*60}")
    print(f"   TRAINING CONFIGURATION")
    print(f"   {'='*60}")
    print(f"   Epochs:          {cfg['epochs']}")
    print(f"   Batch size:      {cfg['batch_size']} (eff: {cfg['batch_size'] * cfg['grad_accum_steps']})")
    print(f"   Backbone LR:     {cfg['backbone_lr']}")
    print(f"   Head LR:         {cfg['head_lr']}")
    print(f"   Focal γ:         {cfg['focal_gamma']}")
    print(f"   Label smooth:    {cfg['label_smoothing']}")
    print(f"   Weight decay:    {cfg['weight_decay']}")
    print(f"   Warmup frac:     {cfg['warmup_fraction']}")
    print(f"   Grad accum:      {cfg['grad_accum_steps']}")
    print(f"   Max grad norm:   {cfg['max_grad_norm']}")
    print(f"   Patience:        {cfg['patience']}")
    print(f"   Dropout:         {cfg['dropout']}")
    print(f"   Layers unfrozen: {cfg['num_layers_to_unfreeze']}")
    print(f"   Log every:       {log_every} steps")
    print(f"   Save dir:        {save_dir}")
    print(f"   {'='*60}")

    # ------------------------------------------------------------------
    # Optimizer — differential LR (backbone very low, head higher)
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['backbone_lr'], cfg['head_lr']),
        weight_decay=cfg['weight_decay']
    )

    # ------------------------------------------------------------------
    # Cosine schedule with warmup
    # ------------------------------------------------------------------
    total_steps = (len(train_loader) // cfg['grad_accum_steps']) * cfg['epochs']
    warmup_steps = int(total_steps * cfg['warmup_fraction'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"\n   Scheduler: cosine | {total_steps} total steps | {warmup_steps} warmup")
    print(f"   Batches/epoch: {len(train_loader)}")

    # ------------------------------------------------------------------
    # Mixed precision scaler (AMP — GPU sm_70+ only)
    # ------------------------------------------------------------------
    scaler = GradScaler('cuda') if use_amp else None

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    best_auroc = 0.0
    best_acc = 0.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    start_epoch = 0
    training_start_time = time.time()
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_auroc': [], 'val_f1': [],
        'val_mcc': [], 'val_precision': [], 'val_recall': [],
        'val_specificity': [],
        'lr': [], 'epoch_time_s': [],
        'grad_norm': [],
    }

    # ------------------------------------------------------------------
    # Resume from checkpoint (if provided)
    # ------------------------------------------------------------------
    if resume_from is not None:
        print(f"\nRESUMING from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and ckpt.get('scaler_state_dict') is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        start_epoch = ckpt.get('epoch', 0) + 1
        best_auroc = ckpt.get('best_auroc', 0.0)
        best_acc = ckpt.get('best_acc', 0.0)
        if 'history' in ckpt:
            history = ckpt['history']
            # Ensure new history keys exist
            for key in ['val_mcc', 'val_precision', 'val_recall',
                        'val_specificity', 'epoch_time_s', 'grad_norm']:
                if key not in history:
                    history[key] = []

        best_state = deepcopy(model.state_dict())
        print(f"Resuming from epoch {start_epoch} "
              f"(best: AUROC={best_auroc:.2f}% Acc={best_acc:.2f}%)")
    else:
        print(f"\nStarting fresh training from epoch 1")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg['epochs']):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_grad_norms = []
        optimizer.zero_grad()

        # Print epoch header
        print(f"\n   {'─'*60}")
        print(f"   EPOCH {epoch + 1}/{cfg['epochs']}"
              f"  |  Best so far: AUROC={best_auroc:.2f}% "
              f"(ep {best_epoch + 1 if best_epoch >= 0 else '-'})"
              f"  |  Patience: {patience_counter}/{cfg['patience']}")
        print(f"   {'─'*60}")

        pbar = tqdm(
            train_loader,
            desc=f"   Train Ep{epoch+1}",
            leave=False,
            bar_format='{l_bar}{bar:30}{r_bar}'
        )

        for step, batch in enumerate(pbar):
            ref_ids = batch['ref_ids'].to(device)
            ref_mask = batch['ref_mask'].to(device)
            alt_ids = batch['alt_ids'].to(device)
            alt_mask = batch['alt_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_amp:
                with autocast('cuda'):
                    logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                    loss = criterion(logits, labels) / cfg['grad_accum_steps']
                scaler.scale(loss).backward()
            else:
                logits = model(ref_ids, ref_mask, alt_ids, alt_mask)
                loss = criterion(logits, labels) / cfg['grad_accum_steps']
                loss.backward()

            running_loss += loss.item() * cfg['grad_accum_steps']
            correct += (torch.argmax(logits, 1) == labels).sum().item()
            total += labels.size(0)

            # Gradient accumulation step
            if (step + 1) % cfg['grad_accum_steps'] == 0 or \
               (step + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)

                # Compute gradient norm BEFORE clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['max_grad_norm']
                ).item()
                epoch_grad_norms.append(grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Update progress bar
            current_acc = 100 * correct / total if total > 0 else 0
            current_loss = running_loss / (step + 1)
            pbar.set_postfix(
                loss=f'{current_loss:.4f}',
                acc=f'{current_acc:.1f}%',
                gnorm=f'{epoch_grad_norms[-1]:.2f}' if epoch_grad_norms else '-'
            )

            # Detailed per-batch logging
            if verbose and (step + 1) % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                print(f"\n   [Step {step+1}/{len(train_loader)}]"
                      f"  Loss={current_loss:.4f}"
                      f"  Acc={current_acc:.2f}%"
                      f"  GradNorm={epoch_grad_norms[-1]:.3f}"
                      f"  LR={lr_now:.2e}"
                      f"  GPU={gpu_mem:.1f}/{gpu_reserved:.1f}GB")

        # ---- Epoch training summary -----------------------------------
        elapsed = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        lr = scheduler.get_last_lr()[0]

        print(f"\nTRAIN  | Loss={avg_loss:.4f} | Acc={train_acc:.2f}% | "
              f"AvgGradNorm={avg_grad_norm:.3f} | LR={lr:.2e} | {elapsed:.0f}s")

        # ---- Full validation ------------------------------------------
        print(f"VALIDATION...")
        val_metrics = evaluate(
            model, val_loader, device, use_amp=use_amp,
            desc=f"Val Ep{epoch+1}"
        )

        # Print full validation breakdown
        print(f"\nVAL    | Acc={val_metrics['accuracy']:.2f}% | "
              f"AUROC={val_metrics['auroc']:.2f}% | "
              f"F1={val_metrics['f1']:.2f}% | "
              f"MCC={val_metrics['mcc']:.4f}")
        print(f"            | Prec={val_metrics['precision']:.2f}% | "
              f"Rec={val_metrics['recall']:.2f}% | "
              f"Spec={val_metrics['specificity']:.2f}%")
        print(f"            | TP={val_metrics['tp']}  FP={val_metrics['fp']}  "
              f"FN={val_metrics['fn']}  TN={val_metrics['tn']}")

        # ---- Epoch deltas (vs previous epoch) -------------------------
        if len(history['val_auroc']) > 0:
            prev_auroc = history['val_auroc'][-1]
            prev_acc = history['val_acc'][-1]
            prev_loss = history['train_loss'][-1]
            delta_auroc = val_metrics['auroc'] - prev_auroc
            delta_acc = val_metrics['accuracy'] - prev_acc
            delta_loss = avg_loss - prev_loss
            sign_auroc = '+' if delta_auroc >= 0 else ''
            sign_acc = '+' if delta_acc >= 0 else ''
            sign_loss = '+' if delta_loss >= 0 else ''
            print(f"DELTA  | "
                  f"ΔAUROC={sign_auroc}{delta_auroc:.2f}% | "
                  f"ΔAcc={sign_acc}{delta_acc:.2f}% | "
                  f"ΔLoss={sign_loss}{delta_loss:.4f}")

        # ---- Estimated time remaining ---------------------------------
        total_elapsed = time.time() - training_start_time
        epochs_done = epoch - start_epoch + 1
        avg_epoch_time = total_elapsed / epochs_done
        epochs_remaining = cfg['epochs'] - epoch - 1
        eta_seconds = avg_epoch_time * epochs_remaining
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_str = str(datetime.timedelta(seconds=int(total_elapsed)))
        print(f"TIME   | Epoch: {elapsed:.0f}s | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str}")

        # Track history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['lr'].append(lr)
        history['epoch_time_s'].append(elapsed)
        history['grad_norm'].append(avg_grad_norm)

        # ---- Best model tracking + early stopping ---------------------
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_acc = val_metrics['accuracy']
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0

            print(f"\nNEW BEST | AUROC={val_metrics['auroc']:.2f}% "
                  f"Acc={val_metrics['accuracy']:.2f}% "
                  f"F1={val_metrics['f1']:.2f}% "
                  f"MCC={val_metrics['mcc']:.4f} (epoch {epoch + 1})")
        else:
            patience_counter += 1
            remaining = cfg['patience'] - patience_counter
            print(f"\n   ⏸  No improvement | "
                  f"Best AUROC remains {best_auroc:.2f}% (epoch {best_epoch + 1}) | "
                  f"Patience: {remaining} epochs remaining")

            if patience_counter >= cfg['patience']:
                print(f"\n   ⏹  EARLY STOPPING triggered after {cfg['patience']} "
                      f"epochs without AUROC improvement")
                print(f"   Best model was at epoch {best_epoch + 1}")
                break

        # Save history JSON after every epoch (cheap, always useful)
        hist_path = os.path.join(save_dir, 'training_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # Restore best model weights before returning
    # ------------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model from epoch {best_epoch + 1} "
              f"(AUROC={best_auroc:.2f}%)")
    else:
        print(f"\nNo improvement during training — using final weights")

    total_time = time.time() - training_start_time
    total_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"   ⏱ Total training time: {total_str}")

    return best_acc, best_auroc, history


# =====================================================================
# Checkpoint helpers (minimal — only for resume support)
# =====================================================================

def load_checkpoint(path: str, model, optimizer=None, scheduler=None,
                    scaler=None, device=None):
    """
    Load a checkpoint and restore model weights.
    Optionally restores optimizer/scheduler/scaler for resume.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict') is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    return {
        'epoch': ckpt.get('epoch', 0),
        'best_auroc': ckpt.get('best_auroc', 0.0),
        'best_acc': ckpt.get('best_acc', 0.0),
        'history': ckpt.get('history', {}),
    }


def find_latest_checkpoint(save_dir: str):
    """
    Scan save_dir for checkpoint files and return the most recent one.
    Returns None if no checkpoint found.
    """
    import glob as _glob
    if not os.path.isdir(save_dir):
        return None
    patterns = ['checkpoint_best.pth', '*.pth']
    all_files = []
    for pat in patterns:
        all_files.extend(_glob.glob(os.path.join(save_dir, pat)))
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)
