import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from copy import deepcopy
from tqdm import tqdm

from compression.shared import evaluate_full


# =============================================================================
# GRADIENT CHECKPOINTING HELPER
# =============================================================================

def _enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing on the NTv2 backbone to trade compute
    for memory: activations are NOT stored during the forward pass and are
    instead recomputed during backward, roughly halving activation VRAM.

    Tries multiple strategies in priority order to handle different
    versions of the HuggingFace ESM implementation:
      1. model.backbone.encoder.gradient_checkpointing = True
         (direct flag used by modeling_esm.py in the forward loop)
      2. model.backbone.gradient_checkpointing_enable()
         (standard HF PreTrainedModel API)
      3. model.backbone.config.gradient_checkpointing = True
         (config-based flag for older HF versions)
    """
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        return

    enabled = False

    # Strategy 1: direct encoder flag (fastest — matches modeling_esm.py)
    encoder = getattr(backbone, 'encoder', None)
    if encoder is not None and hasattr(encoder, 'gradient_checkpointing'):
        encoder.gradient_checkpointing = True
        enabled = True

    # Strategy 2: standard HF API
    if not enabled and hasattr(backbone, 'gradient_checkpointing_enable'):
        try:
            backbone.gradient_checkpointing_enable()
            enabled = True
        except Exception:
            pass

    # Strategy 3: config flag
    if not enabled and hasattr(backbone, 'config'):
        try:
            backbone.config.gradient_checkpointing = True
            enabled = True
        except Exception:
            pass

    if enabled:
        print("Gradient checkpointing enabled on backbone "
              "(activation memory ~50% lower)")
    else:
        print("Could not enable gradient checkpointing — "
              "proceeding without it (OOM risk)")


# =============================================================================
# 1. EMA GRADIENT TRACKER
# =============================================================================

class EMAGradientTracker:
    """
    Tracks the exponential moving average of gradient magnitudes per weight.

    Update rule (per backward pass):
        s_i^(t) = β · s_i^(t-1) + (1 - β) · |∂L/∂w_i|

    Bias correction:
        s_i_corrected = s_i / (1 - β^t)

    Weights with persistently high gradient activity across all training
    steps are most critical — EMA captures this cumulative signal.
    """

    def __init__(self, model: nn.Module, beta: float = 0.999):
        self.beta = beta
        self.step_count = 0
        self.ema: dict = {}
        for name, param in model.named_parameters():
            if _is_prunable(name, param):
                self.ema[name] = torch.zeros_like(param.data, device='cpu')

    def update(self, model: nn.Module):
        """Call immediately after loss.backward() before optimizer step."""
        self.step_count += 1
        for name, param in model.named_parameters():
            if name in self.ema and param.grad is not None:
                g = param.grad.data.abs().cpu()
                self.ema[name].mul_(self.beta).add_(g, alpha=1.0 - self.beta)

    def get_scores(self) -> dict:
        """Return bias-corrected EMA scores."""
        correction = max(1.0 - self.beta ** self.step_count, 1e-8)
        return {name: val / correction for name, val in self.ema.items()}


# =============================================================================
# 2. FISHER INFORMATION ESTIMATOR
# =============================================================================

class FisherInformationEstimator:
    """
    Estimates the diagonal of the Fisher Information Matrix.

    F_i = E[(∂L/∂w_i)²]

    High F_i ⟹ perturbing w_i strongly changes the loss ⟹ w_i is critical.
    Approximated via Monte Carlo over n_batches training samples.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.fisher: dict = {}
        self.n_batches = 0
        for name, param in model.named_parameters():
            if _is_prunable(name, param):
                self.fisher[name] = torch.zeros_like(param.data, device='cpu')

    def accumulate(self, model: nn.Module, loader, n_batches: int = 512):
        """
        Accumulate squared gradients over n_batches.
        Uses the true label CE loss (not focal) for cleaner Fisher estimation.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        count = 0

        pbar = tqdm(loader, desc=f"   Fisher ({n_batches} batches)",
                    total=n_batches, leave=False)
        for batch in pbar:
            if count >= n_batches:
                break
            model.zero_grad()
            ri = batch['ref_ids'].to(self.device)
            rm = batch['ref_mask'].to(self.device)
            ai = batch['alt_ids'].to(self.device)
            am = batch['alt_mask'].to(self.device)
            labs = batch['labels'].to(self.device)

            out = model(ri, rm, ai, am)
            loss = criterion(out, labs)
            loss.backward()

            for name, param in model.named_parameters():
                if name in self.fisher and param.grad is not None:
                    self.fisher[name].add_(param.grad.data.pow(2).cpu())

            count += 1
            self.n_batches += 1

        print(f"Fisher accumulated from {self.n_batches} batches")

    def get_scores(self) -> dict:
        """Return averaged Fisher scores."""
        n = max(self.n_batches, 1)
        return {name: val / n for name, val in self.fisher.items()}


# =============================================================================
# 3. IMPORTANCE SCORING FUNCTIONS
# =============================================================================

def _is_prunable(name: str, param: torch.Tensor) -> bool:
    """A parameter is prunable if it's a weight matrix (≥2D) and not embeddings."""
    return (param.requires_grad
            and param.dim() >= 2
            and 'embedding' not in name.lower())


def percentile_rank(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor's values to percentile ranks in [0, 1].
    Rank 0 = smallest, rank 1 = largest.
    Shape is preserved.
    """
    flat = tensor.flatten().float()
    n = flat.numel()
    if n == 0:
        return tensor.clone()
    sorted_idx = flat.argsort()
    ranks = torch.zeros(n, device='cpu')
    ranks[sorted_idx] = torch.linspace(0.0, 1.0, n)
    return ranks.reshape(tensor.shape)


def log_percentile_rank(tensor: torch.Tensor) -> torch.Tensor:
    """
    Log-smoothed percentile rank: log(1 + x) before ranking.
    Compresses extreme outlier gradients so that one huge value
    doesn't dominate the rank distribution.
    """
    return percentile_rank(torch.log1p(tensor.float().clamp(min=0)))


def compute_magnitude_scores(model: nn.Module) -> dict:
    """Signal 1: absolute weight magnitude |w_i|."""
    scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param):
            scores[name] = param.data.abs().cpu()
    return scores


def compute_ema_scores(ema_tracker: EMAGradientTracker) -> dict:
    """Signal 2: bias-corrected EMA of gradient magnitudes."""
    return ema_tracker.get_scores()


def compute_fisher_scores(fisher_est: FisherInformationEstimator) -> dict:
    """Signal 3: diagonal Fisher Information (averaged)."""
    return fisher_est.get_scores()


def compute_movement_scores(model: nn.Module, initial_weights: dict) -> dict:
    """Signal 4: |w_final - w_init| after a short warmup pass."""
    scores = {}
    for name, param in model.named_parameters():
        if _is_prunable(name, param) and name in initial_weights:
            delta = (param.data.cpu() - initial_weights[name]).abs()
            scores[name] = delta
    return scores


# =============================================================================
# 4. COMPOSITE IMPORTANCE (for hybrid scorer)
# =============================================================================

def compute_composite_scores(
    model: nn.Module,
    initial_weights: dict,
    ema_scores: dict,
    fisher_scores: dict,
    weights: dict,  # {'magnitude': w1, 'ema': w2, 'fisher': w3, 'movement': w4}
) -> dict:
    """
    Weighted rank fusion of all 4 importance signals.
    weights must sum to 1 (normalised internally if not).
    Returns composite importance dict {name: tensor}.
    """
    w_sum = sum(weights.values())
    w = {k: v / w_sum for k, v in weights.items()}

    composite = {}
    for name, param in model.named_parameters():
        if not _is_prunable(name, param):
            continue

        # Signal 1 — magnitude
        mag = param.data.abs().cpu()
        r_mag = percentile_rank(mag)

        # Signal 2 — EMA gradient (log-smoothed for outlier robustness)
        if name in ema_scores:
            r_ema = log_percentile_rank(ema_scores[name])
        else:
            r_ema = torch.full_like(r_mag, 0.5)

        # Signal 3 — Fisher
        if name in fisher_scores:
            r_fisher = percentile_rank(fisher_scores[name])
        else:
            r_fisher = torch.full_like(r_mag, 0.5)

        # Signal 4 — movement
        if name in initial_weights:
            delta = (param.data.cpu() - initial_weights[name]).abs()
            r_move = percentile_rank(delta)
        else:
            r_move = torch.full_like(r_mag, 0.5)

        composite[name] = (
            w['magnitude'] * r_mag +
            w['ema'] * r_ema +
            w['fisher'] * r_fisher +
            w['movement'] * r_move
        )

    return composite


# =============================================================================
# 5. MASK CREATION
# =============================================================================

def create_masks(scores: dict, target_sparsity: float) -> tuple:
    """
    Global threshold pruning on composite scores.
    Lower score = less important = pruned.

    Returns: (masks_dict, achieved_sparsity)
    """
    # Concatenate all scores globally
    all_flat = torch.cat([s.flatten() for s in scores.values()])
    n_total = all_flat.numel()
    n_prune = max(1, min(int(n_total * target_sparsity), n_total - 1))

    threshold = torch.kthvalue(all_flat, n_prune).values.item()

    masks = {}
    total_pruned = 0
    for name, score in scores.items():
        mask = (score > threshold).float()
        masks[name] = mask
        total_pruned += (mask == 0).sum().item()

    achieved = total_pruned / n_total if n_total > 0 else 0.0
    return masks, achieved


def apply_masks(model: nn.Module, masks: dict, device: torch.device):
    """Zero out pruned weights in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name].to(device))


def enforce_mask_on_gradients(model: nn.Module, masks: dict, device: torch.device):
    """Zero gradient of pruned weights to prevent them from recovering."""
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.data.mul_(masks[name].to(device))


# =============================================================================
# 6. LTH REWIND
# =============================================================================

def lth_rewind(model: nn.Module, masks: dict,
               initial_weights: dict, device: torch.device):
    """
    Lottery Ticket Hypothesis rewind:
    w_surviving = w_init   (original initialisation, not current value)
    w_pruned    = 0
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks and name in initial_weights:
                mask = masks[name].to(device)
                orig = initial_weights[name].to(device)
                param.data.copy_(mask * orig)


# =============================================================================
# 7. FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# =============================================================================
# 8. LTH FINE-TUNING LOOP
# =============================================================================

def lth_finetune(
    model: nn.Module,
    masks: dict,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    sparsity_label: str = "",
) -> tuple:
    """
    Fine-tune a pruned model with LTH constraints.

    Strategy:
      - Cosine LR schedule with warmup
      - Focal loss with label smoothing
      - Gradient accumulation
      - Mask enforcement (gradients + weights) every accum step
      - SWA for the last `swa_start_fraction` of epochs
      - Best-AUROC checkpoint tracking with patience-based early stopping

    Returns:
      (best_state_dict, best_metrics_dict)
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None

    # Gradient checkpointing: recompute activations during backward instead of storing them — halves activation memory with ~15% compute cost.
    _enable_gradient_checkpointing(model)

    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )

    n_epochs = cfg['lth_epochs']
    accum = cfg['lth_grad_accum']
    total_steps = (len(train_loader) // accum) * n_epochs
    warmup_steps = int(total_steps * cfg['lth_warmup_fraction'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = FocalLoss(
        gamma=cfg['focal_gamma'],
        label_smoothing=cfg['label_smoothing'],
    )

    best_auroc = 0.0
    best_state = deepcopy(model.state_dict())
    best_metrics = {}
    patience_counter = 0

    swa_start_ep = max(0, int(n_epochs * cfg['swa_start_fraction']))
    swa_state = None
    swa_count = 0

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n   Fine-tuning {sparsity_label} | "
          f"{n_epochs} epochs | backbone_lr={cfg['lth_backbone_lr']:.1e} | "
          f"head_lr={cfg['lth_head_lr']:.1e} | swa_start=ep{swa_start_ep+1}")

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"   [{sparsity_label}] Ep {epoch+1}/{n_epochs}",
                    leave=False)

        for step_idx, batch in enumerate(pbar):
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs) / accum
                scaler.scale(loss).backward()
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs) / accum
                loss.backward()

            # Enforce mask on gradients immediately after backward
            enforce_mask_on_gradients(model, masks, device)

            loss_val = loss.item() * accum
            pred_labels = torch.argmax(out, 1)
            running_loss += loss_val
            correct += (pred_labels == labs).sum().item()
            total += labs.size(0)

            # Free GPU tensors that are no longer needed
            del out, loss, pred_labels, ri, rm, ai, am, labs

            is_accum_step = ((step_idx + 1) % accum == 0 or
                             (step_idx + 1) == len(train_loader))
            if is_accum_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['lth_max_grad_norm'])
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Re-enforce masks on weights after each optimizer step
                apply_masks(model, masks, device)

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                acc=f"{100*correct/max(total,1):.1f}%",
            )

        elapsed = time.time() - t0
        train_acc = 100 * correct / max(total, 1)
        avg_loss = running_loss / len(train_loader)

        # Free cache before validation (eval uses less memory but benefits from
        # releasing any fragmented training allocations)
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Validation
        val_m = evaluate_full(model, val_loader, device, use_amp,
                              desc=f"Val Ep{epoch+1}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"   [{sparsity_label}] Ep {epoch+1}: "
              f"Loss={avg_loss:.4f} | TrainAcc={train_acc:.2f}% | "
              f"ValAcc={val_m['accuracy']:.2f}% | AUROC={val_m['auroc']:.2f}% | "
              f"F1={val_m['f1']:.2f}% | {elapsed:.0f}s")

        # ── SWA: accumulate model averages (CPU tensors to save VRAM) ──────
        if epoch >= swa_start_ep:
            # Keep SWA state on CPU to avoid doubling GPU memory usage
            cpu_state = {k: v.cpu().float() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cpu_state
                swa_count = 1
            else:
                swa_count += 1
                for k in swa_state:
                    swa_state[k].add_(
                        (cpu_state[k] - swa_state[k]) / swa_count
                    )
            del cpu_state

        # ── Best tracking ───────────────────────────────────────────────
        if val_m['auroc'] > best_auroc:
            best_auroc = val_m['auroc']
            # Store best state on CPU to avoid holding two GPU copies
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = val_m
            patience_counter = 0
            print(f"   🎯 [{sparsity_label}] NEW BEST AUROC={best_auroc:.4f}% "
                  f"Acc={val_m['accuracy']:.4f}%")
        else:
            patience_counter += 1
            if patience_counter >= cfg['lth_patience']:
                print(f"   ⏹  [{sparsity_label}] Early stop at epoch {epoch+1}")
                break

    # ── Apply SWA if it improved things ─────────────────────────────────
    if swa_state is not None and swa_count > 1:
        print(f"   Applying SWA ({swa_count} checkpoints)...")
        # Cast SWA state (CPU fp32) back to original dtypes, move to GPU
        orig_state = model.state_dict()
        swa_cast = {}
        for k, v in swa_state.items():
            tgt_dtype = orig_state[k].dtype if k in orig_state else v.dtype
            swa_cast[k] = v.to(device=device, dtype=tgt_dtype)
        del swa_state  # free CPU copy
        model.load_state_dict(swa_cast)
        del swa_cast
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Re-enforce masks after SWA
        apply_masks(model, masks, device)
        swa_m = evaluate_full(model, val_loader, device, use_amp,
                              desc="SWA Eval")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   SWA AUROC={swa_m['auroc']:.4f}%  Best={best_auroc:.4f}%")
        if swa_m['auroc'] > best_auroc:
            best_auroc = swa_m['auroc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = swa_m
            print(f"SWA wins → using SWA weights")
        else:
            # Reload best (CPU tensors → GPU)
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()})
            apply_masks(model, masks, device)
            print(f"   ↩  Best checkpoint wins → reverting")
    elif best_state:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()})
        apply_masks(model, masks, device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_state, best_metrics


# =============================================================================
# 9. EMA WARMUP LOOP (shared)
# =============================================================================

def run_ema_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
    ema_tracker: EMAGradientTracker,
) -> EMAGradientTracker:
    """
    Run `ema_warmup_epochs` epochs of training purely for gradient statistics.
    The model weights are NOT saved (we reload initial weights after this).
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    _enable_gradient_checkpointing(model)
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )
    criterion = FocalLoss(
        gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])

    n_ep = cfg['ema_warmup_epochs']
    print(f"\n   ── EMA Warmup ({n_ep} epochs) ──")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for ep in range(n_ep):
        model.train()
        pbar = tqdm(train_loader,
                    desc=f"   EMA Warmup {ep+1}/{n_ep}", leave=False)
        for batch in pbar:
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs)
                loss.backward()

            # Update EMA BEFORE optimizer step
            ema_tracker.update(model)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg['lth_max_grad_norm'])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            del out, loss, ri, rm, ai, am, labs

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        m = evaluate_full(model, val_loader, device, use_amp,
                          desc=f"EMA Warmup Ep{ep+1}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   EMA Warmup Ep {ep+1}: "
              f"AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%  "
              f"Steps={ema_tracker.step_count}")

    print(f"EMA tracked over {ema_tracker.step_count} gradient steps")
    return ema_tracker


# =============================================================================
# 10. MOVEMENT WARMUP LOOP (short training pass)
# =============================================================================

def run_movement_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    cfg: dict,
) -> None:
    """
    Run `movement_warmup_epochs` of fine-tuning so the model's weights
    move from their initial positions, making movement scores meaningful.
    Modifies model in-place.
    """
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    _enable_gradient_checkpointing(model)
    optimizer = optim.AdamW(
        model.get_param_groups(cfg['lth_backbone_lr'], cfg['lth_head_lr']),
        weight_decay=cfg['lth_weight_decay'],
    )
    criterion = FocalLoss(
        gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])

    n_ep = cfg['movement_warmup_epochs']
    print(f"\n   ── Movement Warmup ({n_ep} epochs) ──")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for ep in range(n_ep):
        model.train()
        pbar = tqdm(train_loader,
                    desc=f"   Move Warmup {ep+1}/{n_ep}", leave=False)
        for batch in pbar:
            ri = batch['ref_ids'].to(device)
            rm = batch['ref_mask'].to(device)
            ai = batch['alt_ids'].to(device)
            am = batch['alt_mask'].to(device)
            labs = batch['labels'].to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast('cuda'):
                    out = model(ri, rm, ai, am)
                    loss = criterion(out, labs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                out = model(ri, rm, ai, am)
                loss = criterion(out, labs)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg['lth_max_grad_norm'])
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            del out, loss, ri, rm, ai, am, labs

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        m = evaluate_full(model, val_loader, device, use_amp,
                          desc=f"Move Warmup Ep{ep+1}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   Move Warmup Ep {ep+1}: "
              f"AUROC={m['auroc']:.2f}%  Acc={m['accuracy']:.2f}%")

    print(f"Movement warmup complete")
