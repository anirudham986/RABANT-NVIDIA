import os
from copy import deepcopy


BASE_CFG = {
    # Model
    'model_name': 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species',
    'seq_length': 1000,               # bp context window around variant

    # Data
    'max_per_class': 50_000,           # Cap per class for memory
    'val_fold': 0,                     # Validation fold for kfold strategy

    # Training — full fine-tuning with differential LR
    'epochs': 20,
    'batch_size': 32,                  # Per-GPU batch
    'grad_accum_steps': 4,             # Effective batch = 128
    'backbone_lr': 5e-6,              # Very low — preserve pre-training
    'head_lr': 5e-4,                  # Higher — learn task fast
    'weight_decay': 0.01,
    'warmup_fraction': 0.15,
    'label_smoothing': 0.05,
    'focal_gamma': 1.5,               # Moderate focal loss
    'max_grad_norm': 1.0,
    'dropout': 0.2,
    'num_layers_to_unfreeze': 22,      # ALL layers (full fine-tuning)
    'patience': 4,                     # Early stopping patience (epochs)
    'seed': 42,

    # Logging
    'log_every_n_steps': 50,           # Print detailed batch stats every N steps
    'verbose': True,                   # Enable detailed per-batch logging

    # Performance
    'cudnn_benchmark': True,           # Faster on fixed-size inputs
    'num_workers': 4,                  # DataLoader workers

    # Paths (relative — works on any machine)
    'save_dir': os.path.join('.', 'output', 'ntv2_trained'),
}

# Human-readable dataset choices for the interactive prompt
DATASET_CHOICES = {
    '1': 'clinvar',
    '2': 'dbsnp',
    '3': 'cbioportal',
    '4': 'consolidated',
    '5': 'consolidated_full',
    'clinvar':            'clinvar',
    'dbsnp':              'dbsnp',
    'cbioportal':         'cbioportal',
    'consolidated':       'consolidated',
    'consolidated_full':  'consolidated_full',
}


DATASET_OVERRIDES = {
    'clinvar': {
        'max_per_class': 37_498,
    },
    'dbsnp': {
        'max_per_class': 31_398,
    },
    'cbioportal': {
        'max_per_class': 50_000,
    },
    'consolidated': {
        'max_per_class': 50_000,     # Per-class cap (handled by build script)
        'val_fold': 0,               # Default validation fold (override with --fold)
        'n_folds': 5,
    },
    'consolidated_full': {
        # Full 100k training + 25k unseen holdout test
        # Used AFTER K-fold validation to produce the final compression-ready model
        'max_per_class': 50_000,
    },
}


def get_config(dataset_name: str, val_fold: int = None) -> dict:
    """
    Build the full configuration for a given dataset by merging
    the base config with dataset-specific overrides.

    Args:
        dataset_name: One of 'clinvar', 'dbsnp', 'cbioportal', 'consolidated',
                      'consolidated_full'
        val_fold:     Override validation fold (0–4) for 'consolidated' only

    Returns:
        Complete configuration dictionary.
    """
    if dataset_name not in DATASET_OVERRIDES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Must be one of: {list(DATASET_OVERRIDES.keys())}"
        )

    cfg = deepcopy(BASE_CFG)
    cfg.update(DATASET_OVERRIDES[dataset_name])
    cfg['dataset_name'] = dataset_name
    cfg['save_dir'] = os.path.join('.', 'output', f'ntv2_{dataset_name}_trained')

    # Apply val_fold override
    if val_fold is not None:
        cfg['val_fold'] = val_fold

    return cfg
