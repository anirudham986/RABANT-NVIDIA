# DNAi: Variant Effect Prediction and Model Compression

This repository contains a GPU-oriented pipeline for classifying genomic variants as **pathogenic** or **benign** using the [Nucleotide Transformer v2 100M multi-species model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species). It includes dataset loading and leakage checks, model training and evaluation, and experimental Lottery Ticket Hypothesis (LTH) pruning/compression workflows.

> **Research software:** This code is intended for experimentation and research. Model predictions are not a substitute for clinical interpretation or professional medical advice.

## Contents

- [What it does](#what-it-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training](#training)
- [Data](#data)
- [Compression workflow](#compression-workflow)
- [Training outputs](#training-outputs)
- [Notes and troubleshooting](#notes-and-troubleshooting)

## What it does

The main training entry point, `train.py`, builds a dual-sequence classifier around the pretrained Nucleotide Transformer v2 backbone. It extracts reference and alternate allele sequence context from the GRCh38/hg38 reference genome, fine-tunes the model with focal loss, evaluates classification metrics, and saves model weights and run metadata.

Available training data modes are:

| Mode | Description |
| --- | --- |
| `clinvar` | ClinVar pathogenic and benign variants |
| `dbsnp` | dbSNP pathogenic and benign variants |
| `cbioportal` | cBioPortal pathogenic variants combined with gnomAD benign variants |
| `consolidated` | Consolidated data, split using one of five `FOLD_ID` folds |
| `consolidated_full` | Full consolidated training set evaluated against a separate unseen holdout |

The consolidated modes run leakage checks before training. `consolidated` uses the selected fold for validation; `consolidated_full` trains on the full training CSV and evaluates on the 25k unseen holdout CSV.

## Repository layout

```text
.
├── compression/                 # LTH pruning scorers and shared compression utilities
├── config/                      # Training hyperparameters and dataset configuration
├── crct dataset/                # Bundled CSV datasets
├── data/                         # Dataset registry, loaders, builders, leakage audit
├── engine/                       # Training and evaluation loops
├── model/                        # Classifier and loss definitions
├── utils/                        # Genome loading, device selection, random seeds
├── compression_pipeline.py      # Compression experiment entry point
└── train.py                      # Main training entry point
```

## Requirements

- Python 3.9 or later
- NVIDIA GPU with CUDA recommended for practical training; the code is designed for NVIDIA GPU environments
- A compatible PyTorch build for your CUDA version (install PyTorch according to the [official instructions](https://pytorch.org/get-started/locally/))
- GRCh38/hg38 reference genome FASTA for sequence extraction
- Internet access on first use to download model/tokenizer files from Hugging Face (and, if no local genome is found, the genome FASTA from UCSC)

Python packages used by the training and data pipeline include `transformers`, `pyfaidx`, `pandas`, `numpy`, `scikit-learn`, and `tqdm`. Install PyTorch separately for the CUDA runtime available on your machine, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install transformers pyfaidx pandas numpy scikit-learn tqdm
```

The compression entry point pins `transformers==4.40.2` and checks/installs some packages at runtime. For reproducible experiments, use a dedicated environment and explicitly install compatible dependency versions rather than relying on runtime installation.

## Setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/anirudham986/dnai-gpu1.git
cd dnai-gpu1
```

Point the project to a local GRCh38/hg38 FASTA to avoid searching for or downloading it automatically:

```bash
export HG38_PATH=/path/to/hg38.fa
```

The genome loader also searches common local paths and downloads hg38 from UCSC if it cannot find a suitable FASTA. The reference genome is large; ensure adequate disk space and network access if downloading it.

The dataset loader searches the repository's `crct dataset/` folder and common Kaggle locations. The repository includes CSV files for the registered datasets; keep the required files together in a searchable data directory.

## Training

Run interactively to select a dataset:

```bash
python train.py
```

Or choose a dataset from the command line:

```bash
python train.py --dataset clinvar
python train.py --dataset dbsnp
python train.py --dataset cbioportal
python train.py --dataset consolidated --fold 0
python train.py --dataset consolidated_full
```

`consolidated` supports folds `0` through `4`. Useful optional overrides include:

```bash
python train.py --dataset consolidated --fold 2 --epochs 10 --batch_size 16
python train.py --dataset clinvar --save_dir ./output/clinvar-run
python train.py --dataset clinvar --resume ./output/checkpoint.pth
```

The default training configuration is in `config/hyperparams.py`. It specifies the pretrained model, 1,000 bp sequence context, batch size, gradient accumulation, learning rates, early stopping, and output location. The default output directory is under `output/`.

## Data

The dataset registry is defined in `data/registry.py`; expected CSV filenames and label columns are configured there. The bundled `crct dataset/` directory contains the source and prepared files, including:

- `01_clinvar_75k_P+B.csv`
- `02_gnomad_55k_B.csv`
- `03_dbsnp_62k_P+B.csv`
- `04_cbioportal_63k_P.csv`
- `05_consolidated_balanced.csv`
- `06_holdout_25k_unseen.csv`
- Additional unseen test CSVs for individual data sources

If you need to regenerate prepared datasets, inspect the builder scripts in `data/` before running them. For example:

```bash
python data/build_consolidated.py
python data/build_holdout.py
```

The consolidated loader expects a `FOLD_ID` column for k-fold validation. The full-training mode expects both the consolidated training CSV and holdout CSV. A mandatory leakage audit is performed before the training loop; training aborts if the audit detects a problem.

## Compression workflow

`compression_pipeline.py` runs LTH-based pruning experiments with magnitude, EMA-gradient, Fisher, movement, and hybrid scoring at sparsity levels of 10%, 20%, 30%, 50%, and 60%. It compares compressed models with a dense baseline and writes results beneath `output/lth_compressed/`.

Before running it, review the configuration at the top of `compression_pipeline.py` and `compression/shared.py`. In particular:

1. Set `DENSE_MODEL_PATH` to the trained weights file to compress; it is a placeholder in the checked-in script.
2. Confirm that the training and validation CSVs expected by the compression data-loader configuration are available.
3. Confirm that the saved result JSON files expected for the magnitude, EMA, and Fisher scorers already exist in their configured output directories. The current pipeline loads these results rather than rerunning those three scorers.
4. Check the GPU memory, runtime, and dependency requirements before launching the full experiment.

Then run:

```bash
python compression_pipeline.py
```

The compression pipeline is an experiment script and may require adapting its paths and saved-result prerequisites for your environment; it is not configured as a plug-and-play command out of the box.

## Training outputs

Training saves artifacts to the selected `--save_dir` (by default, `output/ntv2_<dataset>_trained/`):

- `ntv2_<dataset>_final.pth` (or a fold-specific filename): model `state_dict` weights
- `training_history.json`: per-epoch training history
- `final_metrics.json`: final evaluation metrics
- `training_config.json`: run configuration and model architecture details

The compression pipeline writes scorer checkpoints, per-scorer results, and a `pipeline_summary.json` under `output/lth_compressed/`.

## Notes and troubleshooting

- **GPU memory:** If you run out of memory, try a smaller `--batch_size`; gradient accumulation is configured separately in `config/hyperparams.py`.
- **Missing data:** Confirm the filenames match those in `data/registry.py` and that the CSVs are in `crct dataset/` or another directory searched by the loader.
- **Missing genome:** Set `HG38_PATH` to the FASTA file. The FASTA must be compatible with GRCh38 coordinates used by the data.
- **Model download/access:** The tokenizer and pretrained backbone are fetched through Hugging Face Transformers; ensure network access and enough cache/storage space.
- **Reproducibility:** Training configuration and random seed are centralized in `config/hyperparams.py`; record software versions, hardware, and dataset provenance with experiment results.

## Acknowledgments

The model backbone is [`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species). Dataset sources include ClinVar, dbSNP, cBioPortal, and gnomAD. Refer to their respective terms, licenses, and citation requirements when using or redistributing data.
