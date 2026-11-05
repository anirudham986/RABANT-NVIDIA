# =====================================================================
# data/dataset.py — DualSeqDataset
#
# Dual-sequence variant dataset: for each variant, extracts the
# reference and alternate genome sequences from hg38, centred on the
# variant position. This is the core innovation of the NTv2 approach —
# the model sees BOTH the reference and alternate allele in their
# genomic context and learns the difference.
# =====================================================================

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DualSeqDataset(Dataset):
    """
    Dual-sequence variant effect dataset.

    For each variant (CHROM, POS, REF, ALT):
        1. Extracts `seq_length` bp from hg38 centred on the variant position
        2. Creates REF sequence (with reference allele at centre)
        3. Creates ALT sequence (with alternate allele at centre)
        4. Tokenizes both sequences

    The model receives both sequences and learns to classify variants
    by comparing their embeddings: [emb_ref; emb_alt; emb_ref - emb_alt]
    """

    VALID_BASES = set('ACGT')

    def __init__(self, df, genome, tokenizer, has_chr: bool,
                 seq_len: int = 1000, max_tokens: int = 256,
                 seed: int = 42):
        """
        Args:
            df: DataFrame with columns [CHROM, POS, REF, ALT, LABEL]
            genome: pyfaidx.Fasta object for hg38
            tokenizer: HF tokenizer for NT v2
            has_chr: Whether genome contigs use 'chr' prefix
            seq_len: Base-pair context window around variant
            max_tokens: Maximum tokens after tokenization
            seed: Random seed for N-base replacement
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.ref_seqs = []
        self.alt_seqs = []
        self.labels = []

        rng = np.random.RandomState(seed)
        skipped = 0

        print(f"   Extracting {len(df):,} variant pairs ({seq_len}bp context)...")
        for idx in tqdm(range(len(df)), desc="   Sequences"):
            row = df.iloc[idx]
            chrom = str(row['CHROM']).strip()
            pos = int(row['POS'])
            ref = str(row['REF']).upper().strip()
            alt = str(row['ALT']).upper().strip()
            label = int(row['LABEL'])

            # Normalize chromosome naming
            if has_chr and not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            elif not has_chr and chrom.startswith('chr'):
                chrom = chrom[3:]

            # Skip unmapped chromosomes
            if chrom not in genome.keys():
                skipped += 1
                continue

            # Handle multi-allelic — take first base (SNV approximation)
            if len(ref) != 1:
                ref = ref[0] if ref else 'A'
            if len(alt) != 1:
                alt = alt[0] if alt else 'A'

            # Skip non-standard bases
            if ref not in self.VALID_BASES or alt not in self.VALID_BASES:
                skipped += 1
                continue

            # Extract genomic context
            half = seq_len // 2
            start = pos - 1 - half
            end = pos - 1 + half

            if start < 0 or end > len(genome[chrom]):
                skipped += 1
                continue

            seq = genome[chrom][start:end].upper()
            if len(seq) != seq_len:
                skipped += 1
                continue

            # Handle N bases (max 5% tolerance)
            n_bad = sum(1 for b in seq if b not in self.VALID_BASES)
            if n_bad > seq_len * 0.05:
                skipped += 1
                continue
            if n_bad > 0:
                seq_list = list(seq)
                for i, b in enumerate(seq_list):
                    if b not in self.VALID_BASES:
                        seq_list[i] = rng.choice(list(self.VALID_BASES))
                seq = ''.join(seq_list)

            # Build REF and ALT sequences with allele at centre
            centre = half
            ref_seq = list(seq)
            ref_seq[centre] = ref
            ref_seq = ''.join(ref_seq)

            alt_seq = list(seq)
            alt_seq[centre] = alt
            alt_seq = ''.join(alt_seq)

            self.ref_seqs.append(ref_seq)
            self.alt_seqs.append(alt_seq)
            self.labels.append(label)

        # Summary statistics
        np_labels = np.array(self.labels)
        print(f"{len(self.labels):,} variant pairs extracted")
        if skipped:
            print(f"Skipped {skipped:,} (unmapped/invalid)")
        print(f"      Pathogenic: {np_labels.sum():,} | "
              f"Benign: {(1 - np_labels).sum():,}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ref_enc = self.tokenizer(
            self.ref_seqs[idx],
            max_length=self.max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        alt_enc = self.tokenizer(
            self.alt_seqs[idx],
            max_length=self.max_tokens,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'ref_ids': ref_enc['input_ids'].squeeze(0),
            'ref_mask': ref_enc['attention_mask'].squeeze(0),
            'alt_ids': alt_enc['input_ids'].squeeze(0),
            'alt_mask': alt_enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }
