import os
import glob
from pyfaidx import Fasta


def load_hg38() -> tuple:
    """
    Load the hg38 reference genome.

    Search order:
        1. HG38_PATH environment variable (user-defined)
        2. Common HPC / workstation paths
        3. Kaggle paths (backward compatibility)
        4. Download from UCSC if nothing found

    Returns:
        (genome, has_chr) — pyfaidx.Fasta object and whether contigs
                            use 'chr' prefix (True for hg38)
    """
    fa_path = _find_hg38()
    genome = Fasta(fa_path, as_raw=True, sequence_always_upper=True)
    has_chr = 'chr1' in set(genome.keys())
    n_contigs = len(genome.keys())
    fa_size_gb = os.path.getsize(fa_path) / 1e9
    print(f"hg38 loaded — {n_contigs} contigs ({fa_size_gb:.1f} GB)")
    print(f"Path: {fa_path}")
    return genome, has_chr


def _find_hg38() -> str:
    """Locate or download hg38.fa."""

    # 1. Environment variable (highest priority)
    env_path = os.environ.get('HG38_PATH')
    if env_path and os.path.exists(env_path):
        print(f"HG38_PATH env: {env_path}")
        return env_path

    # 2. Common file names to search for
    fa_names = ['hg38.fa', 'hg38.fasta', 'GRCh38.fa', 'GRCh38.fasta']

    # 3. Search paths — ordered by priority
    search_dirs = [
        # Current working directory and nearby
        os.getcwd(),
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), 'genome'),
        os.path.join(os.getcwd(), 'reference'),
        # Home directory
        os.path.expanduser('~'),
        os.path.expanduser('~/data'),
        os.path.expanduser('~/genome'),
        os.path.expanduser('~/reference'),
        # Common HPC / NVIDIA workstation paths
        '/data',
        '/data/reference',
        '/data/genome',
        '/workspace',
        '/workspace/data',
        '/workspace/genome',
        '/opt/genome',
        '/opt/data',
        '/scratch',
        '/scratch/genome',
        # Kaggle (backward compatibility)
        '/kaggle/working',
        '/kaggle/input',
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in fa_names:
            path = os.path.join(d, name)
            if os.path.exists(path) and os.path.getsize(path) > 1e8:
                print(f"Found: {path}")
                return path

    # 4. Recursive search in Kaggle input (backward compatibility)
    for pattern in ["/kaggle/input/**/hg38.fa", "/kaggle/input/**/*.fa"]:
        for match in glob.glob(pattern, recursive=True):
            if os.path.getsize(match) > 1e8:  # > 100MB = likely real genome
                print(f"Found: {match}")
                return match

    # 5. Download from UCSC as last resort
    print("hg38 not found locally — downloading from UCSC...")
    print("TIP: Set HG38_PATH=/path/to/hg38.fa to skip this step")
    return _download_hg38()


def _download_hg38() -> str:
    """
    Download hg38.fa.gz from UCSC and decompress.

    Tries wget first (best for large files — retries, resume support),
    then curl, then falls back to Python urllib as last resort.
    """
    import subprocess
    import shutil

    # Use current working directory for download
    download_dir = os.path.join(os.getcwd(), 'genome')
    os.makedirs(download_dir, exist_ok=True)

    gz_path = os.path.join(download_dir, 'hg38.fa.gz')
    fa_path = os.path.join(download_dir, 'hg38.fa')
    url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

    # --- Try wget (best for HPC — has retry + resume) ---
    if shutil.which('wget'):
        print("   Using wget (with retry + resume support)...")
        ret = subprocess.run(
            ['wget', '-c', '--tries=5', '--timeout=60',
             '-O', gz_path, url],
            check=False,
        )
        if ret.returncode == 0 and os.path.exists(gz_path):
            return _decompress_and_return(gz_path, fa_path)

    # --- Try curl (common fallback) ---
    if shutil.which('curl'):
        print("   Using curl...")
        ret = subprocess.run(
            ['curl', '-L', '-C', '-', '--retry', '5',
             '-o', gz_path, url],
            check=False,
        )
        if ret.returncode == 0 and os.path.exists(gz_path):
            return _decompress_and_return(gz_path, fa_path)

    # --- Last resort: Python urllib ---
    print("   Using Python urllib (no resume support)...")
    import urllib.request

    def _progress(count, block_size, total_size):
        if count % 500 == 0:
            mb = count * block_size / 1e6
            total_mb = total_size / 1e6 if total_size > 0 else 0
            pct = (mb / total_mb * 100) if total_mb > 0 else 0
            print(f"\r     {mb:.0f}/{total_mb:.0f} MB ({pct:.0f}%)...",
                  end="", flush=True)

    urllib.request.urlretrieve(url, gz_path, reporthook=_progress)
    print()
    return _decompress_and_return(gz_path, fa_path)


def _decompress_and_return(gz_path: str, fa_path: str) -> str:
    """Decompress .gz and return the .fa path."""
    import gzip
    import shutil

    print("   Decompressing (this may take a few minutes)...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(fa_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print(f"Saved: {fa_path}")
    print(f"Set HG38_PATH={fa_path} to avoid re-downloading")
    return fa_path
