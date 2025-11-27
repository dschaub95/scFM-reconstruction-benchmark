"""PBMC68k dataset download utilities."""

from pathlib import Path

import scanpy as sc
from anndata import AnnData


def download_dataset(data_dir: Path) -> tuple[AnnData, Path]:
    """
    Download the PBMC68k dataset from Scanpy to the specified directory.

    Downloads:
    - pbmc68k to data/pbmc68k/raw/pbmc68k_raw.h5ad

    Parameters
    ----------
    data_dir
        Base data directory (e.g., Path("data"))

    Returns
    -------
    Tuple of (pbmc68k AnnData object, path to pbmc68k downloaded file)
    """
    # Create directory for pbmc68k dataset
    pbmc68k_raw_dir = data_dir / "pbmc68k" / "raw"
    pbmc68k_raw_dir.mkdir(parents=True, exist_ok=True)

    # Save original dataset directory
    original_dataset_dir = sc.settings.datasetdir

    # Download pbmc68k dataset
    sc.settings.datasetdir = pbmc68k_raw_dir
    try:
        adata_68k = sc.datasets.pbmc68k_reduced()
        # Save the file explicitly (pbmc68k_reduced may not auto-save)
        downloaded_path_68k = pbmc68k_raw_dir / "pbmc68k_reduced.h5ad"
        adata_68k.write(downloaded_path_68k)
        print(f"Downloaded pbmc68k dataset to {downloaded_path_68k}")
    finally:
        sc.settings.datasetdir = original_dataset_dir

    return adata_68k, downloaded_path_68k

