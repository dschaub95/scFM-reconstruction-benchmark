"""Main entry point for simple-baseline reconstruction benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import anndata as ad
from scipy.stats import nbinom

from reconstruction_benchmark.config import ModelConfig
from reconstruction_benchmark.utils import get_project_root

# Define execution stages in order
STAGES = ["install", "load_data", "inference", "save"]


def load_masked_data(config: ModelConfig) -> ad.AnnData:
    """
    Load masked data and identify masked positions.

    Parameters
    ----------
    config
        Model configuration

    Returns
    -------
    AnnData object with masked data and mask information
    """
    masked_path = config.masked_data_path
    if not masked_path.exists():
        raise FileNotFoundError(f"Masked data not found: {masked_path}")

    adata = ad.read_h5ad(masked_path)

    # Convert sparse to dense if needed
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = X.copy()

    # Store mask information (where values are -1)
    mask_indices = X == -1

    # Store original mask for later reconstruction
    adata.uns["mask_indices"] = mask_indices

    # Set masked values to 0 temporarily
    X[mask_indices] = 0
    adata.X = X

    return adata


def sample_zinb(
    shape: tuple[int, int],
    n: float = 5.0,
    p: float = 0.5,
    pi: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample from a Zero-Inflated Negative Binomial distribution.

    Parameters
    ----------
    shape
        Shape of the output array (n_cells, n_genes)
    n
        Number of successes parameter for negative binomial
    p
        Probability of success parameter for negative binomial
    pi
        Zero inflation probability (probability of observing a zero)
    seed
        Random seed for reproducibility

    Returns
    -------
    Array of samples from ZINB distribution
    """
    rng = np.random.default_rng(seed)

    # Sample from negative binomial
    nb_samples = nbinom.rvs(n=n, p=p, size=shape, random_state=rng)

    # Apply zero inflation: set pi fraction of values to 0
    zero_mask = rng.random(shape) < pi
    nb_samples[zero_mask] = 0

    return nb_samples.astype(np.float32)


def run_inference(
    adata: ad.AnnData,
    seed: int = 42,
    n: float = 5.0,
    p: float = 0.5,
    pi: float = 0.2,
) -> ad.AnnData:
    """
    Run simple baseline inference: generate random ZINB predictions for masked values.

    Parameters
    ----------
    adata
        AnnData object with masked data
    seed
        Random seed for reproducibility
    n
        Number of successes parameter for negative binomial
    p
        Probability of success parameter for negative binomial
    pi
        Zero inflation probability
    Returns
    -------
    AnnData object with reconstructed values
    """
    # Get mask indices
    mask_indices = adata.uns.get("mask_indices", np.zeros_like(adata.X, dtype=bool))

    # Get expression matrix
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    n_cells, n_genes = X.shape
    print(f"Data shape: {n_cells} cells × {n_genes} genes")
    print(
        f"Masked values: {mask_indices.sum()} ({100 * mask_indices.sum() / mask_indices.size:.2f}%)"
    )

    # Generate ZINB predictions only for masked positions
    print("Generating ZINB predictions for masked values...")
    predictions = sample_zinb(
        shape=(n_cells, n_genes),
        n=n,
        p=p,
        pi=pi,
        seed=seed,
    )

    # Create reconstruction: replace masked values with predictions
    reconstruction = adata.copy()
    X_recon = X.copy()
    X_recon[mask_indices] = predictions[mask_indices]

    reconstruction.X = X_recon

    # Remove temporary mask information
    if "mask_indices" in reconstruction.uns:
        del reconstruction.uns["mask_indices"]

    print("Reconstruction complete")
    return reconstruction


def save_reconstruction(
    reconstruction: ad.AnnData,
    config: ModelConfig,
) -> None:
    """
    Save reconstruction results and metadata.

    Parameters
    ----------
    reconstruction
        AnnData object with reconstructed values
    config
        Model configuration

    Raises
    ------
    OSError
        If output directory cannot be created or files cannot be written
    """
    # Create output directory if it doesn't exist
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save reconstruction
    output_path = config.output_dir / "reconstruction.h5ad"
    try:
        reconstruction.write(output_path)
        print(f"Saved reconstruction to {output_path}")
    except Exception as e:
        raise OSError(f"Failed to save reconstruction to {output_path}: {e}") from e

    # Save metadata
    metadata = {
        "model": config.model_name,
        "dataset": config.dataset_name,
        "n_cells": reconstruction.n_obs,
        "n_genes": reconstruction.n_vars,
    }

    metadata_path = config.output_dir / "metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    except Exception as e:
        raise OSError(f"Failed to save metadata to {metadata_path}: {e}") from e


def main():
    """Main entry point for simple-baseline CLI."""
    parser = argparse.ArgumentParser(
        description="Run simple-baseline reconstruction benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pbmc3k",
        help="Dataset name (default: pbmc3k)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data (default: data)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--mask-percentage",
        type=float,
        default=0.15,
        help="Mask percentage used for masked data (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for masking and prediction (default: 42)",
    )
    parser.add_argument(
        "--n",
        type=float,
        default=5.0,
        help="Negative binomial n parameter (default: 5.0)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Negative binomial p parameter (default: 0.5)",
    )
    parser.add_argument(
        "--pi",
        type=float,
        default=0.2,
        help="Zero inflation probability (default: 0.2)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip environment installation step",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode flag (no-op, for compatibility)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=STAGES,
        default=None,
        help=f"Stop after completing this stage. Available stages: {', '.join(STAGES)}",
    )

    args = parser.parse_args()

    # Resolve data_dir and results_dir relative to project root if they are relative paths
    project_root = get_project_root(Path(__file__).parent)

    # If data_dir is relative, resolve it relative to project root
    if not args.data_dir.is_absolute():
        data_dir = (project_root / args.data_dir).resolve()
    else:
        data_dir = args.data_dir

    # If results_dir is relative, resolve it relative to project root
    if not args.results_dir.is_absolute():
        results_dir = (project_root / args.results_dir).resolve()
    else:
        results_dir = args.results_dir

    config = ModelConfig(
        model_name="simple-baseline",
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        data_dir=data_dir,
        results_dir=results_dir,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Install environment (if needed)
    if not args.skip_install:
        from install import install_environment

        print("Checking/installing environment...")
        install_environment()

    if args.stage == "install":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 2: Load masked data
    print(f"Loading masked data from {config.masked_data_path}...")
    masked_data = load_masked_data(config)

    if args.stage == "load_data":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 3: Run inference
    print("Running simple-baseline reconstruction...")
    reconstruction = run_inference(
        adata=masked_data,
        seed=args.seed,
        n=args.n,
        p=args.p,
        pi=args.pi,
    )

    if args.stage == "inference":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 4: Save reconstruction
    save_reconstruction(
        reconstruction=reconstruction,
        config=config,
    )

    if args.stage == "save":
        print(f"Stopping after stage: {args.stage}")
        return


if __name__ == "__main__":
    main()
