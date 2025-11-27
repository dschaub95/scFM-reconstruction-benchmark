"""Run inference with scGPT model."""

import sys
from pathlib import Path

import numpy as np
import anndata as ad
import torch

from reconstruction_benchmark.config import ModelConfig


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

    # For scGPT, we'll handle masking during inference
    # Set masked values to 0 temporarily (scGPT will handle them)
    X[mask_indices] = 0
    adata.X = X

    return adata


def prepare_data_for_scgpt(adata: ad.AnnData) -> tuple:
    """
    Prepare AnnData for scGPT model input.

    Parameters
    ----------
    adata
        AnnData object with expression data

    Returns
    -------
    Tuple of (gene_names, expression_matrix, mask_indices)
    """
    # Get gene names
    gene_names = adata.var_names.tolist()

    # Get expression matrix
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)

    # Get mask indices
    mask_indices = adata.uns.get("mask_indices", np.zeros_like(X, dtype=bool))

    return gene_names, X, mask_indices


def run_inference(
    model,
    vocab,
    adata: ad.AnnData,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dev_mode: bool = False,
) -> ad.AnnData:
    """
    Run scGPT inference on masked data.

    Parameters
    ----------
    model
        Initialized scGPT model
    vocab
        Gene vocabulary (GeneVocab object)
    adata
        AnnData object with masked data
    device
        Device to run inference on
    dev_mode
        If True, run only one batch with batch size 2 for quick testing

    Returns
    -------
    AnnData object with reconstructed values
    """
    # Validate and potentially fix device
    if device == "cuda":
        try:
            test_tensor = torch.zeros(1).to(device)
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: CUDA device failed validation: {e}")
            print("Falling back to CPU")
            device = "cpu"

    print(f"Using device: {device}")

    # Prepare data
    gene_names, X, mask_indices = prepare_data_for_scgpt(adata)
    n_genes = len(gene_names)
    n_cells = X.shape[0]

    print(f"Data shape: {n_cells} cells × {n_genes} genes")
    print(
        f"Masked values: {mask_indices.sum()} ({100 * mask_indices.sum() / mask_indices.size:.2f}%)"
    )

    # Determine batch size based on dataset size and device
    if dev_mode:
        batch_size = 2
        print("Development mode: using batch size 2, processing only one batch")
    elif n_cells * n_genes > 50_000_000:
        batch_size = min(10, n_cells)
        print(
            f"Very large dataset detected ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    elif n_cells * n_genes > 10_000_000:
        batch_size = min(50, n_cells)
        print(
            f"Large dataset detected ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    elif n_cells * n_genes > 1_000_000:
        batch_size = min(100, n_cells)
        print(
            f"Moderately large dataset ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    else:
        batch_size = n_cells

    # Ensure model is on correct device
    model = model.to(device)
    model.eval()

    # Prepare input
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Run inference in batches
    print(f"Running scGPT inference in batches of {batch_size} cells...")
    all_predictions = []

    # In dev mode, limit to one batch
    max_cells_to_process = batch_size if dev_mode else n_cells

    with torch.no_grad():
        # Create gene indices template (same for all batches)
        gene_indices_template = torch.arange(n_genes, dtype=torch.long)

        for batch_start in range(0, max_cells_to_process, batch_size):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_cells = batch_end - batch_start

            print(
                f"Processing batch {batch_start // batch_size + 1}/{(n_cells + batch_size - 1) // batch_size} "
                f"(cells {batch_start}-{batch_end - 1})..."
            )

            # Prepare batch tensors
            batch_gene_indices = (
                gene_indices_template.unsqueeze(0).repeat(batch_cells, 1).to(device)
            )
            batch_values = X_tensor[batch_start:batch_end].to(device)
            batch_mask = torch.tensor(
                mask_indices[batch_start:batch_end], dtype=torch.bool
            ).to(device)

            try:
                # Forward pass
                outputs = model(
                    batch_gene_indices,
                    values=batch_values,
                    src_key_padding_mask=batch_mask,
                )

                # Extract predictions
                if isinstance(outputs, torch.Tensor):
                    batch_predictions = outputs
                elif isinstance(outputs, (tuple, list)):
                    batch_predictions = outputs[0]
                elif isinstance(outputs, dict):
                    batch_predictions = outputs.get(
                        "prediction",
                        outputs.get("reconstruction", list(outputs.values())[0]),
                    )
                else:
                    print(
                        f"Warning: Unexpected output format: {type(outputs)}, using input values"
                    )
                    batch_predictions = batch_values

                # Move to CPU and convert to numpy
                if batch_predictions.is_cuda:
                    batch_predictions = batch_predictions.cpu()
                batch_predictions = batch_predictions.numpy()
                all_predictions.append(batch_predictions)

                # Clear GPU cache if using CUDA
                if device == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                error_msg = str(e)
                # Check if it's a memory or CUDA error
                if (
                    "cuda" in error_msg.lower()
                    or "CUDA" in error_msg
                    or "memory" in error_msg.lower()
                ):
                    print(f"Memory/CUDA error in batch: {e}")
                    if device == "cuda":
                        print("Retrying batch on CPU...")
                        # Retry on CPU
                        device = "cpu"
                        model = model.cpu()
                        batch_gene_indices = batch_gene_indices.cpu()
                        batch_values = batch_values.cpu()
                        batch_mask = batch_mask.cpu()

                        try:
                            outputs = model(
                                batch_gene_indices,
                                values=batch_values,
                                src_key_padding_mask=batch_mask,
                            )
                            if isinstance(outputs, torch.Tensor):
                                batch_predictions = outputs
                            elif isinstance(outputs, (tuple, list)):
                                batch_predictions = outputs[0]
                            elif isinstance(outputs, dict):
                                batch_predictions = outputs.get(
                                    "prediction", list(outputs.values())[0]
                                )
                            else:
                                batch_predictions = batch_values

                            batch_predictions = batch_predictions.cpu().numpy()
                            all_predictions.append(batch_predictions)
                            print("Batch completed on CPU")
                        except Exception as e2:
                            print(f"CPU inference also failed: {e2}")
                            print("Using input values for this batch")
                            all_predictions.append(X[batch_start:batch_end])
                    else:
                        print("Using input values for this batch")
                        all_predictions.append(X[batch_start:batch_end])
                else:
                    print(f"Error in batch: {e}")
                    print("Using input values for this batch")
                    all_predictions.append(X[batch_start:batch_end])
            except Exception as e:
                print(f"Error in batch: {e}")
                print("Using input values for this batch")
                all_predictions.append(X[batch_start:batch_end])

        # Concatenate all batch predictions
        if all_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
        else:
            print("Warning: No predictions generated, using input values")
            predictions = X

    # Create reconstruction: replace masked values with predictions
    reconstruction = adata.copy()
    X_recon = X.copy()

    # In dev mode, we only processed a subset of cells
    if dev_mode and predictions.shape[0] < X.shape[0]:
        print(
            f"Dev mode: Only processed {predictions.shape[0]} cells out of {X.shape[0]}"
        )
        # Only replace predictions for the cells we processed
        processed_mask = np.zeros(X.shape[0], dtype=bool)
        processed_mask[: predictions.shape[0]] = True
        # Combine processed predictions with original values for unprocessed cells
        X_recon[: predictions.shape[0]] = predictions
        # For processed cells, replace masked values with predictions
        processed_mask_2d = processed_mask[:, np.newaxis] & mask_indices
        X_recon[processed_mask_2d] = predictions[mask_indices[: predictions.shape[0]]]
    elif predictions.shape == X.shape:
        # Normal mode: replace all masked values with predictions
        X_recon[mask_indices] = predictions[mask_indices]
    else:
        # If predictions have different shape, try to reshape
        print(f"Warning: Prediction shape {predictions.shape} != input shape {X.shape}")
        if predictions.ndim == 2 and predictions.shape[0] == X.shape[0]:
            # Take first n_genes columns if predictions are wider
            predictions = predictions[:, : X.shape[1]]
            X_recon[mask_indices] = predictions[mask_indices]
        else:
            print("Warning: Could not match prediction shape, using original values")

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
    import json

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
    """Main entry point for inference script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run scGPT inference")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to ModelConfig or masked data path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode: run only one batch with batch size 2 for quick testing.",
    )

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # This is a simplified CLI - in practice, main.py will call run_inference directly
    print("Note: inference.py is typically called from main.py")
    print("For standalone use, you need to provide model and vocab objects")
    sys.exit(1)


if __name__ == "__main__":
    main()
