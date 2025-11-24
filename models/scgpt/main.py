"""Main entry point for scGPT reconstruction benchmark."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import reconstruction_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import anndata as ad
import torch

from reconstruction_benchmark.config import ModelConfig

try:
    # Import scGPT - use the actual API structure
    from scgpt.model.model import TransformerModel as scGPT
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    # scGPTConfig might not exist as a separate class - TransformerModel uses dict/config
    # We'll handle config creation inline
    scGPTConfig = None  # Will create config dict directly

    SCGPT_AVAILABLE = True
except ImportError as e:
    SCGPT_AVAILABLE = False
    print(f"Warning: scGPT imports failed: {e}")
    print("Install with: pip install scgpt 'flash-attn<1.0.5' ipython")


def get_checkpoints_dir() -> Path:
    """
    Get the checkpoints directory for scGPT models.

    Returns
    -------
    Path to the checkpoints directory (models/scgpt/checkpoints/)
    """
    scgpt_dir = Path(__file__).parent
    checkpoints_dir = scgpt_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_local_checkpoint_path(model_name: str) -> Path:
    """
    Get the local checkpoint path for a given model name.

    Parameters
    ----------
    model_name
        Model name (e.g., "bwanglab/scGPT" or "scGPT-human")

    Returns
    -------
    Path to the local checkpoint directory
    """
    checkpoints_dir = get_checkpoints_dir()
    # Convert HuggingFace model ID to a safe directory name
    checkpoint_name = model_name.replace("/", "_")
    checkpoint_path = checkpoints_dir / checkpoint_name
    return checkpoint_path


def download_checkpoint_from_gdrive(
    gdrive_folder_id: str,
    output_dir: Path,
    quiet: bool = False,
) -> bool:
    """
    Download a checkpoint folder from Google Drive.

    Parameters
    ----------
    gdrive_folder_id
        Google Drive folder ID (from the shareable link)
    output_dir
        Directory where the checkpoint files will be saved
    quiet
        If True, suppress output messages

    Returns
    -------
    True if download succeeded, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Error: gdown is not installed. Install with: pip install gdown")
        return False

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct Google Drive folder URL
    folder_url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"

    try:
        if not quiet:
            print(f"Downloading checkpoint from Google Drive: {folder_url}")
            print(f"Saving to: {output_dir}")

        # Download the entire folder
        # gdown.download_folder downloads all files in the folder
        gdown.download_folder(
            folder_url,
            output=str(output_dir),
            quiet=quiet,
            use_cookies=False,
        )

        # Verify that essential files were downloaded
        required_files = ["best_model.pt"]
        missing_files = [f for f in required_files if not (output_dir / f).exists()]

        if missing_files:
            print(f"Warning: Some required files are missing: {missing_files}")
            # Check if files are in subdirectories (gdown sometimes creates subdirs)
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    for file in required_files:
                        if (subdir / file).exists() and not (
                            output_dir / file
                        ).exists():
                            # Move file to output_dir
                            import shutil

                            shutil.move(str(subdir / file), str(output_dir / file))
                            if not quiet:
                                print(f"Moved {file} from {subdir} to {output_dir}")

        if not quiet:
            print(f"Successfully downloaded checkpoint to: {output_dir}")
        return True

    except Exception as e:
        print(f"Error downloading checkpoint from Google Drive: {e}")
        return False


def download_scgpt_checkpoint_from_gdrive(
    checkpoint_name: str = "scGPT-human",
    gdrive_folder_id: str = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
    force_download: bool = False,
) -> Path:
    """
    Download the scGPT checkpoint from Google Drive to local checkpoints directory.

    This function downloads the official scGPT checkpoint from the Google Drive folder
    containing best_model.pt, vocab.json, and args.json.

    Parameters
    ----------
    checkpoint_name
        Name for the checkpoint (used as directory name)
    gdrive_folder_id
        Google Drive folder ID (default: official scGPT checkpoint folder)
    force_download
        If True, re-download even if checkpoint already exists

    Returns
    -------
    Path to the downloaded checkpoint directory
    """
    checkpoint_path = get_local_checkpoint_path(checkpoint_name)

    # Check if checkpoint already exists
    if checkpoint_path.exists() and (checkpoint_path / "best_model.pt").exists():
        if not force_download:
            print(f"Checkpoint already exists at: {checkpoint_path}")
            print("Skipping download. Use force_download=True to re-download.")
            return checkpoint_path
        else:
            print(f"Force download enabled. Re-downloading checkpoint...")

    # Download from Google Drive
    success = download_checkpoint_from_gdrive(
        gdrive_folder_id=gdrive_folder_id,
        output_dir=checkpoint_path,
        quiet=False,
    )

    if success:
        return checkpoint_path
    else:
        raise RuntimeError(
            f"Failed to download checkpoint from Google Drive to {checkpoint_path}"
        )


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


def run_scgpt_reconstruction(
    adata: ad.AnnData,
    model_name: str = "bwanglab/scGPT",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dev_mode: bool = False,
) -> ad.AnnData:
    """
    Run scGPT reconstruction on masked data.

    Parameters
    ----------
    adata
        AnnData object with masked data
    model_name
        Name of the scGPT model checkpoint to use
    device
        Device to run inference on

    Returns
    -------
    AnnData object with reconstructed values
    """
    if not SCGPT_AVAILABLE:
        raise ImportError(
            "scGPT is not installed. Install with: pip install scgpt 'flash-attn<1.0.5'"
        )

    # Validate and potentially fix device
    original_device = device
    if device == "cuda":
        # Test if CUDA actually works
        try:
            test_tensor = torch.zeros(1).to(device)
            _ = test_tensor + 1  # Simple operation to test CUDA
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
    # Development mode: use batch size 2 and only process one batch
    if dev_mode:
        batch_size = 2
        print("Development mode: using batch size 2, processing only one batch")
    elif n_cells * n_genes > 50_000_000:
        # Very large dataset - use very small batches
        batch_size = min(10, n_cells)  # Process 10 cells at a time
        print(
            f"Very large dataset detected ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    elif n_cells * n_genes > 10_000_000:
        # Large dataset - use small batches
        batch_size = min(50, n_cells)  # Process 50 cells at a time
        print(
            f"Large dataset detected ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    elif n_cells * n_genes > 1_000_000:
        # Moderately large dataset
        batch_size = min(100, n_cells)  # Process 100 cells at a time
        print(
            f"Moderately large dataset ({n_cells * n_genes:,} values). Processing in batches of {batch_size} cells."
        )
    else:
        batch_size = n_cells  # Process all at once for smaller datasets

    # Create gene vocabulary with special tokens
    try:
        # GeneVocab needs special tokens - add them to gene list
        special_tokens = [
            "<pad>",
            "<mask>",
            "<cls>",
            "<eoc>",
        ]  # Common scGPT special tokens
        all_tokens = special_tokens + gene_names
        gene_vocab = GeneVocab(all_tokens)
        # Set default index for unknown tokens
        if hasattr(gene_vocab, "set_default_index"):
            pad_idx = gene_vocab["<pad>"]
            gene_vocab.set_default_index(pad_idx)
        vocab_size = len(gene_vocab)
    except Exception as e:
        print(f"Warning: Could not create gene vocabulary: {e}")
        # Fallback: create vocab with special tokens manually
        special_tokens = ["<pad>", "<mask>", "<cls>", "<eoc>"]
        all_tokens = special_tokens + gene_names
        gene_vocab = GeneVocab(all_tokens)
        vocab_size = len(gene_vocab)

    # Initialize model with appropriate parameters
    # Using default parameters that work for reconstruction
    model = None
    try:
        # Try to load pretrained model if model_name is provided
        if model_name and hasattr(scGPT, "from_pretrained"):
            # Check if we have a local checkpoint first
            local_checkpoint_path = get_local_checkpoint_path(model_name)

            # Special handling for Google Drive checkpoint
            # If model_name is "gdrive" or starts with "gdrive:", download from Google Drive
            if model_name == "gdrive" or model_name.startswith("gdrive:"):
                gdrive_folder_id = (
                    "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"  # Default scGPT checkpoint
                )
                if ":" in model_name:
                    # Allow custom folder ID: "gdrive:FOLDER_ID"
                    gdrive_folder_id = model_name.split(":", 1)[1]

                checkpoint_name = (
                    "scGPT-human"  # Default name for Google Drive checkpoint
                )
                print(
                    f"Downloading checkpoint from Google Drive (folder: {gdrive_folder_id})..."
                )
                try:
                    local_checkpoint_path = download_scgpt_checkpoint_from_gdrive(
                        checkpoint_name=checkpoint_name,
                        gdrive_folder_id=gdrive_folder_id,
                        force_download=False,
                    )
                    model_name = str(
                        local_checkpoint_path
                    )  # Use local path for loading
                except Exception as e:
                    print(f"Failed to download from Google Drive: {e}")
                    print("Falling back to HuggingFace or new model initialization...")

            # Try local checkpoint first if it exists
            if (
                local_checkpoint_path.exists()
                and (local_checkpoint_path / "best_model.pt").exists()
            ):
                try:
                    print(
                        f"Loading checkpoint from local directory: {local_checkpoint_path}"
                    )
                    # Try loading from local directory
                    # scGPT's from_pretrained might accept a local path
                    model = scGPT.from_pretrained(str(local_checkpoint_path))
                    print(
                        f"Loaded pretrained model from local checkpoint: {local_checkpoint_path}"
                    )
                except Exception as e:
                    print(f"Could not load from local checkpoint: {e}")
                    print("Trying HuggingFace download instead...")
                    # Fall through to HuggingFace download

            # If local checkpoint doesn't exist or failed, try HuggingFace
            if model is None:
                try:
                    print(
                        f"Downloading/loading pretrained model from HuggingFace: {model_name}"
                    )
                    model = scGPT.from_pretrained(model_name)
                    print(f"Loaded pretrained model: {model_name}")

                    # Save checkpoint locally for future use
                    try:
                        local_checkpoint_path.mkdir(parents=True, exist_ok=True)
                        # If the model has a save_pretrained method, use it
                        if hasattr(model, "save_pretrained"):
                            model.save_pretrained(str(local_checkpoint_path))
                            print(f"Saved checkpoint to: {local_checkpoint_path}")
                        else:
                            # Try to save the model state dict manually
                            if hasattr(model, "state_dict"):
                                checkpoint_file = (
                                    local_checkpoint_path / "best_model.pt"
                                )
                                torch.save(model.state_dict(), checkpoint_file)
                                print(f"Saved model state dict to: {checkpoint_file}")
                    except Exception as save_error:
                        print(
                            f"Warning: Could not save checkpoint locally: {save_error}"
                        )
                        print("Model loaded but not cached locally")

                except Exception as e:
                    print(f"Could not load pretrained model {model_name}: {e}")
                    print("Initializing new model instead")

        if model is None:
            # Initialize new model with appropriate parameters for gene expression
            # Default parameters based on scGPT typical configuration
            model = scGPT(
                ntoken=vocab_size,
                d_model=512,  # Model dimension
                nhead=8,  # Number of attention heads
                d_hid=2048,  # Feedforward dimension
                nlayers=6,  # Number of transformer layers
                nlayers_cls=3,  # Classification layers
                vocab=gene_vocab,
                dropout=0.1,
                pad_token="<pad>",
                pad_value=0,
                do_mvc=False,  # Not using MVC decoder
                input_emb_style="continuous",  # Continuous value encoding
                cell_emb_style="cls",  # CLS token for cell embedding
            )
            print("Initialized new scGPT model")
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise RuntimeError(f"Failed to initialize scGPT model: {e}")

    if model is None:
        raise RuntimeError("Failed to initialize scGPT model")

    model = model.to(device)
    model.eval()

    # Prepare input
    # Convert expression to tensor (keep on CPU initially, move to device per batch)
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
        # Use mean prediction or handle appropriately
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


def main():
    """Main entry point for scGPT CLI."""
    parser = argparse.ArgumentParser(description="Run scGPT reconstruction benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pbmc",
        help="Dataset name (default: pbmc)",
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
        "--model-config",
        type=Path,
        default=None,
        help="Path to model configuration file (optional)",
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
        help="Random seed used for masking (default: 42)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bwanglab/scGPT",
        help="scGPT model checkpoint name. Options: 'bwanglab/scGPT' (HuggingFace, default), 'gdrive' (download from Google Drive), 'gdrive:FOLDER_ID' (custom Google Drive folder), or local path. Checkpoints are saved in models/scgpt/checkpoints/",
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

    config = ModelConfig(
        model_name="scgpt",
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load masked data
    print(f"Loading masked data from {config.masked_data_path}...")
    masked_data = load_masked_data(config)

    # Run reconstruction
    print("Running scGPT reconstruction...")
    reconstruction = run_scgpt_reconstruction(
        masked_data,
        model_name=args.model_name,
        device=device,
        dev_mode=args.dev,
    )

    # Save reconstruction
    output_path = config.output_dir / "reconstruction.h5ad"
    reconstruction.write(output_path)
    print(f"Saved reconstruction to {output_path}")

    # Save metadata
    metadata = {
        "model": "scgpt",
        "dataset": args.dataset,
        "n_cells": reconstruction.n_obs,
        "n_genes": reconstruction.n_vars,
    }

    metadata_path = config.output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
