"""Main entry point for scGPT reconstruction benchmark."""

import argparse
import sys
from pathlib import Path

import torch

from reconstruction_benchmark.config import ModelConfig
from reconstruction_benchmark.utils import get_project_root

# Import functions from other scripts
from download_ckpt import ensure_checkpoint
from inference import load_masked_data, run_inference, save_reconstruction
from setup import setup_model

# Define execution stages in order
STAGES = ["install", "download", "load_data", "setup", "inference", "save"]


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
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip environment installation step",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip checkpoint download step",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to checkpoint directory (if not using auto-download)",
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
        model_name="scgpt",
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        data_dir=data_dir,
        results_dir=results_dir,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Step 1: Install environment (if needed)
    if not args.skip_install:
        from install import install_environment

        print("Checking/installing environment...")
        install_environment()

    if args.stage == "install":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 2: Ensure checkpoint is available
    checkpoint_path = ensure_checkpoint(
        checkpoint_path=args.checkpoint_path,
        skip_download=args.skip_download,
        checkpoint_name="scGPT-human",
        gdrive_folder_id="1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
    )

    if args.stage == "download":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 3: Load masked data
    print(f"Loading masked data from {config.masked_data_path}...")
    masked_data = load_masked_data(config)

    if args.stage == "load_data":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 4: Setup model (load checkpoint and initialize)
    print("Setting up scGPT model...")
    gene_names = masked_data.var_names.tolist()
    model, vocab = setup_model(
        checkpoint_path=checkpoint_path,
        device=device,
        gene_names=gene_names,
    )

    if args.stage == "setup":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 5: Run inference
    print("Running scGPT reconstruction...")
    reconstruction = run_inference(
        model=model,
        vocab=vocab,
        adata=masked_data,
        device=device,
        dev_mode=args.dev,
    )

    if args.stage == "inference":
        print(f"Stopping after stage: {args.stage}")
        return

    # Step 6: Save reconstruction
    save_reconstruction(
        reconstruction=reconstruction,
        config=config,
    )

    if args.stage == "save":
        print(f"Stopping after stage: {args.stage}")
        return


if __name__ == "__main__":
    main()
