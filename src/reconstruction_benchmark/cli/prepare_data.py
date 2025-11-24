"""CLI for data preparation."""

import argparse
from pathlib import Path

from ..config import DataConfig
from ..data import prepare_dataset


def main():
    """Main entry point for prepare-data CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for reconstruction benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pbmc",
        help="Dataset name (default: pbmc)",
    )
    parser.add_argument(
        "--mask-percentage",
        type=float,
        default=0.15,
        help="Percentage of values to mask (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for masking (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save data (default: data)",
    )
    
    args = parser.parse_args()
    
    config = DataConfig(
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    
    prepare_dataset(config)


if __name__ == "__main__":
    main()

