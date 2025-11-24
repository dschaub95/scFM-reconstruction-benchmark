"""CLI for running evaluations."""

import argparse
import json
from pathlib import Path

import anndata as ad

from ..config import EvaluationConfig, ModelConfig
from ..masking import get_mask_indices
from ..metrics import get_registry


def main():
    """Main entry point for run-evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on reconstruction results"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., scgpt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., pbmc)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing results (default: results)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing data (default: data)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compute (default: all)",
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
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        results_dir=args.results_dir,
    )
    
    # Create ModelConfig to get data paths
    model_config = ModelConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        mask_percentage=args.mask_percentage,
        seed=args.seed,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    
    # Load ground truth (raw data)
    raw_data_path = model_config.raw_data_path
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_data_path}")
    ground_truth = ad.read_h5ad(raw_data_path)
    
    # Load masked data to get mask indices
    masked_data_path = model_config.masked_data_path
    if not masked_data_path.exists():
        raise FileNotFoundError(f"Masked data not found: {masked_data_path}")
    masked_data = ad.read_h5ad(masked_data_path)
    mask_indices = get_mask_indices(masked_data)
    
    # Load reconstruction
    reconstruction_path = config.evaluation_dir / "reconstruction.h5ad"
    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Reconstruction not found: {reconstruction_path}")
    reconstruction = ad.read_h5ad(reconstruction_path)
    
    # Compute metrics
    registry = get_registry()
    
    if args.metrics:
        # Compute specific metrics
        results = {}
        for metric_name in args.metrics:
            metric = registry.get(metric_name)
            try:
                results[metric_name] = metric.compute(
                    ground_truth, reconstruction, mask_indices
                )
            except Exception as e:
                print(f"Warning: Failed to compute {metric_name}: {e}")
                results[metric_name] = None
    else:
        # Compute all metrics
        results = registry.compute_all(ground_truth, reconstruction, mask_indices)
    
    # Save results
    config.evaluation_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.evaluation_dir / "metrics.json"
    
    # Add metadata
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "metrics": results,
    }
    
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Evaluation results saved to {results_path}")
    print("\nMetrics:")
    for metric_name, value in results.items():
        if value is not None:
            print(f"  {metric_name}: {value:.6f}")
        else:
            print(f"  {metric_name}: Failed")


if __name__ == "__main__":
    main()

