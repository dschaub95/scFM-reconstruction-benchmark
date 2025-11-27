# Reconstruction Benchmark

A benchmark for evaluating single-cell foundation models on gene expression reconstruction tasks.

## Structure

```
reconstruction-benchmark/
├── src/
│   └── reconstruction_benchmark/    # Core package (installable)
│       ├── __init__.py
│       ├── config.py                # Configuration schemas
│       ├── masking.py                # Masking utilities
│       ├── metrics.py                # Metrics registry
│       ├── data_prep.py              # Data preparation utilities
│       └── cli/
│           ├── prepare_data.py       # prepare-data CLI
│           ├── run_evaluation.py     # run-evaluation CLI
│           ├── run_models.py         # run-models CLI
│           └── run_benchmark.py      # run-benchmark CLI (complete pipeline)
├── models/
│   └── scgpt/                        # scGPT model implementation
│       ├── pyproject.toml            # Model-specific dependencies
│       └── main.py                   # Model runner CLI
├── data/                             # Prepared datasets (generated)
│   └── <dataset>/
│       ├── raw/
│       │   └── data.h5ad            # Raw data (shared across experiments)
│       └── masked/
│           ├── mask_<pct>_seed_<seed>.h5ad  # Multiple masked files
│           └── ...
└── results/                          # Model results (generated)
    └── <model>/
        └── <dataset>/
            └── mask_<pct>_seed_<seed>/   # Experimental condition
                ├── reconstruction.h5ad
                ├── metadata.json
                └── metrics.json
```

## Installation

Install the core package:

```bash
pip install -e .
```

## Usage

### Complete Pipeline (Recommended)

Run the complete pipeline (prepare data → run models → evaluate):

```bash
run-benchmark --dataset pbmc
```

This automatically:
1. Prepares dataset with masking
2. Discovers and runs all models in `models/`
3. Evaluates all reconstructions

### Individual Steps

**1. Prepare Data**

```bash
prepare-data --dataset pbmc --mask-percentage 0.15 --seed 42
```

**2. Run All Models**

```bash
run-models --dataset pbmc
```

**3. Evaluate Results**

```bash
run-evaluation --model scgpt --dataset pbmc
```

## Adding New Models

1. Create a new folder under `models/` (e.g., `models/newmodel/`)
2. Add a `pyproject.toml` with model-specific dependencies
3. Create a `main.py` with CLI that:
   - Loads masked data from `data/<dataset>/masked/mask_<pct>_seed_<seed>.h5ad`
   - Converts -1 to model-specific mask tokens
   - Runs reconstruction
   - Saves to `results/<model>/<dataset>/mask_<pct>_seed_<seed>/reconstruction.h5ad`

## Adding New Metrics

Metrics are registered in `src/reconstruction_benchmark/metrics.py`. To add a new metric:

1. Create a class inheriting from `Metric`
2. Implement `name` property and `compute()` method
3. Register it in `MetricRegistry._register_defaults()`

## File Structure Conventions

- **Data**: 
  - Raw data: `data/<dataset>/raw/data.h5ad` (shared across all experiments)
  - Masked data: `data/<dataset>/masked/mask_<pct>_seed_<seed>.h5ad` (one file per experimental condition)
  - Each dataset has `raw/` and `masked/` subfolders
  - Multiple masked files can exist in the `masked/` folder for different mask percentages and seeds
- **Results**: `results/<model>/<dataset>/mask_<pct>_seed_<seed>/`
  - All outputs (reconstruction.h5ad, metadata.json, metrics.json) are stored together
  - Results are organized by model first, then dataset, then experimental condition (mask percentage + seed)
  - Makes it easy to find all results for a specific model
- Masked values are stored as `-1` in the data files
- Models convert `-1` to their specific mask tokens during inference

