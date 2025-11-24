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
│           └── run_evaluation.py     # run-evaluation CLI
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
    └── <dataset>/
        └── mask_<pct>_seed_<seed>/   # Experimental condition
            └── <model>/
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

### 1. Prepare Data

Download and mask a dataset:

```bash
prepare-data --dataset pbmc --mask-percentage 0.15 --seed 42
```

This will:
- Download the PBMC dataset from Scanpy (if not already downloaded)
- Mask 15% of values (replaced with -1)
- Save raw data to `data/pbmc/raw/data.h5ad`
- Save masked data to `data/pbmc/masked/mask_15_seed_42.h5ad`

### 2. Run Model

Run a model on the prepared data. For scGPT:

```bash
cd models/scgpt
uv run scgpt-run --dataset pbmc
```

Or with conda:

```bash
cd models/scgpt
conda run -n scgpt-env python main.py --dataset pbmc
```

This will:
- Load masked data from `data/pbmc/masked/mask_15_seed_42.h5ad`
- Convert -1 values to model-specific mask tokens
- Run reconstruction
- Save results to `results/pbmc/mask_15_seed_42/scgpt/`

### 3. Evaluate Results

Run evaluation metrics:

```bash
run-evaluation --model scgpt --dataset pbmc
```

This will:
- Load ground truth and reconstruction
- Compute metrics (MSE, MAE, Pearson correlation)
- Save results to `results/pbmc/mask_15_seed_42/scgpt/metrics.json`

## Adding New Models

1. Create a new folder under `models/` (e.g., `models/newmodel/`)
2. Add a `pyproject.toml` with model-specific dependencies
3. Create a `main.py` with CLI that:
   - Loads masked data from `data/<dataset>/masked/mask_<pct>_seed_<seed>.h5ad`
   - Converts -1 to model-specific mask tokens
   - Runs reconstruction
   - Saves to `results/<dataset>/mask_<pct>_seed_<seed>/<model>/reconstruction.h5ad`

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
- **Results**: `results/<dataset>/mask_<pct>_seed_<seed>/<model>/`
  - All outputs (reconstruction.h5ad, metadata.json, metrics.json) are stored together
  - Experimental conditions (mask percentage + seed) are grouped together
  - Makes it easy to compare models on the same experimental setup
- Masked values are stored as `-1` in the data files
- Models convert `-1` to their specific mask tokens during inference

