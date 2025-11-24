"""CLI for running the complete reconstruction benchmark pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    When running from the installed package, the project root is 3 levels up
    from this file: cli -> reconstruction_benchmark -> src -> project_root

    Returns
    -------
    Path to the project root directory
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    return project_root


def discover_models(models_dir: Path) -> List[str]:
    """
    Discover all models in the models directory.

    A model is identified by having both a pyproject.toml and main.py file.

    Parameters
    ----------
    models_dir
        Path to the models directory

    Returns
    -------
    List of model names (directory names)
    """
    models = []
    if not models_dir.exists():
        return models

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        pyproject_path = model_dir / "pyproject.toml"
        main_path = model_dir / "main.py"

        if pyproject_path.exists() and main_path.exists():
            models.append(model_dir.name)

    return sorted(models)


def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    """
    Run a command and return success status.

    Parameters
    ----------
    cmd
        Command to run as a list of strings
    description
        Description of what the command does
    cwd
        Working directory for the command

    Returns
    -------
    True if command succeeded, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}", file=sys.stderr)
        return False


def main():
    """Main entry point for run-benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run complete reconstruction benchmark pipeline: prepare data, run models, evaluate results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pbmc",
        help="Dataset name (default: pbmc)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing data (default: data relative to project root)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory to save results (default: results relative to project root)",
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
        help="Random seed for masking (default: 42)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing models (default: models relative to project root)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to run (default: all discovered models)",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Use uv instead of pip for running models",
    )
    parser.add_argument(
        "--skip-parent-install",
        action="store_true",
        help="Skip installing parent package (assumes already installed)",
    )
    parser.add_argument(
        "--skip-prepare-data",
        action="store_true",
        help="Skip data preparation step (assumes data already prepared)",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip running models step (assumes models already run)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to compute (default: all)",
    )

    args = parser.parse_args()

    # Get project root
    project_root = get_project_root()

    # Set default paths relative to project root
    models_dir = (project_root / (args.models_dir or Path("models"))).resolve()
    data_dir = (project_root / (args.data_dir or Path("data"))).resolve()
    results_dir = (project_root / (args.results_dir or Path("results"))).resolve()

    # Step 1: Prepare data
    if not args.skip_prepare_data:
        prepare_cmd = [
            "prepare-data",
            "--dataset",
            args.dataset,
            "--mask-percentage",
            str(args.mask_percentage),
            "--seed",
            str(args.seed),
            "--data-dir",
            str(data_dir),
        ]
        if not run_command(prepare_cmd, "Preparing data", cwd=project_root):
            print("Data preparation failed. Exiting.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Skipping data preparation step")

    # Step 2: Run models
    if not args.skip_models:
        models_cmd = [
            "run-models",
            "--dataset",
            args.dataset,
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(results_dir),
            "--mask-percentage",
            str(args.mask_percentage),
            "--seed",
            str(args.seed),
            "--models-dir",
            str(models_dir),
        ]
        if args.models:
            models_cmd.extend(["--models"] + args.models)
        if args.use_uv:
            models_cmd.append("--use-uv")
        if args.skip_parent_install:
            models_cmd.append("--skip-parent-install")

        if not run_command(models_cmd, "Running models", cwd=project_root):
            print("Running models failed. Exiting.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Skipping models step")

    # Step 3: Run evaluation for all models
    if not args.skip_evaluation:
        # Discover models
        all_models = discover_models(models_dir)
        if not all_models:
            print(f"No models found in {models_dir}", file=sys.stderr)
            sys.exit(1)

        # Filter to requested models if specified
        if args.models:
            models_to_evaluate = [m for m in args.models if m in all_models]
            if not models_to_evaluate:
                print(
                    f"None of the specified models found: {args.models}",
                    file=sys.stderr,
                )
                print(f"Available models: {all_models}", file=sys.stderr)
                sys.exit(1)
        else:
            models_to_evaluate = all_models

        print(f"\n{'=' * 60}")
        print(f"Running evaluation for {len(models_to_evaluate)} model(s)")
        print(f"{'=' * 60}")

        evaluation_results = {}
        for model_name in models_to_evaluate:
            eval_cmd = [
                "run-evaluation",
                "--model",
                model_name,
                "--dataset",
                args.dataset,
                "--results-dir",
                str(results_dir),
                "--data-dir",
                str(data_dir),
                "--mask-percentage",
                str(args.mask_percentage),
                "--seed",
                str(args.seed),
            ]
            if args.metrics:
                eval_cmd.extend(["--metrics"] + args.metrics)

            success = run_command(
                eval_cmd,
                f"Evaluating {model_name}",
                cwd=project_root,
            )
            evaluation_results[model_name] = success

        # Print evaluation summary
        print(f"\n{'=' * 60}")
        print("Evaluation Summary")
        print(f"{'=' * 60}")
        for model_name, success in evaluation_results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{model_name}: {status}")

        # Exit with error if any evaluation failed
        if not all(evaluation_results.values()):
            print("\nSome evaluations failed.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Skipping evaluation step")

    print(f"\n{'=' * 60}")
    print("Benchmark completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
