"""Run all models using subprocesses."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


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


def get_model_script_name(model_dir: Path) -> Optional[str]:
    """
    Extract the script name from a model's pyproject.toml.

    Parameters
    ----------
    model_dir
        Path to the model directory

    Returns
    -------
    Script name if found, None otherwise
    """
    pyproject_path = model_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    try:
        import tomli
    except ImportError:
        # Fallback to basic parsing if tomli is not available
        with open(pyproject_path) as f:
            in_scripts_section = False
            for line in f:
                line = line.strip()
                if line.startswith("[project.scripts]"):
                    in_scripts_section = True
                    continue
                if line.startswith("[") and in_scripts_section:
                    break
                if in_scripts_section and "=" in line:
                    # Extract script name (everything before =)
                    script_name = line.split("=")[0].strip()
                    if script_name:
                        return script_name
        return None

    with open(pyproject_path, "rb") as f:
        config = tomli.load(f)

    scripts = config.get("project", {}).get("scripts", {})
    if scripts:
        # Return the first script name
        return list(scripts.keys())[0]

    return None


def get_project_root() -> Path:
    """
    Get the project root directory.

    When running from the installed package, the project root is 3 levels up
    from this file: cli -> reconstruction_benchmark -> src -> project_root

    Returns
    -------
    Path to the project root directory
    """
    # This file is at: src/reconstruction_benchmark/cli/run_models.py
    # Project root is: ../../../
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent
    return project_root


def check_uv_available() -> bool:
    """
    Check if uv command is available.

    Returns
    -------
    True if uv is available, False otherwise
    """
    try:
        subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_parent_package(project_root: Path, use_uv: bool = False) -> bool:
    """
    Install the parent reconstruction-benchmark package in editable mode.

    Parameters
    ----------
    project_root
        Path to the project root directory
    use_uv
        Whether to use uv instead of pip

    Returns
    -------
    True if installation succeeded, False otherwise
    """
    print(f"Installing parent package from {project_root}...")

    if use_uv:
        # Use uv pip install
        install_cmd = [
            "uv",
            "pip",
            "install",
            "-e",
            str(project_root),
        ]
    else:
        # Use standard pip
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(project_root),
        ]

    try:
        subprocess.run(
            install_cmd,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        print("Parent package installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing parent package: {e.stderr}", file=sys.stderr)
        return False


def run_model(
    model_name: str,
    model_dir: Path,
    project_root: Path,
    dataset: str = "pbmc",
    data_dir: Path = Path("data"),
    results_dir: Path = Path("results"),
    mask_percentage: float = 0.15,
    seed: int = 42,
    use_uv: bool = False,
    dev_mode: bool = False,
) -> bool:
    """
    Install and run a single model.

    Parameters
    ----------
    model_name
        Name of the model (directory name)
    model_dir
        Path to the model directory
    project_root
        Path to the project root directory
    dataset
        Dataset name to use
    data_dir
        Data directory path
    results_dir
        Results directory path
    mask_percentage
        Mask percentage used for masked data
    use_uv
        Whether to use uv instead of pip
    dev_mode
        Whether to run in development mode (pass --dev flag to model)

    Returns
    -------
    True if model ran successfully, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"Running model: {model_name}")
    print(f"{'=' * 60}")

    # Get script name from pyproject.toml
    script_name = get_model_script_name(model_dir)

    if use_uv:
        # Use uv run --directory to run the model
        # First sync dependencies, then install parent package
        print(f"Preparing environment for {model_name}...")

        # First, sync dependencies from pyproject.toml
        # This will create the venv and install all dependencies including the model package
        # reconstruction-benchmark is not in pyproject.toml, so uv sync will only install
        # the other dependencies, which is what we want
        try:
            subprocess.run(
                ["uv", "sync", "--directory", str(model_dir)],
                cwd=model_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error syncing model dependencies: {e.stderr}", file=sys.stderr)
            return False

        # Get the venv Python path (should exist after uv sync)
        venv_python = model_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():
            print(f"Error: venv not found after sync for {model_name}", file=sys.stderr)
            return False

        # Install the parent package in the model's venv
        # This makes reconstruction-benchmark available to the model
        try:
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "-e",
                    str(project_root),
                ],
                cwd=model_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error installing parent package: {e.stderr}", file=sys.stderr)
            return False

        # Run the model script using the venv's Python directly
        # This avoids needing to install the model as a package
        print(f"Running {model_name}...")
        # Always run main.py directly since we're not installing the model package
        # Use -u flag for unbuffered output to ensure checkpoint loading messages are visible
        run_cmd = [str(venv_python), "-u", str(model_dir / "main.py")]
    else:
        # Use pip to install and run
        print(f"Installing {model_name}...")
        # Install model with parent package explicitly to ensure local version is used
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(model_dir),
            "-e",
            str(project_root),
        ]

        try:
            subprocess.run(
                install_cmd,
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error installing {model_name}: {e.stderr}", file=sys.stderr)
            return False

        # Run the model script
        if script_name:
            run_cmd = [script_name]
        else:
            # Fallback to running main.py directly
            # Use -u flag for unbuffered output to ensure checkpoint loading messages are visible
            run_cmd = [sys.executable, "-u", str(model_dir / "main.py")]

    # Add common arguments
    run_cmd.extend(
        [
            "--dataset",
            dataset,
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(results_dir),
            "--mask-percentage",
            str(mask_percentage),
            "--seed",
            str(seed),
        ]
    )

    # Add dev flag if dev_mode is enabled
    if dev_mode:
        run_cmd.append("--dev")

    print(f"Running: {' '.join(run_cmd)}")
    try:
        subprocess.run(
            run_cmd,
            cwd=model_dir,
            check=True,
        )
        print(f"✓ {model_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {model_name} failed with exit code {e.returncode}", file=sys.stderr)
        return False


def main():
    """Main entry point for run-models script."""
    parser = argparse.ArgumentParser(
        description="Run all models for reconstruction benchmark"
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
        help="Random seed used for masking (default: 42)",
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
        "--dev",
        action="store_true",
        help="Development mode: pass --dev flag to all model scripts for quick testing",
    )

    args = parser.parse_args()

    # Get project root
    project_root = get_project_root()

    # Set default paths relative to project root
    models_dir = (project_root / (args.models_dir or Path("models"))).resolve()
    data_dir = (project_root / (args.data_dir or Path("data"))).resolve()
    results_dir = (project_root / (args.results_dir or Path("results"))).resolve()

    # Discover models
    all_models = discover_models(models_dir)

    if not all_models:
        print(f"No models found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Filter to requested models if specified
    if args.models:
        models_to_run = [m for m in args.models if m in all_models]
        if not models_to_run:
            print(f"None of the specified models found: {args.models}", file=sys.stderr)
            print(f"Available models: {all_models}", file=sys.stderr)
            sys.exit(1)
        if len(models_to_run) < len(args.models):
            missing = set(args.models) - set(models_to_run)
            print(f"Warning: Some models not found: {missing}", file=sys.stderr)
    else:
        models_to_run = all_models

    print(f"Found {len(all_models)} model(s): {', '.join(all_models)}")
    print(f"Will run {len(models_to_run)} model(s): {', '.join(models_to_run)}")

    # Auto-detect uv if not explicitly set
    if not args.use_uv:
        args.use_uv = check_uv_available()
        if args.use_uv:
            print("Detected uv, using it for package management")

    # Install parent package if needed
    if not args.skip_parent_install:
        if not install_parent_package(project_root, use_uv=args.use_uv):
            print("Failed to install parent package. Exiting.", file=sys.stderr)
            sys.exit(1)

    # Run each model
    results = {}
    for model_name in models_to_run:
        model_dir = models_dir / model_name
        success = run_model(
            model_name=model_name,
            model_dir=model_dir,
            project_root=project_root,
            dataset=args.dataset,
            data_dir=data_dir,
            results_dir=results_dir,
            mask_percentage=args.mask_percentage,
            seed=args.seed,
            use_uv=args.use_uv,
            dev_mode=args.dev,
        )
        results[model_name] = success

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{model_name}: {status}")

    # Exit with error if any model failed
    if not all(results.values()):
        sys.exit(1)
