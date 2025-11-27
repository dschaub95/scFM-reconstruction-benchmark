"""Run all models using subprocesses."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from reconstruction_benchmark.utils import get_project_root


def resolve_file_path(model_dir: Path, file_name: str) -> Optional[Path]:
    """
    Resolve file path in model directory, auto-appending .py if needed.

    Parameters
    ----------
    model_dir
        Path to the model directory
    file_name
        File name (with or without .py extension)

    Returns
    -------
    Path to the file if it exists, None otherwise
    """
    if not file_name.endswith(".py"):
        file_name = file_name + ".py"

    file_path = model_dir / file_name
    if file_path.exists():
        return file_path
    return None


def detect_venv_mode(model_dir: Path) -> str:
    """
    Detect whether to use uv or conda mode based on pyproject.toml presence.

    Parameters
    ----------
    model_dir
        Directory containing the model (where pyproject.toml would be)

    Returns
    -------
    "uv" if pyproject.toml exists, "conda" otherwise
    """
    pyproject_path = model_dir / "pyproject.toml"
    return "uv" if pyproject_path.exists() else "conda"


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


def run_model(
    model_name: str,
    model_dir: Path,
    file_name: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    Run a single model script.

    Automatically detects venv mode (uv vs conda) and ensures venv exists before running.

    Parameters
    ----------
    model_name
        Name of the model (directory name)
    model_dir
        Path to the model directory
    file_name
        Specific file to run (e.g., 'install.py'). If None, runs main.py
    extra_args
        Additional arguments to pass to the file being run

    Returns
    -------
    True if model ran successfully, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"Running model: {model_name}")
    if file_name:
        print(f"Running file: {file_name}")
    print(f"{'=' * 60}")

    # Resolve file path
    if file_name:
        target_file = resolve_file_path(model_dir, file_name)
        if target_file is None:
            print(
                f"Error: File '{file_name}' not found in {model_dir}", file=sys.stderr
            )
            return False
    else:
        target_file = model_dir / "main.py"
        if not target_file.exists():
            print(f"Error: main.py not found in {model_dir}", file=sys.stderr)
            return False

    # Detect venv mode
    venv_mode = detect_venv_mode(model_dir)

    if venv_mode == "uv":
        # UV mode: use uv run --directory
        # Check if venv needs to be created
        venv_python = model_dir / ".venv" / "bin" / "python"
        if not venv_python.exists():
            # Run install.py to create venv
            install_script = model_dir / "install.py"
            if not install_script.exists():
                print(
                    f"Error: install.py not found at {install_script}",
                    file=sys.stderr,
                )
                return False

            print("Venv not found. Running install.py to create venv...")
            result = subprocess.run(
                [sys.executable, str(install_script)],
                cwd=model_dir,
                check=False,
            )
            if result.returncode != 0:
                print("Error: Failed to create venv via install.py", file=sys.stderr)
                return False

            # Verify venv was created
            venv_python = model_dir / ".venv" / "bin" / "python"
            if not venv_python.exists():
                print("Error: venv not found after running install.py", file=sys.stderr)
                return False

        # Build command using uv run --directory
        run_cmd = [
            "uv",
            "run",
            "--directory",
            str(model_dir),
            "python",
            "-u",
            str(target_file),
        ]

        # Add any extra arguments (flags) passed through
        if extra_args:
            run_cmd.extend(extra_args)

    else:
        # Conda mode: use current behavior (not implemented yet)
        print(
            "Error: Conda mode is not yet implemented. "
            "Please use a model with pyproject.toml for uv mode.",
            file=sys.stderr,
        )
        return False

    # Execute model using subprocess
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
        description="Run all models for reconstruction benchmark",
        allow_abbrev=False,  # Prevent abbreviation conflicts with model arguments
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
        "-m",
        "--model",
        type=str,
        default=None,
        help="Specific model to run (default: all discovered models)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Specific file to run (e.g., 'install', 'install.py', 'download_ckpt.py'). Defaults to main.py. All other arguments will be passed to the model's main.py.",
    )

    # Use parse_known_args to allow pass-through of arguments to the file being run
    # This allows extra arguments to be passed to main.py when a single model is specified
    args, unknown_args = parser.parse_known_args()

    # Get project root
    project_root = get_project_root()

    # Set default paths relative to project root
    models_dir = (project_root / (args.models_dir or Path("models"))).resolve()

    # Discover models
    all_models = discover_models(models_dir)

    if not all_models:
        print(f"No models found in {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Handle model filtering: --model takes precedence over --models
    if args.model:
        if args.model not in all_models:
            print(f"Model '{args.model}' not found", file=sys.stderr)
            print(f"Available models: {all_models}", file=sys.stderr)
            sys.exit(1)
        models_to_run = [args.model]
    elif args.models:
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

    # Validate that pass-through arguments are only used with a single model
    if unknown_args and len(models_to_run) > 1:
        print(
            "Error: Cannot pass arguments to file when running multiple models.",
            file=sys.stderr,
        )
        print(
            "Please specify a single model with -m/--model to pass additional arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If unknown arguments are provided but no single model is specified, treat as error
    if unknown_args and len(models_to_run) != 1:
        parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")

    print(f"Found {len(all_models)} model(s): {', '.join(all_models)}")
    print(f"Will run {len(models_to_run)} model(s): {', '.join(models_to_run)}")
    if args.file:
        print(f"Will run file: {args.file}")
    if unknown_args and len(models_to_run) == 1:
        file_name = args.file or "main.py"
        print(f"Passing arguments to {file_name}: {' '.join(unknown_args)}")

    # Run each model
    results = {}
    for model_name in models_to_run:
        model_dir = models_dir / model_name
        # Pass all unknown args to model's main.py (only if single model)
        extra_args_to_pass = unknown_args if len(models_to_run) == 1 else None
        success = run_model(
            model_name=model_name,
            model_dir=model_dir,
            file_name=args.file,
            extra_args=extra_args_to_pass,
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
