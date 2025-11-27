"""Install environment for simple-baseline model runner."""

import subprocess
import sys
from pathlib import Path

from reconstruction_benchmark.utils import get_project_root


def verify_pyproject_toml(script_dir: Path) -> Path:
    """
    Verify that pyproject.toml exists in the script directory.

    Parameters
    ----------
    script_dir
        Directory where pyproject.toml should exist

    Returns
    -------
    Path to pyproject.toml

    Raises
    ------
    FileNotFoundError
        If pyproject.toml does not exist
    """
    pyproject_path = script_dir / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(
            f"pyproject.toml not found at {pyproject_path}. "
            "Please ensure pyproject.toml exists before running install.py"
        )

    print(f"✓ Found pyproject.toml at {pyproject_path}")
    return pyproject_path


def install_environment() -> bool:
    """
    Install the environment by using existing pyproject.toml and uv.lock (if present),
    running uv sync, and installing parent package.

    Returns
    -------
    True if installation succeeded, False otherwise
    """
    script_dir = Path(__file__).parent
    project_root = get_project_root()

    print(f"{'=' * 60}")
    print("Installing simple-baseline environment...")
    print(f"{'=' * 60}")

    # Step 1: Verify pyproject.toml exists
    try:
        verify_pyproject_toml(script_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    # Check if uv.lock exists
    uv_lock_path = script_dir / "uv.lock"
    if uv_lock_path.exists():
        print(f"✓ Found uv.lock at {uv_lock_path} (will be used by uv sync)")

    # Step 2: Run uv sync to create venv and install dependencies
    # uv sync will automatically use uv.lock if it exists
    print("\nSyncing dependencies with uv...")
    try:
        subprocess.run(
            ["uv", "sync", "--directory", str(script_dir)],
            cwd=script_dir,
            check=True,
            capture_output=False,  # Show output to user
            text=True,
        )
        print("✓ Dependencies synced successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error syncing dependencies: {e}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: uv command not found. Please install uv first.", file=sys.stderr)
        return False

    # Step 3: Verify venv was created
    venv_python = script_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print("Error: venv not found after sync", file=sys.stderr)
        return False

    # Step 4: Install parent package (reconstruction-benchmark) in the venv
    print("\nInstalling parent package (reconstruction-benchmark)...")
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
            cwd=script_dir,
            check=True,
            capture_output=False,  # Show output to user
            text=True,
        )
        print("✓ Parent package installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing parent package: {e}", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print("✓ Environment installation complete!")
    print(f"Venv location: {script_dir / '.venv'}")
    print(f"{'=' * 60}")
    return True


def main():
    """Main entry point for install script."""
    success = install_environment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
