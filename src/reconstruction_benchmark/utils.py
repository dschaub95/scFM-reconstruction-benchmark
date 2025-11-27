"""General utility functions."""

from pathlib import Path
from typing import Optional


def get_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Get the project root directory by searching for .gitignore file.

    Iteratively searches parent directories starting from start_path (or current
    working directory if not provided) while .gitignore does not exist in the
    current directory, stopping when .gitignore is found (indicating project root).

    Parameters
    ----------
    start_path
        Starting directory for the search. If None, uses current working directory.

    Returns
    -------
    Path to the project root directory

    Raises
    ------
    RuntimeError
        If .gitignore is not found in any parent directory
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()

    current = start_path
    gitignore_path = current / ".gitignore"

    # Iterate while .gitignore does not exist in the current directory
    while not gitignore_path.exists() and current != current.parent:
        current = current.parent
        gitignore_path = current / ".gitignore"

    # Check if we found .gitignore (project root)
    if gitignore_path.exists():
        return current

    raise RuntimeError(
        f"Could not find project root (no .gitignore found starting from {start_path})"
    )
