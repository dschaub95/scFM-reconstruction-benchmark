"""Download checkpoint for scGPT model."""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_checkpoints_dir() -> Path:
    """
    Get the checkpoints directory for scGPT models.

    Returns
    -------
    Path to the checkpoints directory (models/scgpt/checkpoints/)
    """
    scgpt_dir = Path(__file__).parent
    checkpoints_dir = scgpt_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def get_local_checkpoint_path(model_name: str) -> Path:
    """
    Get the local checkpoint path for a given model name.

    Parameters
    ----------
    model_name
        Model name (e.g., "bwanglab/scGPT" or "scGPT-human")

    Returns
    -------
    Path to the local checkpoint directory
    """
    checkpoints_dir = get_checkpoints_dir()
    # Convert HuggingFace model ID to a safe directory name
    checkpoint_name = model_name.replace("/", "_")
    checkpoint_path = checkpoints_dir / checkpoint_name
    return checkpoint_path


def download_checkpoint_from_gdrive(
    gdrive_folder_id: str,
    output_dir: Path,
    quiet: bool = False,
) -> bool:
    """
    Download a checkpoint folder from Google Drive.

    Parameters
    ----------
    gdrive_folder_id
        Google Drive folder ID (from the shareable link)
    output_dir
        Directory where the checkpoint files will be saved
    quiet
        If True, suppress output messages

    Returns
    -------
    True if download succeeded, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Error: gdown is not installed. Install with: pip install gdown")
        return False

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct Google Drive folder URL
    folder_url = f"https://drive.google.com/drive/folders/{gdrive_folder_id}"

    try:
        if not quiet:
            print(f"Downloading checkpoint from Google Drive: {folder_url}")
            print(f"Saving to: {output_dir}")

        # Download the entire folder
        # gdown.download_folder downloads all files in the folder
        gdown.download_folder(
            folder_url,
            output=str(output_dir),
            quiet=quiet,
            use_cookies=False,
        )

        # Verify that essential files were downloaded
        required_files = ["best_model.pt"]
        missing_files = [f for f in required_files if not (output_dir / f).exists()]

        if missing_files:
            print(f"Warning: Some required files are missing: {missing_files}")
            # Check if files are in subdirectories (gdown sometimes creates subdirs)
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    for file in required_files:
                        if (subdir / file).exists() and not (
                            output_dir / file
                        ).exists():
                            # Move file to output_dir
                            import shutil

                            try:
                                shutil.move(str(subdir / file), str(output_dir / file))
                                if not quiet:
                                    print(f"Moved {file} from {subdir} to {output_dir}")
                            except Exception as e:
                                print(
                                    f"Error moving {file} from {subdir} to {output_dir}: {e}",
                                    file=sys.stderr,
                                )

        # Re-verify that all required files are now present after move operations
        missing_files = [f for f in required_files if not (output_dir / f).exists()]
        if missing_files:
            print(
                f"Error: Required files are still missing after download: {missing_files}",
                file=sys.stderr,
            )
            return False

        if not quiet:
            print(f"✓ Successfully downloaded checkpoint to: {output_dir}")
            sys.stdout.flush()
        return True

    except Exception as e:
        print(f"Error downloading checkpoint from Google Drive: {e}")
        return False


def download_checkpoint(
    checkpoint_name: str = "scGPT-human",
    gdrive_folder_id: str = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
    force_download: bool = False,
) -> Path:
    """
    Download the scGPT checkpoint from Google Drive to local checkpoints directory.

    This function downloads the official scGPT checkpoint from the Google Drive folder
    containing best_model.pt, vocab.json, and args.json.

    Parameters
    ----------
    checkpoint_name
        Name for the checkpoint (used as directory name)
    gdrive_folder_id
        Google Drive folder ID (default: official scGPT checkpoint folder)
    force_download
        If True, re-download even if checkpoint already exists

    Returns
    -------
    Path to the downloaded checkpoint directory

    Raises
    ------
    RuntimeError
        If download fails
    """
    checkpoint_path = get_local_checkpoint_path(checkpoint_name)

    # Check if checkpoint already exists
    if checkpoint_path.exists() and (checkpoint_path / "best_model.pt").exists():
        if not force_download:
            print(f"✓ Checkpoint already exists at: {checkpoint_path}")
            print("Skipping download. Use force_download=True to re-download.")
            sys.stdout.flush()
            return checkpoint_path
        else:
            print("Force download enabled. Re-downloading checkpoint...")

    # Download from Google Drive
    success = download_checkpoint_from_gdrive(
        gdrive_folder_id=gdrive_folder_id,
        output_dir=checkpoint_path,
        quiet=False,
    )

    if success:
        return checkpoint_path
    else:
        raise RuntimeError(
            f"Failed to download checkpoint from Google Drive to {checkpoint_path}"
        )


def ensure_checkpoint(
    checkpoint_path: Optional[Path] = None,
    skip_download: bool = False,
    checkpoint_name: str = "scGPT-human",
    gdrive_folder_id: str = "1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
) -> Optional[Path]:
    """
    Ensure checkpoint is available, checking for existing or downloading if needed.

    This function encapsulates all checkpoint handling logic: checking for existing
    checkpoints, downloading if needed, and handling errors gracefully.

    Parameters
    ----------
    checkpoint_path
        Optional specific checkpoint path. If provided, returns it if valid.
    skip_download
        If True, skip download attempt and only check for existing checkpoint
    checkpoint_name
        Name for the checkpoint (used as directory name if downloading)
    gdrive_folder_id
        Google Drive folder ID (default: official scGPT checkpoint folder)

    Returns
    -------
    Path to checkpoint directory, or None if checkpoint unavailable but should continue

    Raises
    ------
    RuntimeError
        If checkpoint is required but unavailable and download fails
    """
    # If specific path provided, validate and return it
    if checkpoint_path is not None:
        if checkpoint_path.is_file():
            checkpoint_file = checkpoint_path
            checkpoint_dir = checkpoint_path.parent
        elif checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "best_model.pt"
            checkpoint_dir = checkpoint_path
        else:
            print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
            print("Will try to find or download checkpoint...")
            checkpoint_path = None

        if checkpoint_path and checkpoint_file.exists():
            print(f"✓ Using provided checkpoint at: {checkpoint_dir}")
            return checkpoint_dir

    # Check for existing checkpoint in checkpoints directory
    checkpoints_dir = get_checkpoints_dir()
    if checkpoints_dir.exists():
        for checkpoint_subdir in checkpoints_dir.iterdir():
            if checkpoint_subdir.is_dir():
                potential_checkpoint = checkpoint_subdir / "best_model.pt"
                if potential_checkpoint.exists():
                    print(f"✓ Found existing checkpoint at: {checkpoint_subdir}")
                    return checkpoint_subdir

    # If skip_download is True, return None (checkpoint unavailable but should continue)
    if skip_download:
        print("Skipping checkpoint download (--skip-download flag set)")
        return None

    # Try downloading checkpoint
    try:
        print("No local checkpoint found. Downloading from Google Drive...")
        downloaded_path = download_checkpoint(
            checkpoint_name=checkpoint_name,
            gdrive_folder_id=gdrive_folder_id,
            force_download=False,
        )
        print(f"✓ Checkpoint downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Warning: Could not download checkpoint: {e}")
        print("Will try to proceed without checkpoint...")
        return None


def main():
    """Main entry point for download_ckpt script."""
    import argparse

    parser = argparse.ArgumentParser(description="Download scGPT checkpoint")
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="scGPT-human",
        help="Name for the checkpoint (default: scGPT-human)",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        type=str,
        default="1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y",
        help="Google Drive folder ID (default: official scGPT checkpoint)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if checkpoint exists",
    )

    args = parser.parse_args()

    try:
        checkpoint_path = download_checkpoint(
            checkpoint_name=args.checkpoint_name,
            gdrive_folder_id=args.gdrive_folder_id,
            force_download=args.force,
        )
        print(f"✓ Checkpoint available at: {checkpoint_path}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
