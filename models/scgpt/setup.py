"""Setup and load scGPT model from checkpoint."""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    # Import scGPT - use the actual API structure
    from scgpt.model.model import TransformerModel as scGPT
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    SCGPT_AVAILABLE = True
except ImportError as e:
    SCGPT_AVAILABLE = False
    print(f"Warning: scGPT imports failed: {e}")


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


def find_checkpoint(
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find checkpoint file and directory.

    Parameters
    ----------
    checkpoint_path
        Optional specific checkpoint path. If None, searches for any checkpoint.

    Returns
    -------
    Tuple of (checkpoint_file, checkpoint_dir) or (None, None) if not found
    """
    checkpoints_dir = get_checkpoints_dir()

    if checkpoint_path:
        # Use specified checkpoint path
        if checkpoint_path.is_file():
            checkpoint_file = checkpoint_path
            checkpoint_dir = checkpoint_path.parent
        elif checkpoint_path.is_dir():
            checkpoint_file = checkpoint_path / "best_model.pt"
            checkpoint_dir = checkpoint_path
        else:
            return None, None

        if checkpoint_file.exists():
            return checkpoint_file, checkpoint_dir
        return None, None

    # Search for any checkpoint in the checkpoints directory
    if checkpoints_dir.exists():
        for checkpoint_subdir in checkpoints_dir.iterdir():
            if checkpoint_subdir.is_dir():
                potential_checkpoint = checkpoint_subdir / "best_model.pt"
                if potential_checkpoint.exists():
                    return potential_checkpoint, checkpoint_subdir

    return None, None


def load_vocab_from_checkpoint(checkpoint_dir: Path, gene_names: list = None):
    """
    Load vocabulary from checkpoint vocab.json or create from gene names.

    Parameters
    ----------
    checkpoint_dir
        Checkpoint directory containing vocab.json
    gene_names
        Optional list of gene names to create vocab if checkpoint vocab not available

    Returns
    -------
    Tuple of (GeneVocab, vocab_size)
    """
    gene_vocab = None
    vocab_size = None

    # Try to load from checkpoint vocab.json
    if checkpoint_dir and (checkpoint_dir / "vocab.json").exists():
        vocab_json_path = checkpoint_dir / "vocab.json"
        try:
            print(f"Loading vocabulary from checkpoint: {vocab_json_path}")
            sys.stdout.flush()

            with open(vocab_json_path) as f:
                vocab_data = json.load(f)

            # Handle different vocab.json formats
            if isinstance(vocab_data, list):
                vocab_tokens = vocab_data
            elif isinstance(vocab_data, dict):
                vocab_tokens = sorted(vocab_data.keys(), key=lambda k: vocab_data[k])
            else:
                raise ValueError(f"Unexpected vocab.json format: {type(vocab_data)}")

            # Create GeneVocab from checkpoint vocabulary
            gene_vocab = GeneVocab(vocab_tokens)
            if hasattr(gene_vocab, "set_default_index"):
                if "<pad>" in gene_vocab:
                    pad_idx = gene_vocab["<pad>"]
                else:
                    if isinstance(vocab_data, dict) and "<pad>" in vocab_data:
                        pad_idx = vocab_data["<pad>"]
                    else:
                        pad_idx = 0
                gene_vocab.set_default_index(pad_idx)
            vocab_size = len(gene_vocab)
            print(f"✓ Loaded vocabulary with {vocab_size} tokens from checkpoint")
            sys.stdout.flush()
            return gene_vocab, vocab_size
        except Exception as e:
            print(f"Warning: Could not load checkpoint vocabulary: {e}")
            print("Creating vocabulary from data gene names...")
            sys.stdout.flush()

    # Create vocabulary from gene names if checkpoint vocab not available
    if gene_vocab is None:
        if gene_names is None:
            raise RuntimeError(
                "Cannot create vocabulary: no checkpoint vocab and no gene names provided"
            )

        try:
            special_tokens = ["<pad>", "<mask>", "<cls>", "<eoc>"]
            all_tokens = special_tokens + gene_names
            gene_vocab = GeneVocab(all_tokens)
            if hasattr(gene_vocab, "set_default_index"):
                pad_idx = gene_vocab["<pad>"]
                gene_vocab.set_default_index(pad_idx)
            vocab_size = len(gene_vocab)
            print(f"Created vocabulary with {vocab_size} tokens from data")
            sys.stdout.flush()
        except Exception as e:
            print(f"Warning: Could not create gene vocabulary: {e}")
            special_tokens = ["<pad>", "<mask>", "<cls>", "<eoc>"]
            all_tokens = special_tokens + gene_names
            gene_vocab = GeneVocab(all_tokens)
            vocab_size = len(gene_vocab)

    return gene_vocab, vocab_size


def setup_model(
    checkpoint_path: Path = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    gene_names: list = None,
) -> tuple:
    """
    Load checkpoint and initialize scGPT model.

    Parameters
    ----------
    checkpoint_path
        Optional path to checkpoint. If None, searches for checkpoint automatically.
    device
        Device to load model on
    gene_names
        Optional list of gene names for vocabulary creation if checkpoint vocab not available

    Returns
    -------
    Tuple of (model, gene_vocab)
    """
    if not SCGPT_AVAILABLE:
        raise ImportError(
            "scGPT is not installed. Install with: pip install scgpt 'flash-attn<1.0.5'"
        )

    # Validate device
    if device == "cuda":
        try:
            test_tensor = torch.zeros(1).to(device)
            _ = test_tensor + 1
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: CUDA device failed validation: {e}")
            print("Falling back to CPU")
            device = "cpu"

    print(f"Using device: {device}")

    # Find checkpoint
    checkpoint_file, checkpoint_dir = find_checkpoint(checkpoint_path)

    if checkpoint_file and checkpoint_file.exists():
        print(f"\n{'=' * 60}")
        checkpoint_name = checkpoint_dir.name if checkpoint_dir else "checkpoint"
        print(f"Loading scGPT checkpoint: {checkpoint_name}")
        print(f"{'=' * 60}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"✓ Found checkpoint at: {checkpoint_file}")
        print("Will load checkpoint weights after model initialization...")
        sys.stdout.flush()
    else:
        print("No checkpoint found. Will initialize model with random weights.")
        sys.stdout.flush()

    # Load vocabulary
    gene_vocab, vocab_size = load_vocab_from_checkpoint(checkpoint_dir, gene_names)

    if vocab_size is None:
        raise RuntimeError("Vocabulary size not determined. Cannot initialize model.")

    # Setup special tokens (matching cell_emb.py pattern)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in gene_vocab:
            gene_vocab.append_token(s)

    # Set default index (matching cell_emb.py pattern)
    gene_vocab.set_default_index(gene_vocab["<pad>"])

    # Load model configs from args.json (matching cell_emb.py pattern)
    model_configs = {}
    if checkpoint_file and checkpoint_file.exists():
        model_config_file = checkpoint_file.parent / "args.json"
        if model_config_file.exists():
            try:
                with open(model_config_file, "r") as f:
                    model_configs = json.load(f)
                print("Loading model architecture from checkpoint args.json...")
                sys.stdout.flush()
            except Exception as e:
                print(f"Warning: Could not load args.json: {e}")
                print("Using default architecture parameters...")
                sys.stdout.flush()

    # Set defaults if args.json not available or missing keys
    if not model_configs:
        model_configs = {
            "embsize": 512,
            "nheads": 8,
            "d_hid": 2048,
            "nlayers": 6,
            "n_layers_cls": 3,
            "dropout": 0.1,
            "pad_value": 0,
            "pad_token": "<pad>",
            "use_fast_transformer": False,
        }
    else:
        # Ensure pad_token is set
        if "pad_token" not in model_configs:
            model_configs["pad_token"] = "<pad>"

    # Initialize model (matching cell_emb.py pattern)
    print("Initializing scGPT model architecture...")
    sys.stdout.flush()
    print(f"  Using vocabulary size: {len(gene_vocab)} tokens")
    sys.stdout.flush()

    # Determine use_fast_transformer (default to True if available, matching cell_emb.py)
    use_fast_transformer = model_configs.get("use_fast_transformer", True)

    # Determine do_mvc from checkpoint config (default to True for backward compatibility)
    do_mvc = model_configs.get("do_mvc", True)

    model = scGPT(
        ntoken=len(gene_vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=gene_vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=do_mvc,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    # Load checkpoint weights if available (matching cell_emb.py pattern)
    if checkpoint_file and checkpoint_file.exists():
        try:
            print(f"Loading pretrained weights from: {checkpoint_file}")
            sys.stdout.flush()
            load_pretrained(
                model, torch.load(checkpoint_file, map_location="cpu"), verbose=False
            )
            print("✓ Successfully loaded pretrained weights from checkpoint")
            print(f"  Checkpoint: {checkpoint_file}")
            sys.stdout.flush()
        except Exception as e:
            print(f"✗ Could not load checkpoint weights: {e}")
            print("Continuing with randomly initialized weights...")
            sys.stdout.flush()
    else:
        print("✓ Initialized new scGPT model (no pretrained checkpoint)")
        sys.stdout.flush()

    model = model.to(device)
    model.eval()

    return model, gene_vocab


def main():
    """Main entry point for setup script."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup scGPT model")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to checkpoint directory or file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified.",
    )

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, vocab = setup_model(checkpoint_path=args.checkpoint_path, device=device)
        print(f"✓ Model setup complete on device: {device}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
