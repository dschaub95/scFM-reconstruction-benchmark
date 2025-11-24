"""Utilities for masking gene expression data."""

import numpy as np
from anndata import AnnData


def mask_data(
    adata: AnnData,
    mask_percentage: float = 0.15,
    seed: int = 42,
    inplace: bool = False,
) -> AnnData:
    """
    Mask a percentage of gene expression values by replacing them with -1.

    Parameters
    ----------
    adata
        AnnData object with expression data in .X
    mask_percentage
        Percentage of values to mask (0.0 to 1.0)
    seed
        Random seed for reproducibility
    inplace
        Whether to modify adata in place

    Returns
    -------
    AnnData object with masked values set to -1
    """
    if not inplace:
        adata = adata.copy()

    rng = np.random.default_rng(seed)

    # Get the expression matrix
    X = adata.X

    # Create a mask for values to replace
    # Only mask non-zero values (or all values if desired)
    n_values = X.size
    n_to_mask = int(n_values * mask_percentage)

    # Get random indices to mask
    flat_indices = rng.choice(n_values, size=n_to_mask, replace=False)
    row_indices, col_indices = np.unravel_index(flat_indices, X.shape)

    # Set masked values to -1
    if isinstance(X, np.ndarray):
        X[row_indices, col_indices] = -1
    else:
        # Handle sparse matrices
        X = X.toarray()
        X[row_indices, col_indices] = -1
        adata.X = X

    return adata


def get_mask_indices(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get indices of masked values (where value is -1).

    Parameters
    ----------
    adata
        AnnData object with masked values

    Returns
    -------
    Tuple of (row_indices, col_indices) for masked positions
    """
    X = adata.X

    if hasattr(X, "toarray"):
        X = X.toarray()

    masked_rows, masked_cols = np.where(X == -1)
    return masked_rows, masked_cols
