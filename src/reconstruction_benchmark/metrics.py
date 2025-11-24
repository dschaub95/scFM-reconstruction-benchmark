"""Metrics registry for evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from anndata import AnnData
from scipy.stats import pearsonr


class Metric(ABC):
    """Base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass
    
    @abstractmethod
    def compute(
        self,
        ground_truth: AnnData,
        reconstruction: AnnData,
        mask_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> float:
        """
        Compute the metric value.
        
        Parameters
        ----------
        ground_truth
            Original data before masking
        reconstruction
            Reconstructed data from model
        mask_indices
            Optional tuple of (row_indices, col_indices) for masked positions
        
        Returns
        -------
        Metric value
        """
        pass


class MSE(Metric):
    """Mean Squared Error."""
    
    @property
    def name(self) -> str:
        return "mse"
    
    def compute(
        self,
        ground_truth: AnnData,
        reconstruction: AnnData,
        mask_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> float:
        """Compute MSE on masked positions."""
        gt_X = ground_truth.X
        rec_X = reconstruction.X
        
        if hasattr(gt_X, 'toarray'):
            gt_X = gt_X.toarray()
        if hasattr(rec_X, 'toarray'):
            rec_X = rec_X.toarray()
        
        if mask_indices is not None:
            row_idx, col_idx = mask_indices
            gt_values = gt_X[row_idx, col_idx]
            rec_values = rec_X[row_idx, col_idx]
        else:
            gt_values = gt_X.flatten()
            rec_values = rec_X.flatten()
        
        return float(np.mean((gt_values - rec_values) ** 2))


class MAE(Metric):
    """Mean Absolute Error."""
    
    @property
    def name(self) -> str:
        return "mae"
    
    def compute(
        self,
        ground_truth: AnnData,
        reconstruction: AnnData,
        mask_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> float:
        """Compute MAE on masked positions."""
        gt_X = ground_truth.X
        rec_X = reconstruction.X
        
        if hasattr(gt_X, 'toarray'):
            gt_X = gt_X.toarray()
        if hasattr(rec_X, 'toarray'):
            rec_X = rec_X.toarray()
        
        if mask_indices is not None:
            row_idx, col_idx = mask_indices
            gt_values = gt_X[row_idx, col_idx]
            rec_values = rec_X[row_idx, col_idx]
        else:
            gt_values = gt_X.flatten()
            rec_values = rec_X.flatten()
        
        return float(np.mean(np.abs(gt_values - rec_values)))


class PearsonCorrelation(Metric):
    """Pearson correlation coefficient."""
    
    @property
    def name(self) -> str:
        return "pearson_correlation"
    
    def compute(
        self,
        ground_truth: AnnData,
        reconstruction: AnnData,
        mask_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> float:
        """Compute Pearson correlation on masked positions."""
        gt_X = ground_truth.X
        rec_X = reconstruction.X
        
        if hasattr(gt_X, 'toarray'):
            gt_X = gt_X.toarray()
        if hasattr(rec_X, 'toarray'):
            rec_X = rec_X.toarray()
        
        if mask_indices is not None:
            row_idx, col_idx = mask_indices
            gt_values = gt_X[row_idx, col_idx]
            rec_values = rec_X[row_idx, col_idx]
        else:
            gt_values = gt_X.flatten()
            rec_values = rec_X.flatten()
        
        corr, _ = pearsonr(gt_values, rec_values)
        return float(corr)


class MetricRegistry:
    """Registry for available metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default metrics."""
        self.register(MSE())
        self.register(MAE())
        self.register(PearsonCorrelation())
    
    def register(self, metric: Metric):
        """Register a new metric."""
        self._metrics[metric.name] = metric
    
    def get(self, name: str) -> Metric:
        """Get a metric by name."""
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found. Available: {list(self._metrics.keys())}")
        return self._metrics[name]
    
    def list_all(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())
    
    def compute_all(
        self,
        ground_truth: AnnData,
        reconstruction: AnnData,
        mask_indices: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Compute all registered metrics."""
        results = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(ground_truth, reconstruction, mask_indices)
            except Exception as e:
                results[name] = None
                print(f"Warning: Failed to compute {name}: {e}")
        return results


# Global registry instance
_registry = MetricRegistry()


def get_registry() -> MetricRegistry:
    """Get the global metric registry."""
    return _registry

