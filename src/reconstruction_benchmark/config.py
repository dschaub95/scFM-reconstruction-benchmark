"""Configuration schemas and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data preparation."""
    dataset_name: str
    mask_percentage: float = 0.15
    seed: int = 42
    data_dir: Path = Path("data")
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set default output directory if not provided."""
        if self.output_dir is None:
            self.output_dir = self.data_dir / self.dataset_name


@dataclass
class ModelConfig:
    """Configuration for model execution."""
    model_name: str
    dataset_name: str
    mask_percentage: float = 0.15
    seed: int = 42
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    
    @property
    def masked_data_path(self) -> Path:
        """Path to masked dataset."""
        mask_pct_value = self.mask_percentage * 100
        mask_pct_str = f"{mask_pct_value:g}"
        return self.data_dir / self.dataset_name / "masked" / f"mask_{mask_pct_str}_seed_{self.seed}.h5ad"
    
    @property
    def raw_data_path(self) -> Path:
        """Path to raw dataset."""
        return self.data_dir / self.dataset_name / "raw" / "data.h5ad"
    
    @property
    def experiment_dir(self) -> Path:
        """Directory for this experimental condition (mask percentage + seed)."""
        mask_pct_value = self.mask_percentage * 100
        mask_pct_str = f"{mask_pct_value:g}"
        return self.results_dir / self.dataset_name / f"mask_{mask_pct_str}_seed_{self.seed}"
    
    @property
    def output_dir(self) -> Path:
        """Output directory for model results."""
        return self.experiment_dir / self.model_name


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    model_name: str
    dataset_name: str
    mask_percentage: float = 0.15
    seed: int = 42
    results_dir: Path = Path("results")
    evaluation_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set default evaluation directory if not provided."""
        if self.evaluation_dir is None:
            # Put metrics.json in the same directory as reconstruction.h5ad
            mask_pct_value = self.mask_percentage * 100
            mask_pct_str = f"{mask_pct_value:g}"
            experiment_dir = self.results_dir / self.dataset_name / f"mask_{mask_pct_str}_seed_{self.seed}"
            self.evaluation_dir = experiment_dir / self.model_name

