"""Configuration schemas and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data preparation."""

    dataset_name: Optional[str]
    mask_percentage: float = 0.15
    seed: int = 42
    data_dir: Path = Path("data")
    output_dir: Optional[Path] = None

    def __post_init__(self):
        """Set default output directory if not provided."""
        if self.output_dir is None and self.dataset_name is not None:
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
        from .data.utils import format_mask_percentage

        mask_pct_str = format_mask_percentage(self.mask_percentage)
        return (
            self.data_dir
            / self.dataset_name
            / "masked"
            / f"mask_{mask_pct_str}_seed_{self.seed}.h5ad"
        )

    @property
    def raw_data_path(self) -> Path:
        """Path to raw dataset (auto-discovered)."""
        from .data.utils import discover_raw_data_file

        raw_dir = self.data_dir / self.dataset_name / "raw"
        discovered_path = discover_raw_data_file(raw_dir, self.dataset_name)

        if discovered_path is not None:
            return discovered_path

        raise FileNotFoundError(
            f"No .h5ad files found in raw data directory: {raw_dir}"
        )

    @property
    def experiment_dir(self) -> Path:
        """Directory for this experimental condition (mask percentage + seed)."""
        from .data.utils import format_mask_percentage

        mask_pct_str = format_mask_percentage(self.mask_percentage)
        return (
            self.results_dir
            / self.model_name
            / self.dataset_name
            / f"mask_{mask_pct_str}_seed_{self.seed}"
        )

    @property
    def output_dir(self) -> Path:
        """Output directory for model results."""
        return self.experiment_dir


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
            from .data.utils import format_mask_percentage

            mask_pct_str = format_mask_percentage(self.mask_percentage)
            self.evaluation_dir = (
                self.results_dir
                / self.model_name
                / self.dataset_name
                / f"mask_{mask_pct_str}_seed_{self.seed}"
            )
