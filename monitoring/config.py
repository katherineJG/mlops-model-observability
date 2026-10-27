"""Monitoring thresholds. Defaults are the initial values in docs/ARCHITECTURE.md section 3."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MonitoringConfig:
    window_hours: float = 1.0
    min_samples: int = 200  # windows with fewer rows are reported as insufficient_data
    alpha: float = 0.01  # family-wise significance level for per-feature drift tests
    drift_share_threshold: float = 0.3  # share of drifted features that raises the window flag
    range_warn_fraction: float = 0.01
    unseen_warn_fraction: float = 0.01
    missing_warn_fraction: float = 0.05
    retention_days: int = 7
    # Calibrated on healthy synthetic windows (docs/ARCHITECTURE.md 3.7): at 300 labels about 1%
    # of healthy windows exceed a 0.08 drop. Smaller degradations are not flagged.
    min_labels: int = 300  # below this many labelled rows no performance verdict is given
    auc_drop_threshold: float = 0.08  # ROC AUC this far below the reference baseline is degraded

    def __post_init__(self) -> None:
        if self.window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        for name in ("drift_share_threshold", "range_warn_fraction", "unseen_warn_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.missing_warn_fraction <= 1:
            raise ValueError("missing_warn_fraction must be between 0 and 1")
        if self.retention_days < 1:
            raise ValueError("retention_days must be at least 1")
        if self.min_labels < 2:
            raise ValueError("min_labels must be at least 2")
        if not 0 < self.auc_drop_threshold < 1:
            raise ValueError("auc_drop_threshold must be between 0 and 1")
