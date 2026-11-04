"""Probability-model evaluation (proper scoring rules, calibration, uncertainty) and report helpers."""

from pipeline.evaluation.metrics import (
    bin_index, calibration_bins, calibration_slope_intercept, compare_models, disagreement_analysis,
    expected_calibration_error, paired_bootstrap, per_shot_log_loss, plot_calibration, prob_metrics, subgroup_table,
)

__all__ = [
    "bin_index", "calibration_bins", "calibration_slope_intercept", "compare_models", "disagreement_analysis",
    "expected_calibration_error", "paired_bootstrap", "per_shot_log_loss", "plot_calibration", "prob_metrics",
    "subgroup_table",
]
