"""Shared utility exports used across the paper experiments and analyses."""

from .data import scale_and_center, train_val_split
from .pca import fit_preprocessing_pca, transform_with_pca, lift_filters_from_pca
from .evaluation import (
    qda_accuracy,
    qda_accuracy_gaussian,
    knn_accuracy,
    collect_metric_across_runs,
)
from .artifacts import artifact_path, has_saved_artifacts, load_cached_filters, save_training_artifacts
from .regularization import (
    load_or_validate_noise,
    validate_lfda_k,
    validate_regularization,
)
from .training import fit_sqfa_adaptive_precision, train_sqfa_repeated
from .suppca import SupervisedPCA
from .plot import plot_filter_grid, plot_metric_with_errorbars


__all__ = [
    "scale_and_center",
    "train_val_split",
    "fit_preprocessing_pca",
    "transform_with_pca",
    "lift_filters_from_pca",
    "qda_accuracy",
    "qda_accuracy_gaussian",
    "knn_accuracy",
    "artifact_path",
    "load_cached_filters",
    "has_saved_artifacts",
    "save_training_artifacts",
    "validate_regularization",
    "validate_lfda_k",
    "load_or_validate_noise",
    "fit_sqfa_adaptive_precision",
    "train_sqfa_repeated",
    "balanced_subset_indices",
    "SupervisedPCA",
    "plot_filter_grid",
    "plot_metric_with_errorbars",
]
