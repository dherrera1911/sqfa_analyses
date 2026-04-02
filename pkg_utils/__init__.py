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
    load_or_validate_lda_shrinkage,
    load_or_validate_noise,
    load_or_validate_wda_reg,
    validate_lda_shrinkage,
    validate_lfda_k,
    validate_regularization,
    validate_wda_reg,
)
from .training import (
    fit_sqfa_adaptive_precision,
    train_lfda_repeated,
    train_lmnn_repeated,
    train_metric_learn_repeated,
    train_sqfa_repeated,
)
from .suppca import SupervisedPCA
from .plot import plot_filter_grid, plot_metric_with_errorbars
from .wda import balanced_subset_indices, fit_wda
from .wdatorch import WDATorch


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
    "validate_lda_shrinkage",
    "load_or_validate_lda_shrinkage",
    "validate_wda_reg",
    "load_or_validate_wda_reg",
    "fit_sqfa_adaptive_precision",
    "train_metric_learn_repeated",
    "train_sqfa_repeated",
    "train_lfda_repeated",
    "train_lmnn_repeated",
    "fit_wda",
    "balanced_subset_indices",
    "WDATorch",
    "SupervisedPCA",
    "plot_filter_grid",
    "plot_metric_with_errorbars",
]
