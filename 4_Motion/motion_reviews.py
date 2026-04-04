import os
import sys

import numpy as np
import torch

import sqfa
from functions import load_data, normalize_stim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

sys.path.append("..")  # Add parent directory to path
from pkg_utils import (
    artifact_path,
    collect_metric_across_runs,
    has_saved_artifacts,
    load_cached_filters,
    plot_metric_with_errorbars,
    qda_accuracy,
    save_training_artifacts,
    train_lfda_repeated,
    train_sqfa_repeated,
    validate_lfda_k,
)


N_FILTERS = 8
RESPONSE_NOISE = 0.001
C50 = 0.8
N_REPS = 5
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters_review")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures_review")
SQFA_FIT_KWARGS = {
    "max_epochs": 300,
    "show_progress": False,
    "pairwise": True,
    "estimator": "empirical",
}
LFDA_PCA_DIM = 100
LFDA_K_VALS = torch.tensor([3, 5, 9, 17])
LFDA_EMBEDDING_TYPE = "orthonormalized"
WASSERSTEIN_DTYPES = (torch.float64,)
EVAL_QDA_REG = 1.0e-5
torch.manual_seed(2)

os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def ensure_fixed_scalar(path, value, description):
    """Persist a fixed scalar artifact, overwriting stale cached values."""
    fixed_value = float(value)
    if os.path.exists(path):
        cached_value = float(np.load(path).item())
        if np.isclose(cached_value, fixed_value):
            return fixed_value
        print(
            f"Overwriting cached {description}: "
            f"{cached_value} -> {fixed_value}"
        )
    np.save(path, np.asarray(fixed_value))
    return fixed_value


#############################
#
# LOAD AND PROCESS DATA
#
#############################

x_train_raw, y_train, _ = load_data("train")
x_test_raw, y_test, _ = load_data("test")

x_train = normalize_stim(x_train_raw, C50).to(dtype=torch.float32)
x_test = normalize_stim(x_test_raw, C50).to(dtype=torch.float32)
y_train = y_train.to(dtype=torch.long)
y_test = y_test.to(dtype=torch.long)


#############################
#
# TRAIN MODELS
#
#############################

sqfa_filter_path = artifact_path(FILTERS_DIR, "sqfa", "filters")
sqfa_time_path = artifact_path(FILTERS_DIR, "sqfa", "time")
wasserstein_filter_path = artifact_path(FILTERS_DIR, "wasserstein", "filters")
wasserstein_time_path = artifact_path(FILTERS_DIR, "wasserstein", "time")
jeffreys_filter_path = artifact_path(FILTERS_DIR, "jeffreys", "filters")
jeffreys_time_path = artifact_path(FILTERS_DIR, "jeffreys", "time")
lfda_filter_path = artifact_path(FILTERS_DIR, "lfda", "filters")
lfda_time_path = artifact_path(FILTERS_DIR, "lfda", "time")

# ------------------------------
# Train SQFA
# ------------------------------
if not has_saved_artifacts(sqfa_filter_path, sqfa_time_path):
    sqfa_filters, sqfa_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        run_label="sqfa training",
    )
    save_training_artifacts(
        sqfa_filter_path,
        sqfa_time_path,
        sqfa_filters,
        sqfa_times,
    )
else:
    load_cached_filters(sqfa_filter_path, description="sqfa filters")


# ------------------------------
# Train SQFA-Wasserstein
# ------------------------------
if not has_saved_artifacts(wasserstein_filter_path, wasserstein_time_path):
    wasserstein_filters, wasserstein_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
            distance_fun=sqfa.distances.wasserstein,
            constraint="orthogonal",
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        dtypes=WASSERSTEIN_DTYPES,
        run_label="wasserstein training",
    )
    save_training_artifacts(
        wasserstein_filter_path,
        wasserstein_time_path,
        wasserstein_filters,
        wasserstein_times,
    )
else:
    load_cached_filters(wasserstein_filter_path, description="wasserstein filters")


# ------------------------------
# Train SQFA-Jeffreys
# ------------------------------
if not has_saved_artifacts(jeffreys_filter_path, jeffreys_time_path):
    jeffreys_filters, jeffreys_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
            distance_fun=sqfa.distances.jeffreys,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        run_label="jeffreys training",
    )
    save_training_artifacts(
        jeffreys_filter_path,
        jeffreys_time_path,
        jeffreys_filters,
        jeffreys_times,
    )
else:
    load_cached_filters(jeffreys_filter_path, description="jeffreys filters")


# ------------------------------
# Train LFDA
# ------------------------------
if not has_saved_artifacts(lfda_filter_path, lfda_time_path):
    lfda_accs = validate_lfda_k(
        x_train=x_train,
        y_train=y_train,
        k_vals=LFDA_K_VALS,
        n_filters=N_FILTERS,
        n_pca_components=LFDA_PCA_DIM,
        eval_qda_reg=EVAL_QDA_REG,
        val_size=0.15,
        embedding_type=LFDA_EMBEDDING_TYPE,
    )
    best_k = int(LFDA_K_VALS[torch.argmax(lfda_accs)].item())

    lfda_filters, lfda_times = train_lfda_repeated(
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        n_filters=N_FILTERS,
        k=best_k,
        n_pca_components=LFDA_PCA_DIM,
        embedding_type=LFDA_EMBEDDING_TYPE,
        run_label="lfda training",
    )
    save_training_artifacts(
        lfda_filter_path,
        lfda_time_path,
        lfda_filters,
        lfda_times,
    )
else:
    load_cached_filters(lfda_filter_path, description="lfda filters")


#############################
#
# PLOT QDA ACCURACIES
#
#############################

model_names = [
    "SQFA",
    "SQFA-W",
    "SQFA-J",
    "LFDA",
]

model_keys = ["sqfa", "wasserstein", "jeffreys", "lfda"]
model_filters = [
    np.load(artifact_path(FILTERS_DIR, model_key, "filters"))
    for model_key in model_keys
]

qda_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: qda_accuracy(
            x_train,
            y_train,
            x_test,
            y_test,
            filt,
            eval_qda_noise=0.0,
            eval_qda_reg=EVAL_QDA_REG,
        ).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_scores,
    "QDA Accuracy (%)",
    os.path.join(FIGURES_DIR, "motion_accuracies_review.pdf"),
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
    show_errorbars=True,
    figsize=(4, 3),
)
