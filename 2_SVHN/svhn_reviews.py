import os
import torch
import numpy as np
import torchvision
import sqfa

import sys
sys.path.append('..')  # Add parent directory to path

from pkg_utils import (
  scale_and_center,
  train_val_split,
  train_lfda_repeated,
  train_sqfa_repeated,
  qda_accuracy,
  collect_metric_across_runs,
  load_or_validate_noise,
  validate_lfda_k,
  plot_metric_with_errorbars,
)

N_FILTERS = 9
NOISE_VALS = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
N_REPS =5
FILTERS_DIR = "filters_review"
SQFA_FIT_KWARGS = {"max_epochs": 500, "show_progress": False}
LFDA_PCA_DIM = 100
LFDA_K_VALS = torch.tensor([3, 5, 9, 17])
LFDA_EMBEDDING_TYPE = "orthonormalized"
ARTIFACT_SUFFIX = f"review"
torch.manual_seed(2)

os.makedirs(FILTERS_DIR, exist_ok=True)


def filter_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_filters{ARTIFACT_SUFFIX}.npy"


def time_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_time{ARTIFACT_SUFFIX}.npy"


def noise_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_noise{ARTIFACT_SUFFIX}.npy"


def has_saved_artifacts(model_key):
    return os.path.exists(filter_path(model_key)) and os.path.exists(time_path(model_key))


def save_artifacts(model_key, filters, times):
    np.save(filter_path(model_key), np.asarray(filters))
    np.save(time_path(model_key), np.asarray(times))

#############################
#
# LOAD AND PROCESS DATA
#
#############################

# Download and load training and test datasets
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

# Convert to PyTorch tensors, average channels and reshape
n_samples, n_channels, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float()
x_train = x_train.mean(dim=1).reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.labels, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float()
x_test = x_test.mean(dim=1).reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.labels, dtype=torch.long)

x_train, x_test = scale_and_center(x_train, x_test)


#############################
#
# TRAIN MODELS
#
#############################
x_train_reg, y_train_reg, x_val, y_val = train_val_split(x_train, y_train, val_size=0.15)


# ------------------------------
# Train SQFA
# ------------------------------
sqfa_noise = load_or_validate_noise(
    noise_path=noise_path("sqfa"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
    ),
    x_train=x_train_reg,
    y_train=y_train_reg,
    x_val=x_val,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs=SQFA_FIT_KWARGS,
    run_label="sqfa validation",
)

if not has_saved_artifacts("sqfa"):
    sqfa_filters, sqfa_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=sqfa_noise,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        run_label="sqfa training",
    )
    save_artifacts("sqfa", sqfa_filters, sqfa_times)


# ------------------------------
# Train SQFA-Wasserstein
# ------------------------------
wasserstein_noise = load_or_validate_noise(
    noise_path=noise_path("wasserstein"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.wasserstein,
        constraint="orthogonal",
    ),
    x_train=x_train_reg,
    y_train=y_train_reg,
    x_val=x_val,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs=SQFA_FIT_KWARGS,
    dtypes=(torch.float64,),
    run_label="wasserstein validation",
)

if not has_saved_artifacts("wasserstein"):
    wasserstein_filters, wasserstein_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=wasserstein_noise,
            distance_fun=sqfa.distances.wasserstein,
            constraint="orthogonal",
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        dtypes=(torch.float64,),
        run_label="wasserstein training",
    )
    save_artifacts("wasserstein", wasserstein_filters, wasserstein_times)


# ------------------------------
# Train SQFA-Jeffreys
# ------------------------------
jeffreys_noise = load_or_validate_noise(
    noise_path=noise_path("jeffreys"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.jeffreys,
    ),
    x_train=x_train_reg,
    y_train=y_train_reg,
    x_val=x_val,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs=SQFA_FIT_KWARGS,
    run_label="jeffreys validation",
)

if not has_saved_artifacts("jeffreys"):
    jeffreys_filters, jeffreys_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=jeffreys_noise,
            distance_fun=sqfa.distances.jeffreys,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs=SQFA_FIT_KWARGS,
        run_label="jeffreys training",
    )
    save_artifacts("jeffreys", jeffreys_filters, jeffreys_times)


# ------------------------------
# Train LFDA
# ------------------------------
if not has_saved_artifacts("lfda"):
    lfda_accs = validate_lfda_k(
        x_train=x_train,
        y_train=y_train,
        k_vals=LFDA_K_VALS,
        n_filters=N_FILTERS,
        n_pca_components=LFDA_PCA_DIM,
        eval_qda_reg=1.0e-5,
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
    save_artifacts("lfda", lfda_filters, lfda_times)


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
model_filters = [np.load(filter_path(model_key)) for model_key in model_keys]

qda_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: qda_accuracy(
            x_train,
            y_train,
            x_test,
            y_test,
            filt,
            eval_qda_noise=0.00,
            eval_qda_reg=1.0e-5
        ).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_scores,
    "QDA Accuracy (%)",
    'figures_review/svhn_accuracies_review.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
    show_errorbars=True,
    figsize=(4,3),
)
