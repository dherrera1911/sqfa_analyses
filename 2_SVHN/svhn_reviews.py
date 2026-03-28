import os
import torch
import numpy as np
import torchvision
from metric_learn import LFDA
from sklearn.decomposition import PCA
import sqfa
import time

import sys
sys.path.append('..')  # Add parent directory to path

from pkg_utils import (
  scale_and_center,
  train_val_split,
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
FILTERS_DIR = "filters"
LFDA_PCA_DIM = 200
LFDA_K_VALS = torch.tensor([3, 5, 9, 17])
ARTIFACT_SUFFIX = f"_pca{LFDA_PCA_DIM}"
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


def fit_preprocessing_pca(x_fit):
    n_components = min(LFDA_PCA_DIM, x_fit.shape[0], x_fit.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(torch.as_tensor(x_fit).detach().cpu().numpy())
    return pca


def transform_with_pca(pca, x):
    x_reduced = pca.transform(torch.as_tensor(x).detach().cpu().numpy())
    return torch.as_tensor(x_reduced, dtype=torch.float32)


def lift_filters_from_pca(filters, pca):
    filters = np.asarray(filters, dtype=np.float32)
    return np.asarray(np.matmul(filters, pca.components_), dtype=np.float32)


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

training_pca = fit_preprocessing_pca(x_train)
x_train_pca = transform_with_pca(training_pca, x_train)


#############################
#
# TRAIN MODELS
#
#############################
x_train_reg, y_train_reg, x_val, y_val = train_val_split(x_train, y_train, val_size=0.15)
validation_pca = fit_preprocessing_pca(x_train_reg)
x_train_reg_pca = transform_with_pca(validation_pca, x_train_reg)
x_val_pca = transform_with_pca(validation_pca, x_val)


# ------------------------------
# Train SQFA
# ------------------------------
sqfa_noise = load_or_validate_noise(
    noise_path=noise_path("sqfa"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train_pca.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
    ),
    x_train=x_train_reg_pca,
    y_train=y_train_reg,
    x_val=x_val_pca,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
    run_label="sqfa validation",
)

if not has_saved_artifacts("sqfa"):
    sqfa_filters, sqfa_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train_pca.shape[1],
            n_filters=N_FILTERS,
            feature_noise=sqfa_noise,
        ),
        x_train=x_train_pca,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        run_label="sqfa training",
    )
    sqfa_filters = lift_filters_from_pca(sqfa_filters, training_pca)
    save_artifacts("sqfa", sqfa_filters, sqfa_times)


# ------------------------------
# Train SQFA-Wasserstein
# ------------------------------
wasserstein_noise = load_or_validate_noise(
    noise_path=noise_path("wasserstein"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train_pca.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.wasserstein,
    ),
    x_train=x_train_reg_pca,
    y_train=y_train_reg,
    x_val=x_val_pca,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
    dtypes=(torch.float64,),
    run_label="wasserstein validation",
)

if not has_saved_artifacts("wasserstein"):
    wasserstein_filters, wasserstein_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train_pca.shape[1],
            n_filters=N_FILTERS,
            feature_noise=wasserstein_noise,
            distance_fun=sqfa.distances.wasserstein,
        ),
        x_train=x_train_pca,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        dtypes=(torch.float64,),
        run_label="wasserstein training",
    )
    wasserstein_filters = lift_filters_from_pca(wasserstein_filters, training_pca)
    save_artifacts("wasserstein", wasserstein_filters, wasserstein_times)


# ------------------------------
# Train SQFA-Jeffreys
# ------------------------------
jeffreys_noise = load_or_validate_noise(
    noise_path=noise_path("jeffreys"),
    model_factory=lambda noise: sqfa.model.SQFA(
        n_dim=x_train_pca.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.jeffreys,
    ),
    x_train=x_train_reg_pca,
    y_train=y_train_reg,
    x_val=x_val_pca,
    y_val=y_val,
    noise_vals=NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
    run_label="jeffreys validation",
)

if not has_saved_artifacts("jeffreys"):
    jeffreys_filters, jeffreys_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train_pca.shape[1],
            n_filters=N_FILTERS,
            feature_noise=jeffreys_noise,
            distance_fun=sqfa.distances.jeffreys,
        ),
        x_train=x_train_pca,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        run_label="jeffreys training",
    )
    jeffreys_filters = lift_filters_from_pca(jeffreys_filters, training_pca)
    save_artifacts("jeffreys", jeffreys_filters, jeffreys_times)


# ------------------------------
# Train LFDA
# ------------------------------
if not has_saved_artifacts("lfda"):
    # Select k by validation
    lfda_accs = validate_lfda_k(
        x_train=x_train,
        y_train=y_train,
        k_vals=LFDA_K_VALS,
        n_filters=N_FILTERS,
        n_pca_components=LFDA_PCA_DIM,
        eval_qda_reg=1.0e-5,
        val_size=0.15,
    )
    best_k = int(LFDA_K_VALS[torch.argmax(lfda_accs)].item())

    lfda = LFDA(n_components=N_FILTERS, k=best_k)
    start = time.time()
    lfda.fit(x_train_pca.detach().cpu().numpy(), y_train.detach().cpu().numpy())
    lfda_filters = lift_filters_from_pca(lfda.components_, training_pca)
    lfda_time = time.time() - start

    save_artifacts("lfda", lfda_filters, lfda_time)


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
    'figures/svhn_accuracies_review.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
    show_errorbars=True,
    figsize=(4,3),
)
