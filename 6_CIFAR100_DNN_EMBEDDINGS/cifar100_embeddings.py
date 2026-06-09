import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sqfa_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sqfa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pkg_utils import (
    SupervisedPCA,
    collect_metric_across_runs,
    knn_accuracy,
    plot_metric_with_errorbars,
    qda_accuracy,
    scale_and_center,
    train_sqfa_repeated,
    train_val_split,
    validate_regularization,
)


N_FILTERS = 16
NOISE_VALS = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
N_REPS = 5
FILTERS_DIR = os.path.join(SCRIPT_DIR, "filters")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "embeddings")
LDA_SHRINKAGE = 0.8
torch.manual_seed(5)

os.makedirs(FILTERS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def filter_path(model_key):
    return os.path.join(FILTERS_DIR, f"{model_key}_filters.npy")


def time_path(model_key):
    return os.path.join(FILTERS_DIR, f"{model_key}_time.npy")


def noise_path(model_key):
    return os.path.join(FILTERS_DIR, f"{model_key}_noise.npy")


def has_saved_artifacts(model_key):
    return os.path.exists(filter_path(model_key)) and os.path.exists(time_path(model_key))


def save_artifacts(model_key, filters, times):
    np.save(filter_path(model_key), np.asarray(filters))
    np.save(time_path(model_key), np.asarray(times))


def load_embeddings(split_name):
    split_path = os.path.join(EMBEDDINGS_DIR, f"cifar100_resnet18_{split_name}.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing {split_path}. Run extract_embeddings.py first."
        )

    saved = torch.load(split_path, map_location="cpu")
    x = torch.as_tensor(saved["X"], dtype=torch.float32)
    y = torch.as_tensor(saved["y"], dtype=torch.long)
    return x, y


def get_or_tune_noise(
    model_key,
    model_factory,
    x_train_reg,
    y_train_reg,
    x_val,
    y_val,
    noise_vals,
    fit_kwargs=None,
    dtypes=(torch.float32, torch.float64),
):
    current_noise_path = noise_path(model_key)
    if os.path.exists(current_noise_path):
        return float(np.load(current_noise_path).item())

    noise_accs = validate_regularization(
        model_factory,
        x_train_reg,
        y_train_reg,
        x_val,
        y_val,
        noise_vals,
        fit_kwargs=fit_kwargs,
        dtypes=dtypes,
        run_label=f"{model_key} validation",
    )
    best_noise = float(noise_vals[torch.argmax(noise_accs)].item())
    np.save(current_noise_path, np.asarray(best_noise))
    return best_noise


#############################
#
# LOAD AND PROCESS DATA
#
#############################

x_train, y_train = load_embeddings("train")
x_test, y_test = load_embeddings("test")
x_train, x_test = scale_and_center(x_train, x_test)


#############################
#
# TRAIN MODELS
#
#############################

x_train_reg, y_train_reg, x_val, y_val = train_val_split(
    x_train,
    y_train,
    val_size=0.15,
)


# ------------------------------
# Train PCA
# ------------------------------
if not has_saved_artifacts("pca"):
    pca = PCA(n_components=N_FILTERS)
    start = time.time()
    pca.fit(x_train)
    pca_time = time.time() - start
    save_artifacts("pca", pca.components_, pca_time)


# ------------------------------
# Train Supervised PCA
# ------------------------------
if not has_saved_artifacts("spca"):
    x_subsampled = x_train[::5]
    y_subsampled = y_train[::5]
    spca = SupervisedPCA(n_components=N_FILTERS, label_kernel="delta")
    start = time.time()
    spca.fit(x_subsampled, y_subsampled)
    spca_time = time.time() - start
    save_artifacts("spca", spca.components_, spca_time)


# ------------------------------
# Train smSQFA
# ------------------------------
smsqfa_noise = get_or_tune_noise(
    "smsqfa",
    lambda noise: sqfa.model.SecondMomentsSQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
    ),
    x_train_reg,
    y_train_reg,
    x_val,
    y_val,
    NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
)

if not has_saved_artifacts("smsqfa"):
    smsqfa_filters, smsqfa_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SecondMomentsSQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=smsqfa_noise,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        run_label="smsqfa training",
    )
    save_artifacts("smsqfa", smsqfa_filters, smsqfa_times)


# ------------------------------
# Train SQFA
# ------------------------------
sqfa_noise = get_or_tune_noise(
    "sqfa",
    lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
    ),
    x_train_reg,
    y_train_reg,
    x_val,
    y_val,
    NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
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
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        run_label="sqfa training",
    )
    save_artifacts("sqfa", sqfa_filters, sqfa_times)


# ------------------------------
# Train Bhattacharyya
# ------------------------------
bhattacharyya_noise = get_or_tune_noise(
    "bhattacharyya",
    lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.bhattacharyya,
    ),
    x_train_reg,
    y_train_reg,
    x_val,
    y_val,
    NOISE_VALS,
    fit_kwargs={"lr": 0.2, "max_epochs": 500, "show_progress": False},
)

if not has_saved_artifacts("bhattacharyya"):
    bhattacharyya_filters, bhattacharyya_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=bhattacharyya_noise,
            distance_fun=sqfa.distances.bhattacharyya,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"lr": 0.2, "max_epochs": 500, "show_progress": False},
        run_label="bhattacharyya training",
    )
    save_artifacts("bhattacharyya", bhattacharyya_filters, bhattacharyya_times)


# ------------------------------
# Train Hellinger
# ------------------------------
hellinger_noise = get_or_tune_noise(
    "hellinger",
    lambda noise: sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=noise,
        distance_fun=sqfa.distances.hellinger,
    ),
    x_train_reg,
    y_train_reg,
    x_val,
    y_val,
    NOISE_VALS,
    fit_kwargs={"max_epochs": 300, "show_progress": False},
)

if not has_saved_artifacts("hellinger"):
    hellinger_filters, hellinger_times = train_sqfa_repeated(
        model_factory=lambda: sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=hellinger_noise,
            distance_fun=sqfa.distances.hellinger,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=N_REPS,
        fit_kwargs={"max_epochs": 300, "show_progress": False},
        run_label="hellinger training",
    )
    save_artifacts("hellinger", hellinger_filters, hellinger_times)


# ------------------------------
# Train LDA
# ------------------------------
if not has_saved_artifacts("lda"):
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=LDA_SHRINKAGE)
    start = time.time()
    lda.fit(x_train, y_train)
    lda_time = time.time() - start
    lda_filters = lda.coef_[:N_FILTERS]
    save_artifacts("lda", lda_filters, lda_time)


#############################
#
# LOAD FILTERS
#
#############################

filter_names = [
    "sqfa_filters.npy",
    "smsqfa_filters.npy",
    "bhattacharyya_filters.npy",
    "hellinger_filters.npy",
    "lda_filters.npy",
    "spca_filters.npy",
    "pca_filters.npy",
]

model_names = [
    "SQFA",
    "smSQFA",
    "SQFA-B",
    "SQFA-H",
    "LDA",
    "SPCA",
    "PCA",
]

model_filters = []
model_times = []
for name in filter_names:
    model_filters.append(np.load(os.path.join(FILTERS_DIR, name)))
    model_times.append(
        np.load(os.path.join(FILTERS_DIR, name.replace("filters", "time")))
    )


#############################
#
# PLOT TRAINING TIMES
#
#############################

time_scores = [np.asarray(times, dtype=float).reshape(-1) for times in model_times]
all_times = np.concatenate(time_scores)
time_min = float(all_times.min()) if all_times.size else 0.1
time_max = float(all_times.max()) if all_times.size else 1.0

plot_metric_with_errorbars(
    model_names,
    time_scores,
    "Training Time (s)",
    os.path.join(FIGURES_DIR, "cifar100_embedding_training_times.pdf"),
    unit=" s",
    value_fmt="{:.2f}",
    spread_fmt="{:.2f}",
    yscale="log",
    ylim=(max(time_min * 0.5, 1.0e-3), time_max * 5),
    offset_ratio=0.1,
    min_offset=0.05,
)


#############################
#
# PLOT QDA ACCURACIES
#
#############################

qda_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: qda_accuracy(x_train, y_train, x_test, y_test, filt).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_scores,
    "QDA Accuracy (%)",
    os.path.join(FIGURES_DIR, "cifar100_embedding_accuracies.pdf"),
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
    show_errorbars=True,
)


#############################
#
# PLOT KNN ACCURACIES
#
#############################

knn_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: knn_accuracy(x_train, y_train, x_test, y_test, filt).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    knn_scores,
    "KNN Accuracy (%)",
    os.path.join(FIGURES_DIR, "cifar100_embedding_accuracies_knn.pdf"),
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
)
