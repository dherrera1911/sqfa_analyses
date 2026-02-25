import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from metric_learn import LFDA
from sklearn.decomposition import PCA

import sqfa
from functions import load_data, normalize_stim

sys.path.append('..')  # Add parent directory to path
from pkg_utils import (
    collect_metric_across_runs,
    plot_metric_with_errorbars,
    qda_accuracy,
)


N_FILTERS = 8
RESPONSE_NOISE = 0.001
C50 = 0.8
N_REPS = 20
torch.manual_seed(2)


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

# ------------------------------
# Train SQFA
# ------------------------------
sqfa_filters = []
sqfa_times = []
for rep in range(N_REPS):
    sqfa_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=RESPONSE_NOISE,
    )
    start = time.time()
    sqfa_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
        pairwise=True,
        estimator="empirical",
    )
    sqfa_times.append(time.time() - start)
    sqfa_filters.append(sqfa_model.filters.detach().cpu().numpy())
np.save('filters/sqfa_filters.npy', np.array(sqfa_filters))
np.save('filters/sqfa_time.npy', np.array(sqfa_times))


# ------------------------------
# Train SQFA-Wasserstein
# ------------------------------
wasserstein_filters = []
wasserstein_times = []
for rep in range(N_REPS):
    wasserstein_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=RESPONSE_NOISE,
        distance_fun=sqfa.distances.wasserstein,
        constraint='orthogonal',
    )
    wasserstein_model.to(dtype=torch.float64)
    start = time.time()
    wasserstein_model.fit(
        x_train.to(dtype=torch.float64),
        y_train,
        max_epochs=300,
        show_progress=False,
        pairwise=True,
        estimator="empirical",
    )
    wasserstein_times.append(time.time() - start)
    wasserstein_filters.append(
        wasserstein_model.filters.detach().to(dtype=torch.float32).cpu().numpy()
    )
np.save('filters/wasserstein_filters.npy', np.array(wasserstein_filters))
np.save('filters/wasserstein_time.npy', np.array(wasserstein_times))


# ------------------------------
# Train SQFA-Jeffreys
# ------------------------------
jeffreys_filters = []
jeffreys_times = []
for rep in range(N_REPS):
    torch.manual_seed(302 + rep)
    jeffreys_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=RESPONSE_NOISE,
        distance_fun=sqfa.distances.jeffreys,
    )
    start = time.time()
    jeffreys_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
        pairwise=True,
        estimator="empirical",
    )
    jeffreys_times.append(time.time() - start)
    jeffreys_filters.append(jeffreys_model.filters.detach().cpu().numpy())

np.save('filters/jeffreys_filters.npy', np.array(jeffreys_filters))
np.save('filters/jeffreys_time.npy', np.array(jeffreys_times))


# ------------------------------
# Train LFDA
# ------------------------------
n_dim_lfda = 120
pca_subsample = PCA(n_components=n_dim_lfda)
pca_subsample.fit(x_train)
x_transformed = pca_subsample.transform(x_train)

lfda = LFDA(n_components=N_FILTERS, k=7, embedding_type='orthonormalized')
start = time.time()
lfda.fit(x_transformed, y_train.numpy())
lfda_time = time.time() - start
lfda_filters = pca_subsample.inverse_transform(lfda.components_)
np.save('filters/lfda_filters.npy', np.array(lfda_filters))
np.save('filters/lfda_time.npy', np.array(lfda_time))


#############################
#
# LOAD FILTERS
#
#############################

filter_filenames = [
    'sqfa_filters.npy',
    'wasserstein_filters.npy',
    'jeffreys_filters.npy',
    'lfda_filters.npy',
]

model_names = [
    "SQFA",
    "SQFA-W",
    "SQFA-J",
    "LFDA",
]

model_filters = [np.load(f'filters/{name}') for name in filter_filenames]


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
    'figures/motion_accuracies_review.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
    figsize=(4,3),
)
