import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from metric_learn import LMNN
from sklearn.decomposition import FastICA, FactorAnalysis, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sqfa
from functions import load_data, normalize_stim

sys.path.append(os.path.abspath("./ama_dir"))
from ama_model import AMAGauss
from optim import fit

sys.path.append('..')  # Add parent directory to path
from pkg_utils import (
    SupervisedPCA,
    collect_metric_across_runs,
    knn_accuracy,
    plot_filter_grid,
    plot_metric_with_errorbars,
    qda_accuracy,
    qda_accuracy_gaussian,
    train_sqfa_repeated,
)


N_FILTERS = 8
RESPONSE_NOISE = 0.001
C50 = 0.8
IMAGE_SHAPE = (15, 30)

N_SUBSAMPLE_LMNN = 10
N_DIM_LMNN = 100
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
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS, svd_solver="covariance_eigh")
start = time.time()
pca.fit(x_train)
pca_time = time.time() - start
pca_filters = np.asarray(pca.components_, dtype=np.float32)
np.save('filters/pca_filters.npy', pca_filters)
np.save('filters/pca_time.npy', np.array(pca_time))


# ------------------------------
# Train Supervised PCA
# ------------------------------
spca = SupervisedPCA(n_components=N_FILTERS, label_kernel="delta")
start = time.time()
spca.fit(x_train, y_train)
spca_time = time.time() - start
spca_filters = np.asarray(spca.components_, dtype=np.float32)
np.save('filters/spca_filters.npy', spca_filters)
np.save('filters/spca_time.npy', np.array(spca_time))


# ------------------------------
# Train smSQFA
# ------------------------------
if not (
    os.path.exists('filters/smsqfa_filters.npy')
    and os.path.exists('filters/smsqfa_time.npy')
):
    smsqfa_filters, smsqfa_times = [], []
    for rep in range(N_REPS):
        torch.manual_seed(2 + rep)
        smsqfa_model = sqfa.model.SecondMomentsSQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
        )
        start = time.time()
        smsqfa_model.fit(
            x_train,
            y_train,
            max_epochs=300,
            show_progress=False,
            pairwise=True,
            estimator="empirical",
        )
        smsqfa_times.append(time.time() - start)
        smsqfa_filters.append(smsqfa_model.filters.detach().cpu().numpy())
    np.save('filters/smsqfa_filters.npy', np.array(smsqfa_filters))
    np.save('filters/smsqfa_time.npy', np.array(smsqfa_times))


# ------------------------------
# Train SQFA
# ------------------------------
if not (
    os.path.exists('filters/sqfa_filters.npy')
    and os.path.exists('filters/sqfa_time.npy')
):
    sqfa_filters, sqfa_times = [], []
    for rep in range(N_REPS):
        torch.manual_seed(102 + rep)
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
# Train Bhattacharyya
# ------------------------------
if not (
    os.path.exists('filters/bhattacharyya_filters.npy')
    and os.path.exists('filters/bhattacharyya_time.npy')
):
    bhattacharyya_filters, bhattacharyya_times = [], []
    for rep in range(N_REPS):
        torch.manual_seed(202 + rep)
        bhattacharyya_model = sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
            distance_fun=sqfa.distances.bhattacharyya,
        )
        start = time.time()
        bhattacharyya_model.fit(
            x_train,
            y_train,
            max_epochs=300,
            show_progress=False,
            pairwise=True,
            estimator="empirical",
        )
        bhattacharyya_times.append(time.time() - start)
        bhattacharyya_filters.append(
            bhattacharyya_model.filters.detach().cpu().numpy()
        )
    np.save('filters/bhattacharyya_filters.npy', np.array(bhattacharyya_filters))
    np.save('filters/bhattacharyya_time.npy', np.array(bhattacharyya_times))


# ------------------------------
# Train Hellinger
# ------------------------------
if not (
    os.path.exists('filters/hellinger_filters.npy')
    and os.path.exists('filters/hellinger_time.npy')
):
    hellinger_filter_list = []
    hellinger_times = []
    for _rep in range(N_REPS):
        hellinger_model = sqfa.model.SQFA(
            n_dim=x_train.shape[1],
            n_filters=N_FILTERS,
            feature_noise=RESPONSE_NOISE,
            distance_fun=sqfa.distances.hellinger,
        )
        start = time.time()
        hellinger_model.fit(
            x_train,
            y_train,
            max_epochs=300,
            show_progress=False,
            pairwise=True,
            estimator="empirical",
        )
        hellinger_times.append(time.time() - start)
        hellinger_filter_list.append(hellinger_model.filters.detach().numpy())

    np.save('filters/hellinger_filters.npy', np.array(hellinger_filter_list))
    np.save('filters/hellinger_time.npy', np.array(hellinger_times))


# ------------------------------
# Train Wasserstein
# ------------------------------
if not (
    os.path.exists('filters/wasserstein_filters.npy')
    and os.path.exists('filters/wasserstein_time.npy')
):
    torch.manual_seed(402)
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
        fit_kwargs={
            "max_epochs": 300,
            "show_progress": False,
            "pairwise": True,
            "estimator": "empirical",
        },
        dtypes=(torch.float64,),
        run_label="wasserstein training",
    )
    np.save('filters/wasserstein_filters.npy', wasserstein_filters)
    np.save('filters/wasserstein_time.npy', wasserstein_times)


# ------------------------------
# Train Jeffreys
# ------------------------------
if not (
    os.path.exists('filters/jeffreys_filters.npy')
    and os.path.exists('filters/jeffreys_time.npy')
):
    jeffreys_filters, jeffreys_times = [], []
    for rep in range(N_REPS):
        torch.manual_seed(502 + rep)
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
# Train AMA
# ------------------------------
if not (
    os.path.exists('filters/ama_filters.npy')
    and os.path.exists('filters/ama_time.npy')
):
    x_train_ch = x_train.unsqueeze(1)  # Add channel dimension
    ama_filters, ama_times = [], []
    for rep in range(N_REPS):
        torch.manual_seed(302 + rep)
        ama = AMAGauss(
            stimuli=x_train_ch,
            labels=y_train,
            n_filters=N_FILTERS,
            response_noise=RESPONSE_NOISE,
        )
        start = time.time()
        fit(
            model=ama,
            stimuli=x_train_ch,
            labels=y_train,
            max_epochs=100,
            lr=0.2,
            show_progress=False,
            pairwise=True,
        )
        ama_times.append(time.time() - start)
        ama_filters.append(
            np.asarray(ama.filters.squeeze().detach().cpu().numpy(), dtype=np.float32)
        )

    np.save('filters/ama_filters.npy', np.asarray(ama_filters))
    np.save('filters/ama_time.npy', np.asarray(ama_times))


# ------------------------------
# Train LDA
# ------------------------------
shrinkage = 0.7
lda = LinearDiscriminantAnalysis(
    n_components=N_FILTERS,
    solver='eigen',
    shrinkage=shrinkage,
)
start = time.time()
lda.fit(x_train, y_train)
lda_time = time.time() - start
lda_filters = np.asarray(lda.coef_[:N_FILTERS], dtype=np.float32)
np.save('filters/lda_filters.npy', lda_filters)
np.save('filters/lda_time.npy', np.array(lda_time))


# ------------------------------
# Train Factor Analysis
# ------------------------------
fa = FactorAnalysis(n_components=N_FILTERS)
start = time.time()
fa.fit(x_train)
fa_time = time.time() - start
fa_filters = np.asarray(fa.components_, dtype=np.float32)
np.save('filters/fa_filters.npy', fa_filters)
np.save('filters/fa_time.npy', np.array(fa_time))


# ------------------------------
# Train ICA
# ------------------------------
ica = FastICA(n_components=N_FILTERS)
start = time.time()
ica.fit(x_train)
ica_time = time.time() - start
ica_filters = np.asarray(ica.components_, dtype=np.float32)
np.save('filters/ica_filters.npy', ica_filters)
np.save('filters/ica_time.npy', np.array(ica_time))


# ------------------------------
# Train LMNN
# ------------------------------
if not (
    os.path.exists('filters/lmnn_filters.npy')
    and os.path.exists('filters/lmnn_time.npy')
):
    pca_subsample = PCA(n_components=N_DIM_LMNN)
    pca_subsample.fit(x_train)
    x_transformed = pca_subsample.transform(x_train)

    y_train_sub = y_train.numpy()[::N_SUBSAMPLE_LMNN]
    x_transformed_sub = x_transformed[::N_SUBSAMPLE_LMNN]

    lmnn = LMNN(
        n_neighbors=3,
        learn_rate=1e-6,
        n_components=8,
        init='pca',
        verbose=True,
        max_iter=2000,
        convergence_tol=1.0,
    )
    start = time.time()
    lmnn.fit(x_transformed_sub, y_train_sub)
    lmnn_time = time.time() - start
    lmnn_filters = pca_subsample.inverse_transform(lmnn.components_)
    np.save('filters/lmnn_filters.npy', np.array(lmnn_filters))
    np.save('filters/lmnn_time.npy', np.array(lmnn_time))


#############################
#
# LOAD FILTERS AND TIMES
#
#############################

filter_filenames = [
    'sqfa_filters.npy',
    'smsqfa_filters.npy',
    'bhattacharyya_filters.npy',
    'hellinger_filters.npy',
    'wasserstein_filters.npy',
    'jeffreys_filters.npy',
    'ama_filters.npy',
    'lda_filters.npy',
    'spca_filters.npy',
    'filters_review/lfda_filters_n8.npy',
    'filters_review/wda_filters_n8.npy',
    'pca_filters.npy',
    'lmnn_filters.npy',
]

model_names = [
    "SQFA",
    "smSQFA",
    "Bhatt",
    "Hellinger",
    "SQFA-W",
    "SQFA-J",
    "AMA",
    "LDA",
    "SPCA",
    "LFDA",
    "WDA",
    "PCA",
    "LMNN",
]

model_filters = [
    np.load(name if name.startswith('filters_review/') else f'filters/{name}')
    for name in filter_filenames
]
model_times = [
    np.load(
        name.replace('_filters_', '_time_')
        if name.startswith('filters_review/')
        else f"filters/{name.replace('filters', 'time')}"
    )
    for name in filter_filenames
]


#############################
#
# PLOT TRAINING TIMES
#
#############################

time_scores = [np.asarray(times, dtype=float).reshape(-1) for times in model_times]
concatenated = [scores for scores in time_scores if scores.size]
if concatenated:
    all_times = np.concatenate(concatenated)
    time_min = float(all_times.min())
    time_max = float(all_times.max())
else:
    time_min = 0.1
    time_max = 1.0

plot_metric_with_errorbars(
    model_names,
    time_scores,
    "Training Time (s)",
    'figures/motion_training_times.pdf',
    unit=" s",
    value_fmt="{:.2f}",
    spread_fmt="{:.2f}",
    yscale='log',
    ylim=(max(time_min * 0.5, 1e-3), time_max * 5),
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
    'figures/motion_accuracies.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
)


#############################
#
# PLOT FILTERS
#
#############################

for idx, (name, filters) in enumerate(zip(model_names, model_filters)):
    chosen_filters = filters
    if filters.ndim == 3:
        best_idx = int(np.argmax(qda_scores[idx]))
        chosen_filters = filters[best_idx]
    fig, _ = plot_filter_grid(chosen_filters, IMAGE_SHAPE, n_cols=N_FILTERS, figsize=(7, 1))
    fig.savefig(
        f'figures/motion_{name.lower()}_filters.png',
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close(fig)


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
    'figures/motion_accuracies_knn.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
)


#############################
#
# PLOT QDA ACCURACIES WITH GAUSSIAN DATA
#
#############################

qda_gaussian_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: qda_accuracy_gaussian(x_train, y_train, filt),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_gaussian_scores,
    "QDA Accuracy (%)",
    'figures/motion_accuracies_gaussian.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio=0.05,
    unit="%",
)
