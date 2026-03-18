import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from metric_learn import LMNN, LFDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import sqfa
import time

import sys
sys.path.append('..')  # Add parent directory to path

from pkg_utils import (
  scale_and_center,
  train_val_split,
  qda_accuracy,
  knn_accuracy,
  qda_accuracy_gaussian,
  collect_metric_across_runs,
  train_sqfa_repeated,
  validate_regularization,
  SupervisedPCA,
  plot_filter_grid,
  plot_metric_with_errorbars,
)

N_FILTERS = 9
NOISE_VALS = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
N_REPS = 20
n_subsample = 10
n_dim_lmnn = 100
FILTERS_DIR = "filters"
torch.manual_seed(2)

os.makedirs(FILTERS_DIR, exist_ok=True)


def filter_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_filters.npy"


def time_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_time.npy"


def noise_path(model_key):
    return f"{FILTERS_DIR}/{model_key}_noise.npy"


def has_saved_artifacts(model_key):
    return os.path.exists(filter_path(model_key)) and os.path.exists(time_path(model_key))


def save_artifacts(model_key, filters, times):
    np.save(filter_path(model_key), np.asarray(filters))
    np.save(time_path(model_key), np.asarray(times))


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
# Get noise hyperparameter via validation
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
    fit_kwargs={"lr": 0.2, "max_epochs": 500, "show_progress": True},
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
        fit_kwargs={"lr": 0.2, "max_epochs": 500, "show_progress": True},
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
shrinkage = 0.8  # Set to optimize LDA performance and have smoother filters
if not has_saved_artifacts("lda"):
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
    start = time.time()
    lda.fit(x_train, y_train)
    lda_time = time.time() - start
    lda_filters = lda.coef_[:N_FILTERS]
    save_artifacts("lda", lda_filters, lda_time)


# ------------------------------
# Train ICA
# ------------------------------
if not has_saved_artifacts("ica"):
    ica = FastICA(n_components=N_FILTERS, random_state=0, max_iter=1000)
    start = time.time()
    ica.fit(x_train)
    ica_time = time.time() - start
    save_artifacts("ica", ica.components_, ica_time)


# ------------------------------
# Train Factor Analysis
# ------------------------------
if not has_saved_artifacts("fa"):
    fa = FactorAnalysis(n_components=N_FILTERS, random_state=0, max_iter=1000)
    start = time.time()
    fa.fit(x_train)
    fa_time = time.time() - start
    save_artifacts("fa", fa.components_, fa_time)


# ------------------------------
# Train LFDA
# ------------------------------
if not has_saved_artifacts("lfda"):
    pca_subsample = PCA(n_components=400)
    pca_subsample.fit(x_train)
    x_transformed = pca_subsample.transform(x_train)

    lfda = LFDA(n_components=N_FILTERS, k=5)
    start = time.time()
    lfda.fit(x_train, y_train)
    lfda_time = time.time() - start
    lfda_filters = lfda.components_

    save_artifacts("lfda", lfda_filters, lfda_time)


# ------------------------------
# Train LMNN
# ------------------------------
#pca_subsample = PCA(n_components=n_dim_lmnn)
#pca_subsample.fit(x_train)
#x_transformed = pca_subsample.transform(x_train)
#
#y_train_sub = y_train
#x_transformed, y_train_sub = x_transformed[::n_subsample], y_train[::n_subsample]
#
#lmnn = LMNN(n_neighbors=3, learn_rate=1e-6, n_components=9, init='pca',
#            verbose=True, max_iter=2000, convergence_tol=1.0)
#start = time.time()
#lmnn.fit(x_transformed, y_train_sub)
#lmnn_time = time.time() - start
#
#lmnn_filters = pca_subsample.inverse_transform(lmnn.components_)
#
#np.save('filters/lmnn_filters.npy', np.array(lmnn_filters))
#np.save('filters/lmnn_time.npy', np.array(lmnn_time))


#############################
#
# PLOT FILTERS
#
#############################

# Load filters
filter_names = [
    'sqfa_filters.npy',
    'smsqfa_filters.npy',
    'bhattacharyya_filters.npy',
    'hellinger_filters.npy',
    'lda_filters.npy',
    'spca_filters.npy',
    'lfda_filters.npy',
    'pca_filters.npy',
    'lmnn_filters.npy',
]

model_names = [
  "SQFA",
  "smSQFA",
  "SQFA-B",
  "SQFA-H",
  "LDA",
  "SPCA",
  "LFDA",
  "PCA",
  "LMNN",
]

model_filters = []
model_times = []
for name in filter_names:
    model_filters.append(np.load(f'filters/{name}'))
    model_times.append(np.load(
      f'filters/{name.replace("filters", "time")}')
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
    'figures/svhn_training_times.pdf',
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
    'figures/svhn_accuracies.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio = 0.05,
    unit="%",
    show_errorbars=True,
)


#############################
#
# PLOT FILTERS
#
#############################

# Function to plot filters
i = 0
for name, filters in zip(model_names, model_filters):
    if filters.ndim == 3:
        # Plot filters with best performance 
        ind = np.argmax(qda_scores[i])
        filters = filters[ind]
    plot_filter_grid(filters, (n_row, n_col), n_cols=9, figsize=(7, 1))
    plt.savefig(
      f'figures/svhn_{name.lower()}_filters.png', bbox_inches='tight', pad_inches=0
    )
    plt.close()
    i += 1


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
    'figures/svhn_accuracies_knn.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio = 0.05,
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
    'figures/svhn_accuracies_gaussian.pdf',
    scale=100.0,
    ylim=(0, 100),
    offset_ratio = 0.05,
    unit="%",
)
