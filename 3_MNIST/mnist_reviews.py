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
  qda_accuracy,
  collect_metric_across_runs,
  validate_regularization,
  plot_metric_with_errorbars,
)

N_FILTERS = 9
NOISE_VALS = torch.tensor([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
n_subsample = 10
n_dim_lmnn = 100
N_REPS = 20
PRELOAD_VALS = True
torch.manual_seed(3)

#############################
#
# LOAD AND PROCESS DATA
#
#############################

# Download and load training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

n_samples, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float().reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.targets, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float().reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.targets, dtype=torch.long)

x_train, x_test = scale_and_center(x_train, x_test)

x_train_reg, y_train_reg, x_val, y_val = train_val_split(
  x_train,
  y_train,
  val_size=0.15,
)

#############################
#
# TRAIN MODELS
#
#############################


# ------------------------------
# Train SQFA
# ------------------------------
if not PRELOAD_VALS:
    sqfa_val = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=0,
    )
    sqfa_val.to(dtype=torch.float64)

    sqfa_noise_accs = validate_regularization(
      sqfa_val, x_train_reg.to(dtype=torch.float64), y_train_reg, x_val.to(dtype=torch.float64),
      y_val, NOISE_VALS.to(dtype=torch.float64),
    )
    sqfa_noise = NOISE_VALS[torch.argmax(sqfa_noise_accs)]
    np.save('filters/sqfa_noise.npy', np.array(sqfa_noise))
else:
    sqfa_noise = np.load('filters/sqfa_noise.npy').item()

sqfa_filter_list = []
sqfa_times = []
for _rep in range(N_REPS):
    sqfa_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1], n_filters=N_FILTERS, feature_noise=sqfa_noise,
    )
    start = time.time()
    sqfa_model.fit(x_train, y_train, max_epochs=300, show_progress=False)
    sqfa_times.append(time.time() - start)
    sqfa_filter_list.append(sqfa_model.filters.detach().numpy())

np.save('filters/sqfa_filters.npy', np.array(sqfa_filter_list))
np.save('filters/sqfa_time.npy', np.array(sqfa_times))


# ------------------------------
# Train SQFA-Wasserstein
# ------------------------------
if not PRELOAD_VALS:
    wasserstein_val = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=0,
        distance_fun=sqfa.distances.wasserstein,
    )

    wasserstein_val.to(dtype=torch.float64)

    wasserstein_noise_accs = validate_regularization(
      wasserstein_val,
      x_train_reg.to(dtype=torch.float64),
      y_train_reg,
      x_val.to(dtype=torch.float64),
      y_val,
      NOISE_VALS,
    )
    wasserstein_noise = NOISE_VALS[torch.argmax(wasserstein_noise_accs)]
    np.save('filters/wasserstein_noise.npy', np.array(wasserstein_noise))
else:
    wasserstein_noise = np.load('filters/wasserstein_noise.npy').item()

wasserstein_filter_list = []
wasserstein_times = []
for _rep in range(N_REPS):
    wasserstein_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=wasserstein_noise,
        distance_fun=sqfa.distances.wasserstein,
    )
    wasserstein_model.to(dtype=torch.float64)
    start = time.time()
    wasserstein_model.fit(
        x_train.to(dtype=torch.float64),
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    wasserstein_times.append(time.time() - start)
    wasserstein_filter_list.append(wasserstein_model.filters.detach().to(dtype=torch.float32).numpy())

np.save('filters/wasserstein_filters.npy', np.array(wasserstein_filter_list))
np.save('filters/wasserstein_time.npy', np.array(wasserstein_times))


# ------------------------------
# Train SQFA-Jeffreys
# ------------------------------
if not PRELOAD_VALS:
    jeffreys_val = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=0,
        distance_fun=sqfa.distances.jeffreys,
    )

    jeffreys_noise_accs = validate_regularization(
        jeffreys_val, x_train_reg, y_train_reg, x_val, y_val, NOISE_VALS,
    )
    jeffreys_noise = NOISE_VALS[torch.argmax(jeffreys_noise_accs)]
    np.save('filters/jeffreys_noise.npy', np.array(jeffreys_noise))
else:
    jeffreys_noise = np.load('filters/jeffreys_noise.npy').item()

jeffreys_filter_list = []
jeffreys_times = []
for _rep in range(N_REPS):
    jeffreys_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=jeffreys_noise,
        distance_fun=sqfa.distances.jeffreys,
    )
    start = time.time()
    jeffreys_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    jeffreys_times.append(time.time() - start)
    jeffreys_filter_list.append(jeffreys_model.filters.detach().numpy())

np.save('filters/jeffreys_filters.npy', np.array(jeffreys_filter_list))
np.save('filters/jeffreys_time.npy', np.array(jeffreys_times))


# ------------------------------
# Train LFDA
# ------------------------------
n_dim_lfda = 200
pca_subsample = PCA(n_components=n_dim_lfda)
pca_subsample.fit(x_train)
x_transformed = pca_subsample.transform(x_train)

lfda = LFDA(n_components=N_FILTERS, k=7, embedding_type='orthonormalized')
start = time.time()
lfda.fit(x_transformed, y_train)
lfda_time = time.time() - start
lfda_filters = pca_subsample.inverse_transform(lfda.components_)

np.save('filters/lfda_filters.npy', np.array(lfda_filters))
np.save('filters/lfda_time.npy', np.array(lfda_time))


#############################
#
# PLOT QDA ACCURACIES
#
#############################

filter_names = [
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

model_filters = []
for name in filter_names:
    model_filters.append(np.load(f'filters/{name}'))

qda_scores = [
    collect_metric_across_runs(
        filters,
        lambda filt: qda_accuracy(x_train, y_train, x_test, y_test, filt, noise=0.001).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_scores,
    "QDA Accuracy (%)",
    'figures/mnist_accuracies_review.pdf',
    scale=100.0,
    ylim=(60, 100),
    offset_ratio=0.05,
    unit="%",
    figsize=(4,3),
)
