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
  validate_regularization,
  SupervisedPCA,
  plot_filter_grid,
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
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS)
start = time.time()
pca.fit(x_train)
pca_time = time.time() - start
pca_filters = pca.components_
np.save('filters/pca_filters.npy', np.array(pca_filters))
np.save('filters/pca_time.npy', np.array(pca_time))


# ------------------------------
# Train Supervised PCA
# ------------------------------
x_subsampled = x_train[::5]
y_subsampled = y_train[::5]
spca = SupervisedPCA(n_components=N_FILTERS, label_kernel="delta")
start = time.time()
spca.fit(x_subsampled, y_subsampled)
spca_time = time.time() - start
spca_filters = spca.components_
np.save('filters/spca_filters.npy', np.array(spca_filters))
np.save('filters/spca_time.npy', np.array(spca_time))


# ------------------------------
# Train smSQFA
# ------------------------------
if not PRELOAD_VALS:
    smsqfa_val = sqfa.model.SecondMomentsSQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=0,
    )
    smsqfa_noise_accs = validate_regularization(
      smsqfa_val, x_train_reg, y_train_reg, x_val, y_val, NOISE_VALS,
    )
    smsqfa_noise = NOISE_VALS[torch.argmax(smsqfa_noise_accs)]
    np.save('filters/smsqfa_noise.npy', np.array(smsqfa_noise))
else:
    smsqfa_noise = np.load('filters/smsqfa_noise.npy').item()

smsqfa_filter_list = []
smsqfa_times = []
for _rep in range(N_REPS):
    smsqfa_model = sqfa.model.SecondMomentsSQFA(
        n_dim=x_train.shape[1], n_filters=N_FILTERS, feature_noise=smsqfa_noise,
    )
    start = time.time()
    smsqfa_model.fit(x_train, y_train, max_epochs=300, show_progress=False)
    smsqfa_times.append(time.time() - start)
    smsqfa_filter_list.append(smsqfa_model.filters.detach().numpy())

np.save('filters/smsqfa_filters.npy', np.array(smsqfa_filter_list))
np.save('filters/smsqfa_time.npy', np.array(smsqfa_times))


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
# Train Bhattacharyya
# ------------------------------
if not PRELOAD_VALS:
    bhattacharyya_val = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=0,
        distance_fun=sqfa.distances.bhattacharyya,
    )

    bhattacharyya_noise_accs = validate_regularization(
        bhattacharyya_val, x_train_reg, y_train_reg, x_val, y_val, NOISE_VALS,
    )
    bhattacharyya_noise = NOISE_VALS[torch.argmax(bhattacharyya_noise_accs)]
    np.save('filters/bhattacharyya_noise.npy', np.array(bhattacharyya_noise))
else:
    bhattacharyya_noise = np.load('filters/bhattacharyya_noise.npy').item()

bhattacharyya_filter_list = []
bhattacharyya_times = []
for _rep in range(N_REPS):
    bhattacharyya_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=bhattacharyya_noise,
        distance_fun=sqfa.distances.bhattacharyya,
    )
    start = time.time()
    bhattacharyya_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    bhattacharyya_times.append(time.time() - start)
    bhattacharyya_filter_list.append(bhattacharyya_model.filters.detach().numpy())

np.save('filters/bhattacharyya_filters.npy', np.array(bhattacharyya_filter_list))
np.save('filters/bhattacharyya_time.npy', np.array(bhattacharyya_times))


# ------------------------------
# Train Hellinger
# ------------------------------
if not PRELOAD_VALS:
    hellinger_val = sqfa.model.SQFA(
        n_dim=x_train.shape[1], n_filters=N_FILTERS, feature_noise=0,
        distance_fun=sqfa.distances.hellinger,
    )

    hellinger_noise_accs = validate_regularization(
        hellinger_val, x_train_reg, y_train_reg, x_val, y_val, NOISE_VALS,
    )
    hellinger_noise = NOISE_VALS[torch.argmax(hellinger_noise_accs)]
    np.save('filters/hellinger_noise.npy', np.array(hellinger_noise))
else:
    hellinger_noise = np.load('filters/hellinger_noise.npy').item()

hellinger_filter_list = []
hellinger_times = []
for _rep in range(N_REPS):
    hellinger_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=hellinger_noise,
        distance_fun=sqfa.distances.hellinger,
    )
    start = time.time()
    hellinger_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    hellinger_times.append(time.time() - start)
    hellinger_filter_list.append(hellinger_model.filters.detach().numpy())

np.save('filters/hellinger_filters.npy', np.array(hellinger_filter_list))
np.save('filters/hellinger_time.npy', np.array(hellinger_times))


# ------------------------------
# Train LDA
# ------------------------------
shrinkage = 0.5  # Set to optimize LDA performance and have smoother filters
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
start = time.time()
lda.fit(x_train, y_train)
lda_time = time.time() - start
lda_filters = lda.coef_[:N_FILTERS]
np.save('filters/lda_filters.npy', np.array(lda_filters))
np.save('filters/lda_time.npy', np.array(lda_time))


# ------------------------------
# Train ICA
# ------------------------------
#ica = FastICA(n_components=N_FILTERS, random_state=0, max_iter=1000)
#start = time.time()
#ica.fit(x_train)
#ica_time = time.time() - start
#ica_filters = ica.components_
#np.save('filters/ica_filters.npy', np.array(ica_filters))
#np.save('filters/ica_time.npy', np.array(ica_time))
#
#
## ------------------------------
## Train Factor Analysis
## ------------------------------
#fa = FactorAnalysis(n_components=N_FILTERS, random_state=0, max_iter=1000)
#start = time.time()
#fa.fit(x_train)
#fa_time = time.time() - start
#fa_filters = fa.components_
#np.save('filters/fa_filters.npy', np.array(fa_filters))
#np.save('filters/fa_time.npy', np.array(fa_time))
#

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
#lfda.fit(x_train, y_train)
lfda_time = time.time() - start
#lfda_filters = lfda.components_
lfda_filters = pca_subsample.inverse_transform(lfda.components_)

np.save('filters/lfda_filters.npy', np.array(lfda_filters))
np.save('filters/lfda_time.npy', np.array(lfda_time))

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
#lmnn = LMNN(
#    n_neighbors=3,
#    learn_rate=1e-6,
#    n_components=N_FILTERS,
#    init='pca',
#    verbose=True,
#    max_iter=2000,
#    convergence_tol=1.0,
#)
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
    'figures/mnist_training_times.pdf',
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
        lambda filt: qda_accuracy(x_train, y_train, x_test, y_test, filt, noise=0).item(),
    )
    for filters in model_filters
]

plot_metric_with_errorbars(
    model_names,
    qda_scores,
    "QDA Accuracy (%)",
    'figures/mnist_accuracies.pdf',
    scale=100.0,
    ylim=(80, 105),
    offset_ratio=0.05,
    unit="%",
)

i = 0
for name, filters in zip(model_names, model_filters):
    display_filters = filters
    if filters.ndim == 3:
        display_filters = filters[int(np.argmax(qda_scores[i]))]
    fig, _ = plot_filter_grid(display_filters, (n_row, n_col), n_cols=9, figsize=(7, 1))
    plt.savefig(
      f'figures/mnist_{name.lower()}_filters.png',
      bbox_inches='tight',
      pad_inches=0,
    )
    plt.close(fig)
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
    'figures/mnist_accuracies_knn.pdf',
    scale=100.0,
    ylim=(80, 100),
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
    'figures/mnist_accuracies_gaussian.pdf',
    scale=100.0,
    ylim=(80, 100),
    offset_ratio=0.05,
    unit="%",
)
