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

# ------------------------------
# Load filters
# ------------------------------

class_stats = sqfa.statistics.class_statistics(points=x_train, labels=y_train)
sqfa_filters = np.load(f'filters/sqfa_filters.npy')
N_FILTERS = sqfa_filters[0].shape[0]

fisher_rao = sqfa.model.SQFA(
    n_dim=x_train.shape[1],
    n_filters=N_FILTERS,
)

fisher_rao.filters = torch.as_tensor(torch.as_tensor(sqfa_filters[0]).float())

#------------------ Get Calvo Oller approximated distances ------------------
co_dist = fisher_rao.get_class_distances(class_stats, regularized=False).detach()


#############################
#
# GET FISHER-RAO DISTANCES
#
#############################

with torch.no_grad():
    feature_covs = fisher_rao.transform_scatters(class_stats['covariances']).numpy()
    feature_means = fisher_rao.transform(class_stats['means']).numpy()

from bregman.application.distribution.exponential_family.gaussian import \
  GaussianManifold, GaussianFisherRaoDistance
from bregman.base import LAMBDA_COORDS, Point

to_vector = lambda mu, sigma: np.concatenate([mu, sigma.flatten()])

noise_mat = np.zeros((N_FILTERS, N_FILTERS))
const = 1

# Compute the distance matrix between all pairs of classes
n_classes = co_dist.shape[0]
dist_mat = np.zeros((n_classes, n_classes))

for c1 in range(n_classes):
    for c2 in range(c1 + 1, n_classes):
        print(f'Computing distance between classes {c1} and {c2}...')
        mean1 = feature_means[c1]
        cov1 = feature_covs[c1] + noise_mat
        mean2 = feature_means[c2]
        cov2 = feature_covs[c2] + noise_mat

        mean1 = mean1 * const
        mean2 = mean2 * const
        cov1 = cov1 * const**2
        cov2 = cov2 * const**2

        point_1 = Point(LAMBDA_COORDS, to_vector(mean1, cov1))
        point_2 = Point(LAMBDA_COORDS, to_vector(mean2, cov2))

        manifold = GaussianManifold(input_dimension=N_FILTERS)
        dist = GaussianFisherRaoDistance(manifold=manifold)
        dist_mat[c1, c2] = dist.dissimilarity(point_1, point_2, eps=1e-2)
        dist_mat[c2, c1] = dist_mat[c1, c2]


# Plot the co approximation vs Fisher-Rao distance

max_dist = np.max(dist_mat)

tril_inds = np.tril_indices(n_classes, k=-1)

co_dist = co_dist[tril_inds[0], tril_inds[1]]
fr_dist = dist_mat[tril_inds[0], tril_inds[1]]

plt.figure(figsize=(3, 3))
plt.plot([0, max_dist], [0, max_dist], color='black', linestyle='--')
plt.scatter(fr_dist, co_dist, alpha=0.5)
plt.ylabel('Calvo-Oller Approximation')
plt.xlabel('Fisher-Rao Distance')
# Make identity line
plt.xlim(0, max_dist)
plt.ylim(0, max_dist)
plt.tight_layout()
plt.savefig('figures/svhn_co_vs_fisherrao.pdf')
plt.close()

# Save distances
np.save('motion_fisherrao_distances.npy', dist_mat)
np.save('motion_co_distances.npy', co_dist)

