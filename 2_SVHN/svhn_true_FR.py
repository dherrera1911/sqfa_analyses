import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import sqfa
import time


# Download and load training and test datasets
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

#############################
#
# LOAD AND PROCESS DATA
#
#############################

# Convert to PyTorch tensors, average channels and reshape
n_samples, n_channels, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float()
x_train = x_train.mean(dim=1).reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.labels, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float()
x_test = x_test.mean(dim=1).reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.labels, dtype=torch.long)

# Scale data and subtract global mean
def scale_and_center(x_train, x_test):
    std = x_train.std()
    x_train = x_train / (std * n_row)
    x_test = x_test / (std * n_row)
    global_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - global_mean
    x_test = x_test - global_mean
    return x_train, x_test

x_train, x_test = scale_and_center(x_train, x_test)

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

