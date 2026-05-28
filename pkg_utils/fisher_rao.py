"""Utilities for Fisher-Rao distance analyses in learned feature spaces."""

import numpy as np
import torch
from bregman.application.distribution.exponential_family.gaussian import (
    GaussianFisherRaoDistance,
    GaussianManifold,
)
from bregman.base import LAMBDA_COORDS, Point


def select_cached_filter_set(filter_bank, run_idx=0):
    """Return a single filter set from a cached artifact."""
    filters = np.asarray(filter_bank)
    if filters.ndim == 2:
        return filters
    if filters.ndim == 3:
        if run_idx < 0 or run_idx >= filters.shape[0]:
            raise IndexError(
                f"run_idx={run_idx} is out of bounds for {filters.shape[0]} cached runs"
            )
        return filters[run_idx]
    raise ValueError(
        f"Unexpected cached filter dimensionality {filters.ndim}; expected 2 or 3."
    )


def project_class_statistics(class_stats, filters, dtype=torch.float64):
    """Project class means and covariances into the feature space defined by filters."""
    means = torch.as_tensor(class_stats["means"], dtype=dtype)
    covariances = torch.as_tensor(class_stats["covariances"], dtype=dtype)
    filters = torch.as_tensor(filters, dtype=dtype)

    feature_means = torch.matmul(means, filters.T)
    feature_covariances = torch.matmul(
        filters.unsqueeze(0),
        torch.matmul(covariances, filters.T),
    )
    return feature_means.detach().cpu().numpy(), feature_covariances.detach().cpu().numpy()


def compute_true_fisher_rao_matrix(
    feature_means,
    feature_covariances,
    covariance_noise=0.0,
    eps=1.0e-2,
):
    """Compute the full pairwise Fisher-Rao distance matrix between Gaussian classes."""
    feature_means = np.asarray(feature_means, dtype=np.float64)
    feature_covariances = np.asarray(feature_covariances, dtype=np.float64)

    n_classes, n_filters = feature_means.shape
    noise_mat = float(covariance_noise) * np.eye(n_filters, dtype=np.float64)
    manifold = GaussianManifold(input_dimension=n_filters)
    distance_fun = GaussianFisherRaoDistance(manifold=manifold)
    dist_mat = np.zeros((n_classes, n_classes), dtype=np.float64)

    for c1 in range(n_classes):
        for c2 in range(c1 + 1, n_classes):
            point_1 = Point(
                LAMBDA_COORDS,
                np.concatenate(
                    [feature_means[c1], (feature_covariances[c1] + noise_mat).ravel()]
                ),
            )
            point_2 = Point(
                LAMBDA_COORDS,
                np.concatenate(
                    [feature_means[c2], (feature_covariances[c2] + noise_mat).ravel()]
                ),
            )
            dist_value = distance_fun.dissimilarity(point_1, point_2, eps=eps)
            dist_mat[c1, c2] = dist_value
            dist_mat[c2, c1] = dist_value

    return dist_mat


def mean_pairwise_distance(distance_matrix):
    """Return the mean off-diagonal pairwise distance."""
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    upper_tri = np.triu_indices(distance_matrix.shape[0], k=1)
    if upper_tri[0].size == 0:
        return 0.0
    return float(distance_matrix[upper_tri].mean())


def compute_mean_true_fisher_rao(
    class_stats,
    filters,
    covariance_noise=0.0,
    eps=1.0e-2,
    dtype=torch.float64,
):
    """Project class statistics, compute the Fisher-Rao matrix, and average it."""
    feature_means, feature_covariances = project_class_statistics(
        class_stats=class_stats,
        filters=filters,
        dtype=dtype,
    )
    distance_matrix = compute_true_fisher_rao_matrix(
        feature_means=feature_means,
        feature_covariances=feature_covariances,
        covariance_noise=covariance_noise,
        eps=eps,
    )
    return distance_matrix, mean_pairwise_distance(distance_matrix)
