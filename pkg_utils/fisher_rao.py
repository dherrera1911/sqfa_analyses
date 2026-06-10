"""Utilities for Fisher-Rao distance analyses in learned feature spaces."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from bregman.application.distribution.exponential_family.gaussian import (
    GaussianFisherRaoDistance,
    GaussianManifold,
)
from bregman.base import LAMBDA_COORDS, Point

import sqfa


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


def compute_calvo_oller_matrix(class_stats, filters, covariance_noise=0.0):
    """Compute Calvo-Oller lower bounds using an SQFA model with fixed filters."""
    filters = np.asarray(filters, dtype=np.float32)
    data_statistics = {
        "means": torch.as_tensor(class_stats["means"], dtype=torch.float32),
        "covariances": torch.as_tensor(class_stats["covariances"], dtype=torch.float32),
    }
    model = sqfa.model.SQFA(
        n_dim=filters.shape[1],
        n_filters=filters.shape[0],
        feature_noise=float(covariance_noise),
        filters=filters,
        constraint="none",
    )
    with torch.no_grad():
        distance_matrix = model.get_class_distances(
            data_statistics,
            regularized=True,
        )
    return distance_matrix.detach().cpu().numpy()


def save_pairwise_distance_csv(
    fisher_rao_matrix,
    calvo_oller_matrix,
    output_path,
    extra_fields=None,
):
    """Save pairwise true Fisher-Rao and Calvo-Oller distances in long format."""
    extra_fields = {} if extra_fields is None else dict(extra_fields)
    fieldnames = [
        *extra_fields.keys(),
        "class_1",
        "class_2",
        "true_fisher_rao_distance",
        "calvo_oller_distance",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for class_1 in range(fisher_rao_matrix.shape[0]):
            for class_2 in range(class_1 + 1, fisher_rao_matrix.shape[0]):
                row = {
                    **extra_fields,
                    "class_1": class_1,
                    "class_2": class_2,
                    "true_fisher_rao_distance": float(
                        fisher_rao_matrix[class_1, class_2]
                    ),
                    "calvo_oller_distance": float(
                        calvo_oller_matrix[class_1, class_2]
                    ),
                }
                writer.writerow(row)


def plot_calvo_oller_vs_fisher_rao(
    fisher_rao_matrix,
    calvo_oller_matrix,
    output_path,
):
    """Plot pairwise Calvo-Oller lower bounds against true Fisher-Rao distances."""
    upper_tri = np.triu_indices(fisher_rao_matrix.shape[0], k=1)
    fisher_rao_values = fisher_rao_matrix[upper_tri]
    calvo_oller_values = calvo_oller_matrix[upper_tri]

    max_dist = float(np.nanmax([fisher_rao_values.max(), calvo_oller_values.max()]))
    max_dist = max_dist * 1.02

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot([0, max_dist], [0, max_dist], color="black", linestyle="--", linewidth=1)
    ax.scatter(fisher_rao_values, calvo_oller_values, alpha=0.6, s=18)
    ax.set_xlabel("True Fisher-Rao distance")
    ax.set_ylabel("Calvo-Oller lower bound")
    ax.set_xlim(0, max_dist)
    ax.set_ylim(0, max_dist)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
