"""Helpers for PCA preprocessing and lifting PCA-space filters."""

import numpy as np
import torch
from sklearn.decomposition import PCA


def fit_preprocessing_pca(x_fit, n_components):
    """Fit a PCA model with the requested cap on retained components."""
    n_components = min(int(n_components), x_fit.shape[0], x_fit.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(torch.as_tensor(x_fit).detach().cpu().numpy())
    return pca


def transform_with_pca(pca, x):
    """Project data into the PCA space and return a float32 tensor."""
    x_reduced = pca.transform(torch.as_tensor(x).detach().cpu().numpy())
    return torch.as_tensor(x_reduced, dtype=torch.float32)


def lift_filters_from_pca(filters, pca):
    """Map PCA-space filters back into the original feature space."""
    filters = np.asarray(filters, dtype=np.float32)

    if filters.ndim == 2:
        return np.asarray(pca.inverse_transform(filters), dtype=np.float32)

    if filters.ndim == 3:
        lifted = [pca.inverse_transform(filter_set) for filter_set in filters]
        return np.asarray(lifted, dtype=np.float32)

    raise ValueError(
        f"Unexpected filter dimensionality: {filters.ndim}, expected 2 or 3"
    )
