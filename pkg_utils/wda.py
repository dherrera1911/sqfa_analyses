"""Utilities for fitting POT Wasserstein Discriminant Analysis baselines."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.decomposition import PCA


def _get_pot_wda():
    try:
        from ot.dr import wda as pot_wda
    except ImportError as exc:
        raise ImportError(
            "WDA requires POT's `dr` extra. Install it with `uv add \"pot[dr]\"` "
            "or ensure `ot`, `autograd`, and `pymanopt` are available."
        ) from exc
    return pot_wda


def balanced_subset_indices(labels, samples_per_class, seed=0):
    """Return a balanced index subset with up to `samples_per_class` per class."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if samples_per_class is None or samples_per_class <= 0:
        return np.arange(labels.shape[0])

    rng = np.random.default_rng(seed)
    subset_indices = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        n_take = min(int(samples_per_class), label_indices.size)
        chosen = rng.choice(label_indices, size=n_take, replace=False)
        subset_indices.append(np.sort(chosen))

    return np.concatenate(subset_indices)


def _initial_projection(n_dim, n_filters):
    initial = np.zeros((n_dim, n_filters), dtype=np.float64)
    initial[:n_filters, :] = np.eye(n_filters, dtype=np.float64)
    return initial


def fit_wda(
    x_train,
    y_train,
    n_filters,
    n_pca_components,
    reg=1.0,
    samples_per_class=200,
    seed=0,
    sinkhorn_iters=10,
    maxiter=40,
    sinkhorn_method="sinkhorn",
    solver="steepest",
    normalize=True,
    pca_solver="randomized",
    verbose=False,
):
    """Fit POT WDA after PCA pre-reduction and return filters in the original space.

    This helper includes balanced class subsampling internally because POT's
    WDA optimization becomes expensive on large datasets, and using a balanced
    subset keeps the learned projection from being driven disproportionately by
    classes with more training examples.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training inputs in the original feature space. Can be a NumPy array or
        a PyTorch tensor.
    y_train : array-like of shape (n_samples,)
        Class labels associated with ``x_train``.
    n_filters : int
        Number of WDA directions to learn. This is the dimensionality of the
        reduced feature space returned by the learned filters.
    n_pca_components : int
        Number of PCA components used before WDA. WDA is fitted in this PCA
        space rather than in the original feature space for tractability. The
        effective value is clipped to the valid range allowed by the data.
    reg : float, optional
        Entropic regularization parameter used by POT's Sinkhorn-based WDA
        objective. Smaller values give a sharper OT geometry but can be less
        stable numerically; larger values smooth the transport problem more.
    samples_per_class : int, optional
        Maximum number of training examples per class used to fit WDA after the
        PCA step. This balanced subsampling is used because POT's WDA scales
        poorly with the full dataset size. Use ``None`` or a non-positive value
        to disable subsampling and keep all samples.
    seed : int, optional
        Random seed used for the PCA randomized solver and for balanced class
        subsampling.
    sinkhorn_iters : int, optional
        Number of Sinkhorn iterations used inside each OT computation in POT's
        WDA objective.
    maxiter : int, optional
        Maximum number of manifold-optimization iterations used by POT to
        optimize the WDA projection.
    sinkhorn_method : {"sinkhorn", "sinkhorn_log"}, optional
        Sinkhorn implementation used inside POT. ``"sinkhorn_log"`` is usually
        safer for small ``reg`` values.
    solver : {"steepest", "trustregions"}, optional
        Manifold optimizer passed to POT. ``"steepest"`` maps to POT's default
        steepest-descent optimizer and ``"trustregions"`` maps to POT's trust
        regions optimizer.
    normalize : bool, optional
        Whether POT should normalize the entropic regularization scale by an
        average pairwise distance computed from the initialization.
    pca_solver : str, optional
        PCA solver passed to scikit-learn's ``PCA`` implementation.
    verbose : bool or int, optional
        Verbosity forwarded to POT's optimizer.

    Returns
    -------
    filters : ndarray of shape (n_filters, n_features)
        Learned WDA directions expressed back in the original input space, so
        they can be used by the rest of the project exactly like the other
        linear filter methods.
    elapsed : float
        Wall-clock training time in seconds for the WDA optimization step. This
        excludes data loading and the downstream evaluation code.
    """
    # Convert inputs to NumPy arrays because POT and scikit-learn operate there.
    x_train_np = torch.as_tensor(x_train).detach().cpu().numpy()
    y_train_np = torch.as_tensor(y_train).detach().cpu().numpy()

    # Enforce a valid PCA dimensionality and basic shape constraints.
    n_pca_components = int(
        min(n_pca_components, x_train_np.shape[0], x_train_np.shape[1])
    )
    if n_filters <= 0:
        raise ValueError("n_filters must be positive")
    if n_pca_components < n_filters:
        raise ValueError("n_pca_components must be at least as large as n_filters")

    # First reduce the original feature space with PCA to make WDA feasible on
    # high-dimensional datasets such as MNIST or SVHN.
    pca_kwargs = {"n_components": n_pca_components, "svd_solver": pca_solver}
    if pca_solver == "randomized":
        pca_kwargs["random_state"] = seed
    pca = PCA(**pca_kwargs)
    x_train_pca = pca.fit_transform(x_train_np)

    # Build a balanced subset in PCA space so WDA sees a similar number of
    # examples per class while keeping the optimization problem manageable.
    subset_idx = balanced_subset_indices(
        y_train_np,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    x_wda = np.asarray(x_train_pca[subset_idx], dtype=np.float64)
    y_wda = np.asarray(y_train_np[subset_idx], dtype=np.int64)

    # Configure POT's WDA solver and fit the projection in the PCA space.
    pot_wda = _get_pot_wda()
    solver_name = None if solver == "steepest" else "TrustRegions"

    start = time.time()
    projection, _proj = pot_wda(
        x_wda.copy(),
        y_wda,
        p=n_filters,
        reg=float(reg),
        k=int(sinkhorn_iters),
        solver=solver_name,
        sinkhorn_method=sinkhorn_method,
        maxiter=int(maxiter),
        verbose=verbose,
        P0=_initial_projection(x_wda.shape[1], n_filters),
        normalize=normalize,
    )
    elapsed = time.time() - start

    # Map the learned PCA-space projection back to the original input space so
    # downstream code can apply the filters directly to the original features.
    filters = projection.T @ pca.components_
    return np.asarray(filters, dtype=np.float32), elapsed
