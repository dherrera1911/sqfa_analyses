"""Utilities for fitting WDATorch-based Wasserstein Discriminant Analysis."""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from .wdatorch import WDATorch


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
    sinkhorn_method="sinkhorn_log",
    solver="steepest",
    normalize=True,
    pca_solver="randomized",
    verbose=False,
    gradient_mode="autodiff",
    sinkhorn_tolerance=1.0e-9,
    optimizer=None,
    lr=1.0,
    lbfgs_max_iter=5,
    loss_change_tol=1.0e-7,
    update_mean_each_step=False,
    dtype=torch.float64,
    device="cpu",
):
    """Fit WDATorch after PCA pre-reduction and return filters in original space."""
    x_train_np = torch.as_tensor(x_train).detach().cpu().numpy()
    y_train_np = torch.as_tensor(y_train).detach().cpu().numpy()

    n_pca_components = int(
        min(n_pca_components, x_train_np.shape[0], x_train_np.shape[1])
    )
    if n_filters <= 0:
        raise ValueError("n_filters must be positive")
    if n_pca_components < n_filters:
        raise ValueError("n_pca_components must be at least as large as n_filters")

    pca_kwargs = {"n_components": n_pca_components, "svd_solver": pca_solver}
    if pca_solver == "randomized":
        pca_kwargs["random_state"] = seed
    pca = PCA(**pca_kwargs)
    x_train_pca = pca.fit_transform(x_train_np)

    subset_idx = balanced_subset_indices(
        y_train_np,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    x_wda = np.asarray(x_train_pca[subset_idx], dtype=np.float64)
    y_wda = np.asarray(y_train_np[subset_idx], dtype=np.int64)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    device = torch.device(device)

    x_wda_t = torch.as_tensor(x_wda, dtype=dtype, device=device)
    y_wda_t = torch.as_tensor(y_wda, dtype=torch.long, device=device)

    # `solver` and `normalize` are kept for compatibility with older call sites.
    if optimizer is None:
        if solver == "trustregions":
            raise ValueError("solver='trustregions' is not supported by WDATorch.")
        optimizer_obj = None
    elif isinstance(optimizer, str):
        if optimizer == "adam":
            optimizer_obj = torch.optim.Adam
        elif optimizer == "lbfgs":
            optimizer_obj = None
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        optimizer_obj = optimizer

    model = WDATorch(
        input_dim=x_wda.shape[1],
        output_dim=n_filters,
        reg=float(reg),
        sinkhorn_method=sinkhorn_method,
        sinkhorn_iterations=int(sinkhorn_iters),
        sinkhorn_tolerance=float(sinkhorn_tolerance),
        gradient_mode=gradient_mode,
        device=device,
        dtype=dtype,
    )

    optimizer_instance = None
    if optimizer_obj is torch.optim.Adam:
        optimizer_instance = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_obj is not None:
        optimizer_instance = optimizer_obj(model.parameters())

    start = time.time()
    model.fit(
        x_wda_t,
        y_wda_t,
        num_steps=int(maxiter),
        lr=lr,
        optimizer=optimizer_instance,
        lbfgs_max_iter=int(lbfgs_max_iter),
        loss_change_tol=float(loss_change_tol),
        update_mean_each_step=update_mean_each_step,
        verbose=bool(verbose),
        print_every=25,
    )
    elapsed = time.time() - start

    projection = model.projection_matrix.detach().cpu().numpy()
    filters = projection.T @ pca.components_
    return np.asarray(filters, dtype=np.float32), elapsed
