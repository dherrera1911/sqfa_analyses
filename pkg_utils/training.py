"""Training helpers shared across the paper analyses."""

import time

import numpy as np
import torch
from metric_learn import LFDA, LMNN

from .pca import fit_preprocessing_pca, lift_filters_from_pca, transform_with_pca


def fit_sqfa_adaptive_precision(
    model_factory,
    x_train,
    y_train,
    fit_kwargs=None,
    dtypes=(torch.float32, torch.float64),
    run_label="fit",
):
    """Fit a fresh SQFA model, retrying in higher precision if needed."""
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    last_error = None

    for dtype in dtypes:
        try:
            model = model_factory()
            x_fit = x_train.to(dtype=dtype)
            model = model.to(dtype=dtype)

            start = time.time()
            model.fit(x_fit, y_train, **fit_kwargs)
            return model, time.time() - start, dtype
        except Exception as exc:
            last_error = exc
            if dtype != dtypes[-1]:
                print(
                    f"{run_label} failed in {dtype} ({exc}). "
                    "Retrying with higher precision."
                )
            else:
                raise

    raise last_error


def train_sqfa_repeated(
    model_factory,
    x_train,
    y_train,
    n_reps,
    fit_kwargs=None,
    dtypes=(torch.float32, torch.float64),
    run_label="training",
):
    """Train fresh SQFA-style models repeatedly and collect filters and times."""
    filters = []
    times = []

    for rep in range(n_reps):
        model, elapsed, _train_dtype = fit_sqfa_adaptive_precision(
            model_factory=model_factory,
            x_train=x_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            dtypes=dtypes,
            run_label=f"{run_label}, rep={rep}",
        )
        filters.append(
            model.filters.detach().to(dtype=torch.float32).cpu().numpy()
        )
        times.append(elapsed)

    return np.asarray(filters), np.asarray(times)


def train_metric_learn_repeated(
    estimator_factory,
    x_train,
    y_train,
    n_reps,
    fit_x=None,
    extract_filters=None,
    run_label="training",
):
    """Fit fresh metric-learn estimators repeatedly and collect filters and times."""
    fit_x = x_train if fit_x is None else fit_x
    fit_x_np = torch.as_tensor(fit_x).detach().cpu().numpy()
    y_train_np = torch.as_tensor(y_train).detach().cpu().numpy()
    extract_filters = (
        (lambda estimator: estimator.components_)
        if extract_filters is None
        else extract_filters
    )

    filters = []
    times = []

    for rep in range(n_reps):
        estimator = estimator_factory()
        start = time.time()
        estimator.fit(fit_x_np, y_train_np)
        elapsed = time.time() - start

        current_filters = np.asarray(extract_filters(estimator), dtype=np.float32)
        filters.append(current_filters)
        times.append(elapsed)
        print(f"{run_label}, rep={rep}, elapsed={elapsed:.3f}s")

    return np.asarray(filters), np.asarray(times)


def train_lfda_repeated(
    x_train,
    y_train,
    n_reps,
    n_filters,
    k,
    n_pca_components=None,
    embedding_type="orthonormalized",
    run_label="lfda training",
):
    """Train LFDA multiple times, optionally in PCA space, and return lifted filters."""
    pca = None
    fit_x = x_train

    if n_pca_components is not None:
        pca = fit_preprocessing_pca(x_train, n_pca_components)
        fit_x = transform_with_pca(pca, x_train)

    filters, times = train_metric_learn_repeated(
        estimator_factory=lambda: LFDA(
            n_components=n_filters,
            k=k,
            embedding_type=embedding_type,
        ),
        x_train=x_train,
        y_train=y_train,
        n_reps=n_reps,
        fit_x=fit_x,
        run_label=run_label,
    )

    if pca is not None:
        filters = lift_filters_from_pca(filters, pca)

    return filters, times


def train_lmnn_repeated(
    x_train,
    y_train,
    n_reps,
    n_filters,
    n_neighbors=3,
    n_pca_components=None,
    subsample_step=None,
    init="pca",
    learn_rate=1.0e-6,
    max_iter=2000,
    convergence_tol=1.0,
    verbose=False,
    run_label="lmnn training",
):
    """Train LMNN multiple times, optionally in PCA space, and return lifted filters."""
    pca = None
    fit_x = x_train

    if n_pca_components is not None:
        pca = fit_preprocessing_pca(x_train, n_pca_components)
        fit_x = transform_with_pca(pca, x_train)

    fit_y = y_train
    if subsample_step is not None and subsample_step > 1:
        fit_x = fit_x[::subsample_step]
        fit_y = y_train[::subsample_step]

    filters, times = train_metric_learn_repeated(
        estimator_factory=lambda: LMNN(
            n_neighbors=n_neighbors,
            learn_rate=learn_rate,
            n_components=n_filters,
            init=init,
            verbose=verbose,
            max_iter=max_iter,
            convergence_tol=convergence_tol,
        ),
        x_train=x_train,
        y_train=fit_y,
        n_reps=n_reps,
        fit_x=fit_x,
        run_label=run_label,
    )

    if pca is not None:
        filters = lift_filters_from_pca(filters, pca)

    return filters, times
