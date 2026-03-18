"""Training helpers shared across the paper analyses."""

import time

import numpy as np
import torch


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
