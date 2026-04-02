"""Helpers for sweeping regularization settings and comparing validation accuracy."""

import os

import numpy as np
import torch
from metric_learn import LFDA
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from .data import train_val_split
from .evaluation import qda_accuracy
from .training import fit_sqfa_adaptive_precision
from .wda import fit_wda


def validate_regularization(
    model_factory,
    x_train,
    y_train,
    x_val,
    y_val,
    noise_vals,
    eval_qda_noise=0,
    eval_qda_reg=1.0e-5,
    fit_kwargs=None,
    dtypes=(torch.float32, torch.float64),
    run_label="regularization sweep",
):
    """
    Perform cross-validation for the regularization parameter
    """
    n_vals = len(noise_vals)
    accuracies = torch.zeros(n_vals)
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    for i, noise in enumerate(noise_vals):
        noise_value = float(noise.item())
        model, _elapsed, train_dtype = fit_sqfa_adaptive_precision(
            model_factory=lambda noise_value=noise_value: model_factory(noise_value),
            x_train=x_train,
            y_train=y_train,
            fit_kwargs=fit_kwargs,
            dtypes=dtypes,
            run_label=f"{run_label}, noise={noise_value}",
        )

        acc = qda_accuracy(
            x_train.to(dtype=train_dtype),
            y_train,
            x_val.to(dtype=train_dtype),
            y_val,
            model.filters.detach(),
            eval_qda_noise=eval_qda_noise,
            eval_qda_reg=eval_qda_reg,
        )
        accuracies[i] = acc
    return accuracies


def validate_lfda_k(
    x_train,
    y_train,
    k_vals,
    n_filters,
    n_pca_components,
    eval_qda_reg=1.0e-5,
    val_size=0.2,
    embedding_type="weighted",
):
    """Validate the LFDA neighborhood parameter k with a PCA->LFDA->QDA pipeline."""
    k_vals = torch.as_tensor(k_vals, dtype=torch.int64)
    if k_vals.ndim != 1:
        raise ValueError("k_vals must be a one-dimensional vector of candidate values")

    x_train_fit, y_train_fit, x_val, y_val = train_val_split(
        torch.as_tensor(x_train),
        torch.as_tensor(y_train),
        val_size=val_size,
    )

    x_train_np = x_train_fit.detach().cpu().numpy()
    y_train_np = y_train_fit.detach().cpu().numpy()
    x_val_np = x_val.detach().cpu().numpy()
    y_val_np = y_val.detach().cpu().numpy()
    accuracies = torch.empty(len(k_vals), dtype=torch.float32)
    pipeline = Pipeline(
        [
            ("pca", PCA(n_components=n_pca_components)),
            ("lfda", LFDA(n_components=n_filters, embedding_type=embedding_type)),
            (
                "qda",
                QuadraticDiscriminantAnalysis(
                    solver="eigen",
                    shrinkage=eval_qda_reg,
                    tol=1.0e-7,
                ),
            ),
        ]
    )

    for i, k_value in enumerate(k_vals):
        model = clone(pipeline)
        model.set_params(lfda__k=int(k_value.item()))
        model.fit(x_train_np, y_train_np)
        accuracies[i] = model.score(x_val_np, y_val_np)
    return accuracies


def load_or_validate_noise(
    noise_path,
    model_factory,
    x_train,
    y_train,
    x_val,
    y_val,
    noise_vals,
    fit_kwargs=None,
    dtypes=(torch.float32, torch.float64),
    run_label="regularization sweep",
    eval_qda_noise=0,
    eval_qda_reg=1.0e-5,
):
    """Load a saved noise value or tune, save, and return it if missing."""
    if os.path.exists(noise_path):
        return float(np.load(noise_path).item())

    noise_accs = validate_regularization(
        model_factory=model_factory,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        noise_vals=noise_vals,
        eval_qda_noise=eval_qda_noise,
        eval_qda_reg=eval_qda_reg,
        fit_kwargs=fit_kwargs,
        dtypes=dtypes,
        run_label=run_label,
    )
    best_noise = float(noise_vals[torch.argmax(noise_accs)].item())
    np.save(noise_path, np.asarray(best_noise))
    return best_noise


def validate_lda_shrinkage(
    x_train,
    y_train,
    x_val,
    y_val,
    shrinkage_vals,
    n_filters,
    eval_qda_reg=1.0e-5,
):
    """Validate LDA shrinkage values with downstream QDA on a validation split."""
    shrinkage_vals = np.asarray(shrinkage_vals, dtype=float).reshape(-1)
    if shrinkage_vals.size == 0:
        raise ValueError("shrinkage_vals must contain at least one candidate value")

    accuracies = torch.empty(shrinkage_vals.size, dtype=torch.float32)
    x_train_np = torch.as_tensor(x_train).detach().cpu().numpy()
    y_train_np = torch.as_tensor(y_train).detach().cpu().numpy()

    for idx, shrinkage in enumerate(shrinkage_vals):
        lda = LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage=float(shrinkage),
        )
        lda.fit(x_train_np, y_train_np)
        filters = np.asarray(lda.scalings_.T, dtype=np.float32)
        if filters.shape[0] == 0:
            accuracies[idx] = float("-inf")
            continue

        filters = filters[: min(n_filters, filters.shape[0])]
        accuracies[idx] = qda_accuracy(
            x_train,
            y_train,
            x_val,
            y_val,
            filters,
            eval_qda_reg=eval_qda_reg,
        )

    return accuracies


def load_or_validate_lda_shrinkage(
    shrinkage_path,
    x_train,
    y_train,
    x_val,
    y_val,
    shrinkage_vals,
    n_filters,
    eval_qda_reg=1.0e-5,
):
    """Load a saved LDA shrinkage value or select and cache it."""
    if os.path.exists(shrinkage_path):
        return float(np.load(shrinkage_path).item())

    shrinkage_accs = validate_lda_shrinkage(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        shrinkage_vals=shrinkage_vals,
        n_filters=n_filters,
        eval_qda_reg=eval_qda_reg,
    )
    best_shrinkage = float(shrinkage_vals[int(torch.argmax(shrinkage_accs).item())])
    np.save(shrinkage_path, np.asarray(best_shrinkage))
    return best_shrinkage


def validate_wda_reg(
    x_train,
    y_train,
    x_val,
    y_val,
    reg_vals,
    n_filters,
    n_pca_components,
    samples_per_class,
    eval_qda_reg=1.0e-5,
    seed=0,
    sinkhorn_iters=10,
    maxiter=40,
    sinkhorn_method="sinkhorn_log",
    solver="steepest",
    normalize=True,
    pca_solver="randomized",
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
    """Validate WDATorch regularization values with downstream QDA."""
    reg_vals = np.asarray(reg_vals, dtype=float).reshape(-1)
    if reg_vals.size == 0:
        raise ValueError("reg_vals must contain at least one candidate value")

    accuracies = torch.empty(reg_vals.size, dtype=torch.float32)
    for idx, reg in enumerate(reg_vals):
        filters, _elapsed = fit_wda(
            x_train=x_train,
            y_train=y_train,
            n_filters=n_filters,
            n_pca_components=n_pca_components,
            reg=float(reg),
            samples_per_class=samples_per_class,
            seed=seed,
            sinkhorn_iters=sinkhorn_iters,
            maxiter=maxiter,
            sinkhorn_method=sinkhorn_method,
            solver=solver,
            normalize=normalize,
            pca_solver=pca_solver,
            gradient_mode=gradient_mode,
            sinkhorn_tolerance=sinkhorn_tolerance,
            optimizer=optimizer,
            lr=lr,
            lbfgs_max_iter=lbfgs_max_iter,
            loss_change_tol=loss_change_tol,
            update_mean_each_step=update_mean_each_step,
            dtype=dtype,
            device=device,
        )
        accuracies[idx] = qda_accuracy(
            x_train,
            y_train,
            x_val,
            y_val,
            filters,
            eval_qda_reg=eval_qda_reg,
        )
    return accuracies


def load_or_validate_wda_reg(
    reg_path,
    x_train,
    y_train,
    x_val,
    y_val,
    reg_vals,
    n_filters,
    n_pca_components,
    samples_per_class,
    eval_qda_reg=1.0e-5,
    seed=0,
    sinkhorn_iters=10,
    maxiter=40,
    sinkhorn_method="sinkhorn_log",
    solver="steepest",
    normalize=True,
    pca_solver="randomized",
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
    """Load a saved WDA regularization value or select and cache it."""
    if os.path.exists(reg_path):
        return float(np.load(reg_path).item())

    reg_accs = validate_wda_reg(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        reg_vals=reg_vals,
        n_filters=n_filters,
        n_pca_components=n_pca_components,
        samples_per_class=samples_per_class,
        eval_qda_reg=eval_qda_reg,
        seed=seed,
        sinkhorn_iters=sinkhorn_iters,
        maxiter=maxiter,
        sinkhorn_method=sinkhorn_method,
        solver=solver,
        normalize=normalize,
        pca_solver=pca_solver,
        gradient_mode=gradient_mode,
        sinkhorn_tolerance=sinkhorn_tolerance,
        optimizer=optimizer,
        lr=lr,
        lbfgs_max_iter=lbfgs_max_iter,
        loss_change_tol=loss_change_tol,
        update_mean_each_step=update_mean_each_step,
        dtype=dtype,
        device=device,
    )
    best_reg = float(reg_vals[int(torch.argmax(reg_accs).item())])
    np.save(reg_path, np.asarray(best_reg))
    return best_reg
