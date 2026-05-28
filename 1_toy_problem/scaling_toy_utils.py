"""Utilities for toy analyses of SQFA scaling behavior."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import sqfa


TOY_DIR = Path(__file__).resolve().parent
FIGURES_DIR = TOY_DIR / "figures"
FILTERS_DIR = TOY_DIR / "filters"
RESULTS_DIR = TOY_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True)
FILTERS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def make_rotation_matrix(theta_degrees: float) -> torch.Tensor:
    """Return a 2D rotation matrix for a given angle in degrees."""
    theta = torch.deg2rad(torch.as_tensor(theta_degrees, dtype=torch.float32))
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def make_random_subspace_basis(n_dim: int, subspace_dim: int, seed: int) -> torch.Tensor:
    """Generate a random orthonormal basis with `subspace_dim` columns."""
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(n_dim, subspace_dim, generator=generator)
    basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return basis


def make_informative_statistics(
    n_classes: int,
    informative_dim: int,
    mean_scale: float = 1.0,
    cov_axes=[2.2, 0.35],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic class means and covariances in an informative subspace."""
    if informative_dim < 4:
        raise ValueError("informative_dim must be at least 4")
    if (informative_dim - 4) % 2 != 0:
        raise ValueError("informative_dim - 4 must be even")

    means = torch.zeros(n_classes, informative_dim, dtype=torch.float32)
    mean_vals = torch.linspace(-mean_scale, mean_scale, n_classes)
    for dim_idx in range(4):
        scale = 1.0 - 0.15 * dim_idx
        means[:, dim_idx] = mean_vals * scale

    covariances = torch.stack(
        [torch.eye(informative_dim, dtype=torch.float32) for _ in range(n_classes)]
    )
    angle_vals = torch.linspace(0.0, 144.0, n_classes)
    n_cov_blocks = (informative_dim - 4) // 2
    for block_idx in range(n_cov_blocks):
        start = 4 + 2 * block_idx
        stretch_major = cov_axes[0]
        stretch_minor = cov_axes[1]
        base_cov = torch.diag(torch.tensor([stretch_major, stretch_minor]))
        block_offset = 11.0 * block_idx
        for class_idx, angle in enumerate(angle_vals):
            rotation = make_rotation_matrix(float(angle.item()) + block_offset)
            rotated_cov = rotation @ base_cov @ rotation.T
            covariances[class_idx, start : start + 2, start : start + 2] = rotated_cov

    return means, covariances


def embed_statistics_in_ambient_space(
    means: torch.Tensor,
    covariances: torch.Tensor,
    basis: torch.Tensor,
    null_variance: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed informative statistics into a dense ambient space."""
    basis = torch.as_tensor(basis, dtype=means.dtype)
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix of shape (ambient_dim, informative_dim)")
    if basis.shape[1] != means.shape[1]:
        raise ValueError("basis and means must agree on the informative dimensionality")

    ambient_dim = basis.shape[0]
    n_classes = means.shape[0]
    projector = basis @ basis.T
    identity = torch.eye(ambient_dim, dtype=means.dtype)
    null_covariance = null_variance * (identity - projector)

    embedded_means = means @ basis.T
    embedded_covariances = torch.empty(
        n_classes,
        ambient_dim,
        ambient_dim,
        dtype=means.dtype,
    )
    for class_idx in range(n_classes):
        embedded_covariances[class_idx] = (
            basis @ covariances[class_idx] @ basis.T + null_covariance
        )

    return embedded_means, embedded_covariances


def fit_sqfa_from_statistics(
    means: torch.Tensor,
    covariances: torch.Tensor,
    n_filters: int,
    feature_noise: float,
    constraint: str = "sphere",
    fit_kwargs: dict | None = None,
    dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
    seed: int = 0,
) -> tuple[sqfa.model.SQFA, float]:
    """Fit SQFA directly to class statistics with a simple precision fallback."""
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    last_error = None

    for dtype in dtypes:
        try:
            torch.manual_seed(seed)
            model = sqfa.model.SQFA(
                n_dim=means.shape[1],
                n_filters=n_filters,
                feature_noise=feature_noise,
                constraint=constraint,
            ).to(dtype=dtype)
            stats = {
                "means": means.to(dtype=dtype),
                "covariances": covariances.to(dtype=dtype),
            }
            start = time.time()
            model.fit(data_statistics=stats, **fit_kwargs)
            return model, time.time() - start
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            if dtype != dtypes[-1]:
                print(f"SQFA fit failed in {dtype} ({exc}). Retrying in higher precision.")
            else:
                raise

    raise last_error


def orthonormalize_filters(filters: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis spanning the same filter row space."""
    filters = torch.as_tensor(filters)
    q, _ = torch.linalg.qr(filters.T, mode="reduced")
    return q.T


def project_statistics(
    means: torch.Tensor,
    covariances: torch.Tensor,
    filters: torch.Tensor,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project class statistics into a feature space defined by `filters`."""
    filters = torch.as_tensor(filters, dtype=means.dtype)
    projected_means = means @ filters.T
    projected_covariances = torch.einsum("fd,ndk,gk->nfg", filters, covariances, filters)
    eye = torch.eye(filters.shape[0], dtype=projected_covariances.dtype)
    projected_covariances = projected_covariances + jitter * eye[None, :, :]
    return projected_means, projected_covariances


def simulate_qda_accuracy_from_statistics(
    means: torch.Tensor,
    covariances: torch.Tensor,
    n_train_per_class: int,
    n_test_per_class: int,
    seed: int,
    qda_reg: float = 0.0,
) -> float:
    """Sample Gaussian data from class statistics and evaluate QDA accuracy."""
    x_train, y_train, x_test, y_test = sample_qda_dataset_from_statistics(
        means=means,
        covariances=covariances,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        seed=seed,
    )
    return qda_accuracy_from_samples(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        qda_reg=qda_reg,
    )


def sample_qda_dataset_from_statistics(
    means: torch.Tensor,
    covariances: torch.Tensor,
    n_train_per_class: int,
    n_test_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one train/test split from the class statistics."""
    generator = torch.Generator().manual_seed(seed)
    n_classes = means.shape[0]
    n_dim = means.shape[1]
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for class_idx in range(n_classes):
        jitter = 1.0e-6 * torch.eye(n_dim, dtype=covariances.dtype)
        cholesky = torch.linalg.cholesky(covariances[class_idx] + jitter)
        train_noise = torch.randn(
            n_train_per_class,
            n_dim,
            generator=generator,
            dtype=means.dtype,
        )
        test_noise = torch.randn(
            n_test_per_class,
            n_dim,
            generator=generator,
            dtype=means.dtype,
        )
        train_samples.append(means[class_idx] + train_noise @ cholesky.T)
        test_samples.append(means[class_idx] + test_noise @ cholesky.T)
        train_labels.append(torch.full((n_train_per_class,), class_idx))
        test_labels.append(torch.full((n_test_per_class,), class_idx))

    x_train = torch.cat(train_samples).cpu().numpy()
    y_train = torch.cat(train_labels).cpu().numpy()
    x_test = torch.cat(test_samples).cpu().numpy()
    y_test = torch.cat(test_labels).cpu().numpy()

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def qda_accuracy_from_samples(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    filters: torch.Tensor | None = None,
    qda_reg: float = 0.0,
) -> float:
    """Fit scikit-learn QDA on a shared sampled dataset."""
    x_train = torch.as_tensor(x_train, dtype=torch.float64)
    x_test = torch.as_tensor(x_test, dtype=torch.float64)

    if filters is not None:
        filters = torch.as_tensor(filters, dtype=torch.float64)
        x_train = x_train @ filters.T
        x_test = x_test @ filters.T

    x_train_np = x_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    x_test_np = x_test.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    qda = QuadraticDiscriminantAnalysis(
        solver="eigen",
        shrinkage=qda_reg,
        tol=1.0e-7,
    )
    qda.fit(x_train_np, y_train_np)
    return float(qda.score(x_test_np, y_test_np))


def save_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    """Write a list of dictionaries to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_filters(filters: torch.Tensor, output_path: Path) -> None:
    """Save learned filters as a NumPy array."""
    np.save(output_path, filters.detach().cpu().numpy())


def plot_accuracy_comparison(
    x_values: list[int],
    true_accs: list[float],
    sqfa_accs: list[float],
    xlabel: str,
    output_path: Path,
    xscale: str = "linear",
) -> None:
    """Plot QDA accuracy in the true and SQFA-learned subspaces."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(x_values, np.asarray(true_accs) * 100.0, marker="o", linewidth=2, label="True subspace")
    ax.plot(x_values, np.asarray(sqfa_accs) * 100.0, marker="o", linewidth=2, label="SQFA")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("QDA Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xscale(xscale)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
