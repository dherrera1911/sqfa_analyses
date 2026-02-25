import matplotlib.pyplot as plt
import numpy as np
import torch
import sqfa


def plot_data_covariances(ax, covariances, means=None):
    """Plot class covariances as ellipses and class means as points."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])

    sqfa.plot.statistics_ellipses(
        ellipses=covariances,
        centers=means,
        dim_pair=[0, 1],
        ax=ax,
    )
    sqfa.plot.scatter_data(
        data=means,
        labels=torch.arange(means.shape[0]),
        dim_pair=[0, 1],
        ax=ax,
    )


def plot_filter(ax, filter_vec, color, label):
    """Plot one filter as an arrow in 2D space."""
    awidth = 0.05
    ax.arrow(
        0.0,
        0.0,
        filter_vec[0].item(),
        filter_vec[1].item(),
        width=awidth,
        head_width=awidth * 5,
        color=color,
        label=label,
        length_includes_head=True,
    )


def normalize_and_orient(filter_vec):
    """Normalize filter and orient it to point roughly to positive x."""
    filter_vec = filter_vec / torch.linalg.norm(filter_vec)
    if filter_vec[0] < 0:
        filter_vec = -filter_vec
    return filter_vec


def project_statistics(means, covariances, filter_vec):
    """Project class means/covariances onto a 1D filter direction."""
    projected_means = means @ filter_vec
    projected_variances = torch.einsum("i,nij,j->n", filter_vec, covariances, filter_vec)
    projected_variances = torch.clamp(projected_variances, min=1e-10)
    return projected_means, projected_variances


def class_colors(n_classes):
    """Match sqfa.plot default class colors (viridis over class indices)."""
    values = np.arange(n_classes)
    color_map = plt.get_cmap("viridis")
    if n_classes == 1:
        return color_map(np.array([0.5]))
    color_normalizer = plt.Normalize(vmin=values.min(), vmax=values.max())
    return color_map(color_normalizer(values))


def plot_projected_distributions(ax, projected_means, projected_variances, colors):
    """Plot 1D Gaussian densities for each class."""
    std = torch.sqrt(projected_variances)
    x_min = torch.min(projected_means - 4.0 * std).item()
    x_max = torch.max(projected_means + 4.0 * std).item()
    x = torch.linspace(x_min, x_max, 1000)

    for class_id in range(projected_means.shape[0]):
        mu = projected_means[class_id]
        var = projected_variances[class_id]
        y = torch.exp(-0.5 * (x - mu) ** 2 / var) / torch.sqrt(2.0 * torch.pi * var)
        ax.plot(x, y, color=colors[class_id])


# Same toy problem as lda_template.py
x_dist = 0.8
class_means = torch.tensor(
    [
        [-x_dist, 0.2],
        [-x_dist * 1.05, -0.2],
        [x_dist * 1.05, 0.6],
        [x_dist, -0.6],
    ]
)
class_covariances = torch.diag(torch.tensor([0.008, 0.008]))
class_covariances = torch.stack([class_covariances] * 4)
colors = class_colors(class_means.shape[0])

# LDA: keep top generalized eigenvector
scatter_within = torch.mean(class_covariances, dim=0)
scatter_between = class_means.T @ class_means
lda_eigvec, _ = sqfa.linalg.generalized_eigenvectors(scatter_between, scatter_within)
lda_filter = normalize_and_orient(lda_eigvec[:, 0])

# SQFA: learn one filter on the same means/covariances
torch.manual_seed(7)
sqfa_model = sqfa.model.SQFA(
    n_dim=2,
    n_filters=1,
    feature_noise=0.001,
)
sqfa_model.fit(
    data_statistics={"means": class_means, "covariances": class_covariances},
    show_progress=False,
)
sqfa_filter = normalize_and_orient(sqfa_model.filters.detach()[0])

# Projected class distributions for each filter
lda_proj_means, lda_proj_vars = project_statistics(
    class_means, class_covariances, lda_filter
)
sqfa_proj_means, sqfa_proj_vars = project_statistics(
    class_means, class_covariances, sqfa_filter
)

print("Projected class distributions onto LDA filter:")
for class_id, (mu, var) in enumerate(zip(lda_proj_means, lda_proj_vars)):
    print(f"  Class {class_id}: mean={mu.item(): .4f}, var={var.item(): .6f}")

print("Projected class distributions onto SQFA filter:")
for class_id, (mu, var) in enumerate(zip(sqfa_proj_means, sqfa_proj_vars)):
    print(f"  Class {class_id}: mean={mu.item(): .4f}, var={var.item(): .6f}")

# 2D toy data with both filters
fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plot_filter(ax, lda_filter * 0.95, color="tab:red", label="LDA filter")
plot_filter(ax, sqfa_filter * 0.95, color="tab:blue", label="SQFA filter")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("sqfa_vs_lda.pdf")
plt.close()

# 1D projected distributions onto LDA filter
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
plot_projected_distributions(ax, lda_proj_means, lda_proj_vars, colors)
ax.set_xlabel("Projection onto LDA filter")
ax.set_ylabel("Probability density")
ax.set_xlim(-1.1, 1.1)
plt.tight_layout()
plt.savefig("lda_projection.pdf")
plt.close()

# 1D projected distributions onto SQFA filter
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
plot_projected_distributions(ax, sqfa_proj_means, sqfa_proj_vars, colors)
ax.set_xlabel("Projection onto SQFA filter")
ax.set_ylabel("Probability density")
ax.set_xlim(-1.1, 1.1)
plt.tight_layout()
plt.savefig("sqfa_projection.pdf")
plt.close()
