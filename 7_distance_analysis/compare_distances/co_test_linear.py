"""Compare Calvo-Oller bound to true FR distance for same-covariance case"""

import torch
import matplotlib.pyplot as plt

from sqfa.distances import (
    bhattacharyya,
    fisher_rao_lower_bound,
    fisher_rao_same_cov,
    hellinger,
)


RAD = 10.0
N_CLASSES = 41
REF_INDEX = 0


def build_line_means(n_classes, length):
    """Place class means equally spaced along a one-dimensional line."""
    positions = torch.linspace(0.0, length, n_classes)
    base = torch.zeros(4)
    direction = torch.randn(4)
    direction = direction / direction.norm()
    return base.unsqueeze(0) + positions.unsqueeze(1) * direction


def make_stats(means, covariances):
    """Package means and covariances as expected by sqfa distance helpers."""
    return {"means": means, "covariances": covariances}


torch.manual_seed(0)

means = build_line_means(N_CLASSES, RAD)
shared_covariance = torch.eye(means.size(1)).expand(N_CLASSES, -1, -1).clone()

reference_mean = means[REF_INDEX]
euclidean = torch.linalg.norm(means - reference_mean, dim=1)

reference_stats = make_stats(means[REF_INDEX:REF_INDEX + 1],
                             shared_covariance[REF_INDEX:REF_INDEX + 1])
all_stats = make_stats(means, shared_covariance)

fr_exact = fisher_rao_same_cov(reference_stats, all_stats).squeeze(0)
fr_lower = fisher_rao_lower_bound(reference_stats, all_stats).squeeze(0)
hellinger_dist = hellinger(reference_stats, all_stats).squeeze(0)
bhatt = bhattacharyya(reference_stats, all_stats).squeeze(0)

class_indices = torch.arange(N_CLASSES)

# Plot Calvo-Oller vs Fisher-Rao distance
max_dist = fr_exact.max().item()
plt.figure(figsize=(3, 3))
plt.plot([0, max_dist], [0, max_dist], color='black', linestyle='--')
plt.plot(fr_exact, fr_lower)
plt.ylabel('Calvo-Oller Approximation')
plt.xlabel('Fisher-Rao Distance')
# Make identity line
plt.xlim(0, max_dist)
plt.ylim(0, max_dist)
plt.tight_layout()
plt.savefig('co_vs_fisherrao_linear.pdf')
plt.close()

