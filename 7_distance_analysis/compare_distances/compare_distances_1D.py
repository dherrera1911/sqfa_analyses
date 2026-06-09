"""Make plots of how distances scale with mean differences and variance differences."""
import torch
import matplotlib.pyplot as plt

from sqfa.distances import (
    bhattacharyya,
    fisher_rao_lower_bound,
    fisher_rao_same_cov,
    hellinger,
)

###################
# Plot distances vs mean differences
###################

MAX_DIST = 10.0
N_POINTS = 41

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

means = build_line_means(N_POINTS, MAX_DIST)
shared_covariance = torch.eye(means.size(1)).expand(N_POINTS, -1, -1).clone()

reference_mean = means[0]
euclidean = torch.linalg.norm(means - reference_mean, dim=1)

reference_stats = make_stats(means[0:1], shared_covariance[0:1])
all_stats = make_stats(means, shared_covariance)

fisher_rao = fisher_rao_same_cov(reference_stats, all_stats).squeeze(0)
hellinger = hellinger(reference_stats, all_stats).squeeze(0)
bhatt = bhattacharyya(reference_stats, all_stats).squeeze(0)

plt.figure(figsize=(4, 3))
plt.plot(euclidean.numpy(), fisher_rao.numpy(), label="Fisher–Rao")
plt.plot(euclidean.numpy(), hellinger.numpy(), label="Hellinger")
plt.plot(euclidean.numpy(), bhatt.numpy(), label="Bhattacharyya")
plt.xlabel("Mahalanobis distance")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.savefig('distances_means.pdf')
#plt.show()


#############
# Plot distances vs variance ratio
#############

lim = 3.0
eigval_log = torch.linspace(-lim, lim, 51)
eigval = 10**eigval_log

fisher_rao = torch.sqrt(torch.log(eigval)**2) / torch.sqrt(torch.tensor(2.0))
bhatt = torch.log((1 + eigval) / 2) - 0.5 * torch.log(eigval)
bhatt = bhatt * torch.sign(bhatt)
hellinger = torch.sqrt(1 - torch.exp(-bhatt))

plt.figure(figsize=(4, 3))
plt.plot(eigval_log, fisher_rao.numpy(), label="Fisher–Rao")
plt.plot(eigval_log, hellinger.numpy(), label="Hellinger")
plt.plot(eigval_log, bhatt.numpy(), label="Bhattacharyya")
plt.xlabel(r'$\log_{10}(\sigma_1^2/\sigma_2^2)$')
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()
plt.savefig('distances_variance.pdf')
#plt.show()

