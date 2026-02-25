import torch
import sqfa


def make_rotation_matrix(theta_degrees: float) -> torch.Tensor:
    theta = torch.deg2rad(torch.as_tensor(theta_degrees, dtype=torch.float32))
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


torch.manual_seed(7)

n_dim = 6
n_filters = 2
feature_noise = 0.002

# Same toy statistics as Toy Example 1.
mean_val = 0.25
means = torch.tensor(
    [
        [0.0, 0.0, -mean_val, -mean_val, 0.0, 0.0],
        [0.0, 0.0, +mean_val, -mean_val, 0.0, 0.0],
        [0.0, 0.0, 0.0, +mean_val, 0.0, 0.0],
    ]
)

covariances = torch.stack([torch.eye(n_dim) for _ in range(3)])
covariances[:, 2:4, 2:4] = covariances[:, 2:4, 2:4] * 0.6
covariances[:, 4:6, 4:6] = covariances[:, 4:6, 4:6] * torch.tensor(
    [0.9, 1.00, 1.10]
).view(-1, 1, 1)

cov_ref = torch.diag(torch.tensor([0.9, 0.02]))
for ind, angle in enumerate([10, 45, 80]):
    rot_mat = make_rotation_matrix(angle)
    covariances[ind, 0:2, 0:2] = rot_mat @ cov_ref @ rot_mat.T

stats = {"means": means, "covariances": covariances}

sqfa_model = sqfa.model.SQFA(
    n_dim=n_dim,
    n_filters=n_filters,
    feature_noise=feature_noise,
)
sqfa_model.fit(data_statistics=stats, show_progress=True)

print("Learned SQFA filters (rows):")
print(sqfa_model.filters.detach())

