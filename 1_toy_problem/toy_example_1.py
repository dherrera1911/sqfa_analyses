import torch
import matplotlib.pyplot as plt
import sqfa

################
# TOY EXAMPLE 1: PCA vs LDA vs SQFA
################

torch.manual_seed(7) # Set seed for reproducibility
n_dim_pairs = 3
n_dim = n_dim_pairs * 2
noise = 0.002
lims = (-2.2, 2.2)


# GENERATE COVARIANCE MATRICES
# Define the functions to generate the covariance matrices
def make_rotation_matrix(theta):
    """Make a matrix that rotates a 2D vector by theta.

    Args:
        theta (float): Angle in degrees.
    """
    theta = torch.deg2rad(theta)
    rot_mat_2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                              [torch.sin(theta), torch.cos(theta)]])
    return rot_mat_2


# VISUALIZE
def plot_data_covariances(ax, covariances, means=None, lims=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])
    n_classes = means.shape[0]
    dim_pairs = [[0, 1], [2, 3], [4, 5]]
    for i in range(len(dim_pairs)):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax[i])
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(n_classes),
                               dim_pair=dim_pairs[i], ax=ax[i])
        dim_pairs_label = [d+1 for d in dim_pairs[i]]
        ax[i].set_aspect('equal')
        # Add names to axes
        ax[i].set_xlabel(r'$x_{}$'.format(dim_pairs_label[0]), fontsize=16)
        ax[i].set_ylabel(r'$x_{}$'.format(dim_pairs_label[1]), fontsize=16)
        if lims is not None:
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)


# VISUALIZE FILTERS ON TOP OF DATA COVARIANCES
def plot_filters(ax, filters, color='r'):
    """Plot the filters as arrows in data space."""
    # Draw the filters of sqfa as arrows on the plot
    awidth = 0.07
    for f in range(2):
        if torch.norm(filters[f, 0:2]) > 1e-2:
            ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)
        if torch.norm(filters[f, 2:4]) > 1e-2:
            ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)
        if torch.norm(filters[f, 4:6]) > 1e-2:
            ax[2].arrow(0, 0, filters[f, 4], filters[f, 5], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)


mean_val = 0.25
means = torch.tensor(
  [[0.0, 0.0, -mean_val , -mean_val, 0.0, 0.0],
  [0.0, 0.0, +mean_val, -mean_val, 0.0, 0.0],
  [0.0, 0.0, 0.0 , +mean_val, 0.0, 0.0]]
)

covs = torch.stack([torch.eye(6) for _ in range(3)])
covs[:, 2:4, 2:4] = covs[:, 2:4, 2:4] * 0.6
covs[:, 4:6, 4:6] = covs[:, 4:6, 4:6] * torch.tensor([0.9, 1.00, 1.10]).view(-1, 1, 1)
angles = [10, 45, 80]
cov_ref = torch.diag(torch.tensor([0.9, 0.02]))
for ind, angle in enumerate(angles):
    rot_mat = make_rotation_matrix(torch.as_tensor(angle))
    cov_rot = rot_mat @ cov_ref @ rot_mat.T
    covs[ind, 0:2, 0:2] = cov_rot


##### LDA ####
within_cov = covs.mean(dim=0)
means_mean = means.mean(dim=0, keepdim=True)
between_cov = (means - means_mean).T @ (means - means_mean)  / means.shape[0]
lda_filters, _ = sqfa.linalg.generalized_eigenvectors(between_cov, within_cov)
lda_filters = lda_filters[:,:2].T

#### SQFA ####
torch.manual_seed(9)
stats_dict = {
    'means': means,
    'covariances': covs,
}
sqfa_m = sqfa.model.SQFA(
  n_dim=n_dim,
  n_filters=2,
  feature_noise=noise,
)
sqfa_m.fit(
  data_statistics=stats_dict,
  show_progress=True
)
sqfa_filters = sqfa_m.filters.detach()


##### PCA ####
total_var = within_cov + between_cov
pca_filters = torch.linalg.eigh(total_var)[1][:, -2:].T


figsize = (7, 3)
fig, ax = plt.subplots(1, 3, figsize=figsize)
plot_data_covariances(ax, covs, means, lims)
plot_filters(ax, sqfa_filters, color='black')
plot_filters(ax, lda_filters, color='g')
plot_filters(ax, pca_filters, color='blue')
plt.tight_layout()
plt.savefig('figures/toy_1.pdf')
plt.show()
plt.close()


#######################
# TOY EXAMPLE 2: SQFA vs smSQFA
#######################

torch.manual_seed(4) # Set seed for reproducibility
noise = 0.001
figsize = (5, 3)
lims = (-2.2, 2.2)


# GENERATE COVARIANCE MATRICES
# Define the functions to generate the covariance matrices
def make_rotation_matrix(theta):
    """Make a matrix that rotates a 2D vector by theta.

    Args:
        theta (float): Angle in degrees.
    """
    theta = torch.deg2rad(theta)
    rot_mat_2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                              [torch.sin(theta), torch.cos(theta)]])
    return rot_mat_2

# VISUALIZE
def plot_data_covariances(ax, covariances, means=None, lims=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])
    n_classes = means.shape[0]

    dim_pairs = [[0, 1], [2, 3]]
    for i in range(len(dim_pairs)):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax[i])
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(n_classes),
                               dim_pair=dim_pairs[i], ax=ax[i])
        dim_pairs_label = [d+1 for d in dim_pairs[i]]
        ax[i].set_aspect('equal')
        # Add names to axes
        ax[i].set_xlabel(r'$x_{}$'.format(dim_pairs_label[0]), fontsize=16)
        ax[i].set_ylabel(r'$x_{}$'.format(dim_pairs_label[1]), fontsize=16)
        if lims is not None:
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)


# VISUALIZE FILTERS ON TOP OF DATA COVARIANCES
def plot_filters(ax, filters, color='r'):
    """Plot the filters as arrows in data space."""
    # Draw the filters of sqfa as arrows on the plot
    awidth = 0.07
    for f in range(2):
        if torch.norm(filters[f, 0]) > 1e-2:
            ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)
        if torch.norm(filters[f, 2]) > 1e-2:
            ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)


# Dimensions 1-2: Differences in means_1
mean1 = torch.tensor([1.0, 0.0])
angles = [0, 40, 80]
means_1 = []

for angle in angles:
    rot_mat = make_rotation_matrix(torch.as_tensor(angle))
    means_1.append(torch.matmul(rot_mat, mean1))
# Change sign of second mean
means_1[1] = -means_1[1]
means_1 = torch.stack(means_1)
covariances_1 = torch.stack([torch.eye(2) * 0.05 for _ in range(len(angles))])

# Dimensions 3-4: Differences only in covariances
outer_prod = torch.einsum('ni,nj->nij', means_1, means_1)
covariances_2 = outer_prod + covariances_1 * 0.9
means_2 = torch.zeros(means_1.shape)

# Append covariances to 4-by-4 set of matrices
covariances = torch.zeros(3, 4, 4)
covariances[:, :2, :2] = covariances_1
covariances[:, 2:, 2:] = covariances_2

means = torch.cat([means_1, means_2], dim=1)


# Train the models


#### smSQFA ####
second_moments = covariances + torch.einsum('ni,nj->nij', means, means)

n_dim = 4
smsqfa = sqfa.model.SecondMomentsSQFA(
  n_dim=n_dim, n_filters=2,
  feature_noise=noise,
)
smsqfa.fit(data_statistics=second_moments,
          show_progress=True)


#### SQFA ####
stats_dict = {
    'means': means,
    'covariances': covariances,
}
sqfa_m = sqfa.model.SQFA(
  n_dim=n_dim, n_filters=2,
  feature_noise=noise,
)
sqfa_m.fit(data_statistics=stats_dict,
           show_progress=True)


# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(5, 3))

# Plot the data covariances
plot_data_covariances(ax, covariances, means, lims=[-2.2, 2.2])

# Plot the filters
smsqfa_f = smsqfa.filters.detach()
sqfa_m_f = sqfa_m.filters.detach()

plot_filters(ax, smsqfa_f, color='r')
plot_filters(ax, sqfa_m_f, color='k')
plt.tight_layout()

plt.savefig('figures/filters.pdf')
plt.close()

