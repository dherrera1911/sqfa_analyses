"""
Make toy examples of filters learnt by using different distances,
and plot as ellipses and arrows.
"""
import torch
import sqfa
import matplotlib.pyplot as plt

torch.manual_seed(6)  # Set seed for reproducibility
n_dim_pairs = 2
n_dim = n_dim_pairs * 2
noise = 0.001
figsize = (5, 3)
lims = (-2.2, 2.2)


# GENERATE COVARIANCE MATRICES
# Define the functions to generate the covariance matrices
def make_rotation_matrix(theta, dims):
    """Make a matrix that rotates 2 dimensions of a 6x6 matrix by theta.
    Args:
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    theta = torch.deg2rad(theta)
    rotation = torch.eye(n_dim_pairs*2)
    rot_mat_2 = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                              [torch.sin(theta), torch.cos(theta)]])
    for row in range(2):
        for col in range(2):
            rotation[dims[row], dims[col]] = rot_mat_2[row, col]
    return rotation


def make_rotated_classes(base_cov, angles, dims):
    """Rotate 2 dimensions of base_cov, specified in dims, by the angles in the angles list
    Args:
        base_cov (torch.Tensor): Base covariances
        theta (float): Angle in degrees.
        dims (list): List of 2 dimensions to rotate.
    """
    if len(angles) != base_cov.shape[0]:
        raise ValueError('The number of angles must be equal to the number of classes.')

    for i, theta in enumerate(angles):
        rotation_matrix = make_rotation_matrix(theta, dims)
        base_cov[i] = torch.einsum('ij,jk,kl->il', rotation_matrix, base_cov[i], rotation_matrix.T)
    return base_cov


# VISUALIZE
def plot_data_covariances(ax, covariances, means=None, lims=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])
    n_classes = means.shape[0]
    dim_pairs = [[0, 1], [2, 3]]
    legend_type = ['none', 'discrete']
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
def plot_filters(ax, filters, class_covariances, means=None, color='r'):
    """Plot the filters as arrows in data space."""
    # Plot the statistics of the filters
    plot_data_covariances(ax, class_covariances, means, lims=lims)
    # Draw the filters of sqfa as arrows on the plot
    awidth = 0.08
    for f in range(2):
        if torch.norm(filters[f, 0]) > 1e-2:
            ax[0].arrow(0, 0, filters[f, 0], filters[f, 1], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)
        if torch.norm(filters[f, 2]) > 1e-2:
            ax[1].arrow(0, 0, filters[f, 2], filters[f, 3], width=awidth,
                        head_width=awidth*5, label=f'Filter {f}', color=color)


########################
# MAKE TOY DATA
########################

angles = [
  [0, 40, 80],  # Dimensions 1, 2
  [0, 20, 40],  # Dimensions 3, 4
]

n_angles = len(angles[0])
variances = torch.tensor([0.25, 0.01, 1.0, 0.04])
base_cov = torch.diag(variances)
base_cov = base_cov.repeat(n_angles, 1, 1)

class_covariances = base_cov
for d in range(len(angles)):
    ang = torch.tensor(angles[d])
    class_covariances = make_rotated_classes(
      class_covariances, ang, dims=[2*d, 2*d+1]
    )


########################
# TRAIN SQFA WITH DIFFERENT DISTANCES
########################

def bw_dist(A, B):
    """Compute the Bures-Wasserstein squared distance between all pairs
    of matrices in A and B."""
    tr_A = torch.einsum('ijj->i', A)
    tr_B = torch.einsum('ijj->i', B)
    A_sqrt = sqfa.linalg.spd_sqrt(A)
    C = sqfa.linalg.conjugate_matrix(B, A_sqrt)
    C_sqrt_eigvals = torch.sqrt(torch.linalg.eigvalsh(C))
    tr_C = torch.sum(C_sqrt_eigvals, dim=-1)
    bw_distance_sq = tr_A[None,:] + tr_B[:,None] - 2 * tr_C
    return torch.sqrt(torch.abs(bw_distance_sq) + 1e-6)


def bw_reg_dist(A, B):
    """Compute the Bures-Wasserstein squared distance between all pairs
    of matrices in A and B."""
    tr_A = torch.einsum('ijj->i', A)
    tr_B = torch.einsum('ijj->i', B)
    A_sqrt = sqfa.linalg.spd_sqrt(A)
    C = sqfa.linalg.conjugate_matrix(B, A_sqrt)
    C_sqrt_eigvals = torch.sqrt(torch.linalg.eigvalsh(C))
    tr_C = torch.sum(C_sqrt_eigvals, dim=-1)
    bw_distance_sq = tr_A[None,:] + tr_B[:,None] - 2 * tr_C
    bw_distance_sq_norm = bw_distance_sq / (tr_A[None,:] + tr_B[:,None])
    return torch.sqrt(torch.abs(bw_distance_sq_norm) + 1e-6)


def euclidean_dist(A, B):
    """Compute the Euclidean distance between all pairs of matrices in A and B."""
    diff_sq = (A.unsqueeze(0) - B.unsqueeze(1))**2
    euclidean_distance_sq = torch.sum(diff_sq, dim=(-1,-2))
    return torch.sqrt(euclidean_distance_sq + 1e-6)


def kl_div(A, B):
    """
    Compute the symmetrized KL divergence between each pair of covariance
    """
    # Make the matrix with trace term
    term_trace1 = 0.5 * torch.einsum('nmii->nm',
      torch.einsum('nij,mjk->nmik', torch.linalg.inv(B), A)
    )
    term_trace2 = 0.5 * torch.einsum('nmii->nm',
      torch.einsum('nij,mjk->nmik', torch.linalg.inv(A), B)
    )
    # Compute the KL divergence
    kl_div_sym = term_trace1 + term_trace2 - 2 * A.shape[-1]
    return kl_div_sym


def bhattacharyya(A, B):
    """
    Compute the Bhattacharyya distance between each pair of covariance
    """
    mean_cov = (A[:,None] + B[None,:]) * 0.5
    cov_det_A = torch.logdet(A)
    cov_det_B = torch.logdet(B)
    dist = torch.logdet(mean_cov) - (cov_det_A[:,None] + cov_det_B[None,:]) * 0.5
    return torch.squeeze(dist)


distance_list = [
  sqfa.distances.affine_invariant,
  sqfa.distances.log_euclidean,
  bw_dist,
  bw_reg_dist,
  euclidean_dist,
  kl_div,
  bhattacharyya,
]

distance_names = [
  'FR',
  'LE',
  'BW',
  'BWReg',
  'E',
  'KL',
  'Bhatt',
]


# Train SQFA with different distances and save filters
for i, distance_fn in enumerate(distance_list):

    model = sqfa.model.SecondMomentsSQFA(
      n_dim=n_dim, n_filters=2,
      feature_noise=noise,
      distance_fun=distance_fn,
    )

    model.fit(
      data_statistics=class_covariances,
      show_progress=False,
      atol=1e-7,
    )

    filters = model.filters.detach()

    # Plot learned filters
    fig, ax = plt.subplots(1, n_dim_pairs, figsize=figsize, sharex=True, sharey=True) 
    plot_filters(ax, filters, class_covariances)
    plt.suptitle(distance_names[i], fontsize=16, x=0.42)
    plt.tight_layout()
    # Save
    plt.savefig(f'filters_{distance_names[i]}.png')
    plt.close()

