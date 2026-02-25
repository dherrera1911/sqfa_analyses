import torch
import matplotlib.pyplot as plt
import sqfa

# Make means for 4 classes in a 2D problem
x_dist = 0.8
class_means = torch.tensor([
  [-x_dist, 0.2],
  [-x_dist*1.05, -0.2],
  [x_dist*1.05, 0.6],
  [x_dist, -0.6],
])

class_covariances = torch.diag(torch.tensor([0.008, 0.008]))
class_covariances = torch.stack([class_covariances] * 4)

def plot_data_covariances(ax, covariances, means=None):
    """Plot the covariances as ellipses."""
    if means is None:
        means = torch.zeros(covariances.shape[0], covariances.shape[1])

    dim_pairs = [[0, 1]]
    for i in range(len(dim_pairs)):
        # Plot ellipses 
        sqfa.plot.statistics_ellipses(ellipses=covariances, centers=means,
                                      dim_pair=dim_pairs[i], ax=ax)
        # Plot points for the means
        sqfa.plot.scatter_data(data=means, labels=torch.arange(4),
                               dim_pair=dim_pairs[i], ax=ax)


# Function to plot filters on top of the data covariances
def plot_filters(ax, filters, color, label):
    """Plot the filters as arrows in data space."""
    awidth = 0.05
    ax.arrow(
        0, 0,
        filters[0], filters[1],
        width=awidth,
        head_width=awidth*5,
        label=label,
        color=color
    )

# Get scatter matrices for LDA
scatter_within = torch.mean(class_covariances, dim=0)
scatter_between = class_means.T @ class_means

eigvec, eigval = sqfa.linalg.generalized_eigenvectors(
  scatter_between,
  scatter_within
)
eigvec = eigvec*0.9

figsize = (3, 3)
fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
plot_data_covariances(ax, class_covariances, class_means)
plot_filters(ax, eigvec[:, 0]*-1, 'r', label="LDA1")
plot_filters(ax, eigvec[:, 1]*-1, 'b', label="LDA2")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.tight_layout()
plt.legend()
# Save plot
fig.savefig('lda.png')
plt.close()

# Plot the gaussian densities along the first dimensions of the data

def plot_gaussians(ax, means, variances):
    """For the 1D class means in means and the 1D class variances in variances,
    plot the 1D gaussian densities for each class."""
    x = torch.linspace(-4, 4, 1000)
    for i in range(len(means)):
        y = torch.exp(-0.5 * ((x - means[i]) ** 2) / variances[i])
        ax.plot(x, y)

figsize = (6, 2)
dim = 0
fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
plot_gaussians(ax, class_means[:, dim], class_covariances[:, dim, dim])
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1.2)
ax.set_xlabel('Projection onto LDA1')
ax.set_ylabel('Probability density')
plt.tight_layout()
# Save plot
fig.savefig('lda1.png')
plt.close()

dim = 1
fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
plot_gaussians(ax, class_means[:, dim], class_covariances[:, dim, dim])
ax.set_xlim(-2, 2)
ax.set_ylim(0, 1.2)
ax.set_xlabel('Projection onto LDA2')
ax.set_ylabel('Probability density')
plt.tight_layout()
# Save plot
fig.savefig('lda2.png')
plt.close()

