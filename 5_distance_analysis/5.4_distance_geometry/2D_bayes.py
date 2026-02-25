import torch
import matplotlib.pyplot as plt


def ll_ratio(x, sigmas2, isnull=True):
    """
    Computes ll ratio for a set of x values and a set of sigmas2 values.
    x: tensor of shape (n_samples, 2)
    sigmas2: tensor of shape (2, n_lambda1, n_lambda2)
    """
    coefs = 1/sigmas2 - 1
    if not isnull:
        coefs = coefs * sigmas2 # Give H1 variances to the x's
    ll = -0.5 * (
      torch.einsum('...nd,dij->...nij', x**2, coefs) +
      torch.sum(torch.log(sigmas2), dim=0)
    )
    return ll


def sample_error_rate(sigma2_1, sigma2_2, n_samples=100000):
    sigma_grid1, sigma_grid2 = torch.meshgrid(
      sigma2_1, sigma2_2, indexing='ij'
    )
    sigmas2 = torch.stack((sigma_grid1, sigma_grid2), dim=0)
    x = torch.randn(n_samples, 2)
    # H0 error rate
    ll0 = ll_ratio(x, sigmas2, isnull=True)
    error0 = ll0 >= 0
    error0 = torch.mean(error0.float(), dim=0)
    # H1 error rate
    ll1 = ll_ratio(x, sigmas2, isnull=False)
    error1 = ll1 < 0
    error1 = torch.mean(error1.float(), dim=0)
    # Total error rate
    error = 0.5 * (error0 + error1)
    return error

#################
# GENERALIZED EIGENVALUES TO USE
#################

lim = 10.0  # Limit for the log scale
n_points = 51
eigval1_log = torch.linspace(-lim, lim, n_points)
eigval1 = 10**eigval1_log
eigval2_log = torch.linspace(-lim, lim, n_points)
eigval2 = 10**eigval2_log


#################
# COMPUTE DISTANCES AS FUNCTION OF GENERALIZED EIGENVALUES
#################

ai = torch.log(eigval1)**2 + torch.log(eigval2)[:,None]**2
ai = torch.sqrt(ai) / torch.sqrt(torch.tensor(2))

# compute the bayes error and accuracy
reps = 2
bayes_error = torch.zeros(len(eigval1_log), len(eigval2_log))
for i in range(reps):
    bayes_error += sample_error_rate(
      eigval1, eigval2, n_samples=100000
    )
bayes_error /= reps
accuracy = (1 - bayes_error)
log_odds = torch.log(accuracy / bayes_error)


#################
# PLOT LEVEL CURVES
#################

slice_ind = [25, 32, 39]  # Values to plot slices at
colors = ['k', 'r', 'b']  # Color of the slices plot
n_levels = 8
eigval1_grid, eigval2_grid = torch.meshgrid(
  eigval1_log, eigval2_log, indexing='ij'
)

# 1) Plot contours for accuracy
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.75))
fontsize = 15
si = 0
levels_error = torch.linspace(torch.min(accuracy), torch.max(accuracy).item(), n_levels)
cs1 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    accuracy.numpy(), levels=levels_error.numpy())
ax[si].clabel(cs1, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Accuracy', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)
plt.tight_layout()

# Plot contours for 'lodds'
si = 1
levels_lodds = torch.linspace(0, torch.max(log_odds).item(), n_levels)
cs2 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    log_odds.numpy(), levels=levels_lodds.numpy())
ax[si].clabel(cs2, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Log-odds correct', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)
plt.tight_layout()

# Plot contours for 'ai'
si = 2
levels = torch.linspace(0, torch.max(ai).item(), n_levels)
cs3 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    ai.numpy(), levels=levels.numpy())
ax[si].clabel(cs3, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Fisher-Rao', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)


#################
# PLOT THE SLICES
#################

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 3, figsize=(15, 3))
for j, ind in enumerate(slice_ind):
    ax[0].plot(eigval2_log, accuracy[ind], color=colors[j])
    ax[1].plot(eigval2_log, log_odds[ind], color=colors[j])
    ax[2].plot(eigval2_log, ai[ind] / torch.sqrt(torch.tensor(2)), color=colors[j])
ax[0].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[1].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[2].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[0].set_yticks([0.5, 0.75, 1.0])
ax[0].set_ylabel('Accuracy', fontsize=fontsize)
ax[1].set_ylabel('Log-odds correct', fontsize=fontsize)
ax[2].set_ylabel('Fisher-Rao distance', fontsize=fontsize)
plt.tight_layout()
#plt.show()
plt.savefig('plots/contours_slice.pdf', dpi=300)
plt.close()



#############
# PLOT EXAMPLE ELLIPSES
#############

# VISUALIZE
import sqfa

lims = [-3.5, 3.5]
def plot_func(cov):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sqfa.plot.statistics_ellipses(ellipses=cov, ax=ax)
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')


cov_template = torch.eye(2).repeat(2, 1, 1) / torch.tensor(4)


# Examples
mult = torch.tensor(10)
cov1 = cov_template.clone()
cov1[0] = cov1[0] * mult
cov2 = cov_template.clone()
cov2[0] = cov2[0] * 1/mult
cov3 = cov_template.clone()
cov3[0,0,0] = cov3[0,0,0] * mult
cov3[0,1,1] = cov3[0,1,1] * 1/mult
cov4 = cov_template.clone()
cov4[0,0,0] = cov4[0,0,0] * 1/mult
cov4[0,1,1] = cov4[0,1,1] * mult

# Plot
plot_func(cov1)
plt.savefig('plots/contours_ellipse1.pdf', dpi=300)
plt.close()
plot_func(cov2)
plt.savefig('plots/contours_ellipse2.pdf', dpi=300)
plt.close()
plot_func(cov3)
plt.savefig('plots/contours_ellipse3.pdf', dpi=300)
plt.close()
plot_func(cov4)
plt.savefig('plots/contours_ellipse4.pdf', dpi=300)
plt.close()

