"""Plot contours of different distances for 2D Gaussians with 0-mean"""

import torch
import matplotlib.pyplot as plt

#################
# GENERALIZED EIGENVALUES TO USE
#################

lim = 2.0  # Limit for the log scale
n_points = 51
eigval1_log = torch.linspace(-lim, lim, n_points)
eigval1 = 10**eigval1_log
eigval2_log = torch.linspace(-lim, lim, n_points)
eigval2 = 10**eigval2_log


#################
# COMPUTE DISTANCES AS FUNCTION OF GENERALIZED EIGENVALUES
#################

fisher_rao = torch.log(eigval1)**2 + torch.log(eigval2)[:,None]**2
fisher_rao = torch.sqrt(fisher_rao) / torch.sqrt(torch.tensor(2))

bhatt = 0.5 * torch.log(
  ((eigval1+1)/2) * ((eigval2[:,None] + 1)/2) / \
  torch.sqrt(eigval1 * eigval2[:,None])
)

hellinger = torch.sqrt(1 - torch.exp(-bhatt) + 1e-6)


#################
# PLOT LEVEL CURVES
#################

slice_ind = [25, 35, 45]  # Values to plot slices at
colors = ['k', 'r', 'b']  # Color of the slices plot
n_levels = 8
eigval1_grid, eigval2_grid = torch.meshgrid(
  eigval1_log, eigval2_log, indexing='ij'
)

# 1) Plot contours for Fisher-Rao
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.75))
fontsize = 15
si = 0
levels_error = torch.linspace(torch.min(fisher_rao), torch.max(fisher_rao).item(), n_levels)
cs1 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    fisher_rao.numpy(), levels=levels_error.numpy())
ax[si].clabel(cs1, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Fisher-Rao', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)
plt.tight_layout()

# Plot contours for 'Bhattacharyya'
si = 1
levels_lodds = torch.linspace(0, torch.max(bhatt).item(), n_levels)
cs2 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    bhatt.numpy(), levels=levels_lodds.numpy())
ax[si].clabel(cs2, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Bhattacharyya', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)
plt.tight_layout()

# Plot contours for 'helliner'
si = 2
levels = torch.linspace(0, torch.max(hellinger).item(), n_levels*2)
cs3 = ax[si].contour(eigval1_grid.numpy(), eigval2_grid.numpy(),
                    hellinger.numpy(), levels=levels.numpy())
ax[si].clabel(cs3, inline=True, fontsize=8)
for j, ind in enumerate(slice_ind):
    ax[si].plot([-lim, lim], [eigval2_log[ind], eigval2_log[ind]], color=colors[j],
               linestyle='--', linewidth=2)
ax[si].set_title('Hellinger', fontsize=fontsize*1.2)
ax[si].set_xlabel(r'$log_{10}(\sigma_1^2)$', fontsize=fontsize)
ax[si].set_ylabel(r'$log_{10}(\sigma_2^2)$', fontsize=fontsize)

plt.savefig('plots/contours_all.pdf', dpi=300)

#################
# PLOT THE SLICES
#################

plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for j, ind in enumerate(slice_ind):
    ax[0].plot(eigval2_log, fisher_rao[ind], color=colors[j])
    ax[1].plot(eigval2_log, bhatt[ind], color=colors[j])
    ax[2].plot(eigval2_log, hellinger[ind], color=colors[j])
ax[0].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[1].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[2].set_xlabel(r'$\log_{10}(\sigma_1^2)$')
ax[0].set_ylabel('Fisher-Rao', fontsize=fontsize)
ax[1].set_ylabel('Bhattacharyya', fontsize=fontsize)
ax[2].set_ylabel('Hellinger', fontsize=fontsize)
plt.tight_layout()
#plt.show()
plt.savefig('plots/contours_slice.pdf', dpi=300)
plt.close()

