import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision
import sqfa
import time

import sys
sys.path.append('..')  # Add parent directory to path for sqfa import
from plotting_classes import *

N_FILTERS = 4
NOISE_FISHER = torch.tensor(0.01)
torch.manual_seed(3)

#############################
#
# LOAD AND PROCESS DATA
#
#############################

# Download and load training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Scale data and subtract global mean
def scale_and_center(x_train, x_test):
    std = x_train.std()
    x_train = x_train / (std * n_row)
    x_test = x_test / (std * n_row)
    global_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - global_mean
    x_test = x_test - global_mean
    return x_train, x_test

n_samples, n_row, n_col = trainset.data.shape
n_dim = n_row * n_col
x_train = trainset.data.reshape(-1, n_dim).float()
y_train = trainset.targets
x_test = testset.data.reshape(-1, n_dim).float()
y_test = testset.targets

# Visualize some of the non centered images
names = y_train.unique().tolist()
n_classes = len(y_train.unique())
fig, ax = plt.subplots(2, n_classes // 2, figsize=(8, 4))
for i in range(n_classes):
    row = i // 5
    col = i % 5
    ax[row, col].imshow(x_train[y_train == i][20].reshape(n_row, n_col), cmap='gray')
    ax[row, col].axis('off')
plt.tight_layout()
# Save image
plt.savefig('figures/mnist_images_pre.png', bbox_inches='tight', pad_inches=0)
plt.close()

# Scale data and subtract global mean
x_train, x_test = scale_and_center(x_train, x_test)


# ------------------------------
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS, svd_solver='covariance_eigh')
start = time.time()
pca.fit(x_train)
pca_time = time.time() - start
pca_filters = pca.components_


# ------------------------------
# Train SQFA
# ------------------------------
fisher_rao_model = sqfa.model.SQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=NOISE_FISHER,
)

start = time.time()
fisher_rao_model.fit_pca(x_train) # Initialize filters with PCA
fisher_rao_model.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=True,
  pairwise=True,
)
fisher_rao_time = time.time() - start
fisher_rao_filters = fisher_rao_model.filters.detach()


filters_list = [
  fisher_rao_model.filters.detach().cpu().numpy(),
  pca_filters,
]



plot_class_statistics(
  filter_list, x=x_test, y=y_test, model_names=['SQFA', 'PCA']
)
plt.show()

