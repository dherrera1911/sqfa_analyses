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
NOISE_FISHER = torch.tensor(0.001)
torch.manual_seed(3)

#############################
#
# LOAD AND PROCESS DATA
#
#############################

# Download and load training and test datasets
trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True)
testset = torchvision.datasets.SVHN(root='./data', split='test', download=True)

# Convert to PyTorch tensors, average channels and reshape
n_samples, n_channels, n_row, n_col = trainset.data.shape
x_train = torch.as_tensor(trainset.data).float()
x_train = x_train.mean(dim=1).reshape(-1, n_row * n_col)
y_train = torch.as_tensor(trainset.labels, dtype=torch.long)
x_test = torch.as_tensor(testset.data).float()
x_test = x_test.mean(dim=1).reshape(-1, n_row * n_col)
y_test = torch.as_tensor(testset.labels, dtype=torch.long)

# Scale data and subtract global mean
def scale_and_center(x_train, x_test):
    std = x_train.std()
    x_train = x_train / (std * n_row)
    x_test = x_test / (std * n_row)
    global_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - global_mean
    x_test = x_test - global_mean
    return x_train, x_test

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
  filters_list, x=x_test, y=y_test, model_names=['SQFA', 'PCA']
)
plt.show()

