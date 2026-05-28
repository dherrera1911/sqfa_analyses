"""Train filters on SVHN by maximizing different distances."""
import torch
import matplotlib.pyplot as plt
import torchvision
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sqfa
import time

N_FILTERS = 9
torch.manual_seed(1)
NOISE = torch.tensor(0.001)

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
x_train = x_train.to(dtype=torch.float64)
x_test = x_test.to(dtype=torch.float64)

#############################
#
# TRAIN MODELS WITH DIFFERENT DISTANCES
#
#############################

def bw_dist(A, B):
    """Compute the Bures-Wasserstein distance between all pairs
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
    Compute the bhattacharyya distance between each pair of covariances
    """
    mean_cov = (A[:,None] + B[None,:]) * 0.5
    cov_det_A = torch.logdet(A)
    cov_det_B = torch.logdet(B)
    dist = torch.logdet(mean_cov) - (cov_det_A[:,None] + cov_det_B[None,:]) * 0.5
    return torch.squeeze(dist)


# ------------------------------
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS, svd_solver='covariance_eigh')
pca.fit(x_train)
x_pca = torch.as_tensor(pca.transform(x_train))
pca_var = torch.var(x_pca, dim=0)

# ------------------------------
# Train with different distances
# ------------------------------

distance_list = [
  sqfa.distances.affine_invariant,
  bhattacharyya,
  sqfa.distances.log_euclidean,
  bw_dist,
  bw_reg_dist,
  euclidean_dist,
  kl_div,
]

distance_names = [
  'FR',
  'Bhatt',
  'LE',
  'BW',
  'BWR',
  'E',
  'KL'
]

filter_list = []
time_list = []

for distance in distance_list:

    sqfa_model = sqfa.model.SecondMomentsSQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=NOISE,
        distance_fun=distance,
    )

    # Fit SQFA
    start = time.time()
    sqfa_model.fit_pca(x_train)
    sqfa_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    time_list.append(time.time() - start)
    filter_list.append(sqfa_model.filters.detach())

#############################
#
# PLOT FILTERS
#
#############################

# Function to plot filters
def plot_filters(filters, title):
    fig, ax = plt.subplots(1, N_FILTERS, figsize=(10, 3))
    for i in range(N_FILTERS):
        ax[i].imshow(filters[i].reshape(n_row, n_col), cmap='gray')
        ax[i].axis('off')
        ax[i].set_title(f"Filter {i+1}")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

for name, filters in zip(distance_names, filter_list):
    plot_filters(filters, name)
    plt.savefig(f'figures/svhn_{name.lower()}_filters.png')
    plt.close()


#############################
#
# PLOT QDA ACCURACIES
#
#############################

def get_qda_accuracy(x_train, y_train, x_test, y_test, filters):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    # Get the features
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(z_train, y_train)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []

for name, filters in zip(distance_names, filter_list):
    accuracy = get_qda_accuracy(x_train, y_train, x_test, y_test, filters)
    accuracies.append(accuracy.item() * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(6, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), distance_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("QDA Accuracy (%)", fontsize=14)
plt.xlabel("Features", fontsize=14)
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=12)
plt.tight_layout()
ax.set_ylim([0, 80])  # Adjust if needed
# Save image
plt.savefig('figures/svhn_accuracies.pdf')
plt.close()


#############################
#
# PLOT QDA ACCURACIES WITH GAUSSIAN DATA
#
#############################

def get_qda_accuracy_gaussian(x_train, y_train, filters):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    # Get the features
    n_samples = 20000
    z_train = torch.matmul(x_train, torch.as_tensor(filters.T).float())
    z_train += torch.randn_like(z_train) * torch.sqrt(torch.tensor(NOISE))
    unique_labels = torch.unique(y_train)
    z_test = []
    y_test = []
    for i in unique_labels:
        z_class = z_train[y_train == i]
        z_mean = torch.mean(z_class, dim=0, keepdim=True)
        z_cov = torch.cov(z_class.T)
        z_sm = z_mean.T @ z_mean + z_cov
        dist = torch.distributions.MultivariateNormal(
          torch.zeros(N_FILTERS), z_sm
        )
        z_test.append(dist.sample((n_samples,)))
        y_test.append(torch.full((n_samples,), i))
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(z_test, y_test)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []

for name, filters in zip(distance_names, filter_list):
    accuracy = get_qda_accuracy_gaussian(x_train.float(), y_train, filters.float())
    accuracies.append(accuracy.item() * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(7, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), distance_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("QDA Accuracy (%)", fontsize=14)
plt.xlabel("Features", fontsize=14)
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=12)
plt.tight_layout()
ax.set_ylim([0, 80])  # Adjust if needed
# Save image
plt.savefig('figures/svhn_accuracies_gaussian.pdf')
plt.close()

