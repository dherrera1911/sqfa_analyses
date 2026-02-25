import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from metric_learn import LMNN

import sqfa
from functions import load_data, normalize_stim

sys.path.append(os.path.abspath('./ama_dir'))
from ama_model import AMAGauss
from optim import fit

sys.path.append('..')  # Add parent directory to path
from other_methods import SupervisedPCA


N_FILTERS = 8
RESPONSE_NOISE = 0.001
C50 = 0.8

N_SUBSAMPLE_LMNN = 10
N_DIM_LMNN = 100
torch.manual_seed(2)


#############################
#
# LOAD AND PROCESS DATA
#
#############################

train = load_data("train")
test = load_data("test")

x_train = normalize_stim(train[0], C50)
x_test = normalize_stim(test[0], C50)
y_train = train[1]
y_test = test[1]


#############################
#
# TRAIN MODELS
#
#############################

########################
# OTHER METHODS
########################

# ------------------------------
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS, svd_solver='covariance_eigh')
start = time.time()
pca.fit(x_train)
pca_time = time.time() - start
pca_filters = torch.tensor(pca.components_).float()
np.save('filters/pca_filters.npy', np.array(pca_filters))
np.save('filters/pca_time.npy', np.array(pca_time))


# ------------------------------
# Train Supervised PCA
# ------------------------------

x_subsampled = x_train
y_subsampled = y_train
spca = SupervisedPCA(n_components=N_FILTERS, label_kernel="delta")
start = time.time()
spca.fit(x_subsampled, y_subsampled)
spca_time = time.time() - start
spca_filters = spca.components_
np.save('filters/spca_filters.npy', np.array(spca_filters))
np.save('filters/spca_time.npy', np.array(spca_time))


# ------------------------------
# Train smSQFA
# ------------------------------
smsqfa_model = sqfa.model.SecondMomentsSQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=RESPONSE_NOISE,
)

start = time.time()
smsqfa_model.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=True,
  pairwise=True,
  estimator="empirical",
)
smsqfa_time = time.time() - start
smsqfa_filters = smsqfa_model.filters.detach()
np.save('filters/smsqfa_filters.npy', np.array(smsqfa_filters))
np.save('filters/smsqfa_time.npy', np.array(smsqfa_time))

# ------------------------------
# Train SQFA
# ------------------------------
fisher_rao = sqfa.model.SQFA(
  n_dim=x_train.shape[1],
  n_filters=N_FILTERS,
  feature_noise=RESPONSE_NOISE,
)

start = time.time()
fisher_rao.fit(
  x_train,
  y_train,
  max_epochs=300,
  show_progress=True,
  pairwise=True,
  estimator="empirical",
)
fisher_rao_time = time.time() - start
fisher_rao_filters = fisher_rao.filters.detach()
np.save('filters/sqfa_filters.npy', np.array(fisher_rao_filters))
np.save('filters/sqfa_time.npy', np.array(fisher_rao_time))

# ------------------------------
# Train Bhattacharyya
# ------------------------------
bhattacharyya = sqfa.model.SQFA(
    n_dim=x_train.shape[1],
    n_filters=N_FILTERS,
    feature_noise=RESPONSE_NOISE,
    distance_fun=sqfa.distances.bhattacharyya,
)

# Fit SQFA with Bhattacharyya distance
start = time.time()
bhattacharyya.fit(
    x_train,
    y_train,
    max_epochs=300,
    show_progress=False,
    pairwise=True,
    estimator="empirical",
)
bhattacharyya_time = time.time() - start
bhattacharyya_filters = bhattacharyya.filters.detach()
np.save('filters/bhattacharyya_filters.npy', np.array(bhattacharyya_filters))
np.save('filters/bhattacharyya_time.npy', np.array(bhattacharyya_time))

# ------------------------------
# Train AMA
# ------------------------------

x_train_ch = x_train.unsqueeze(1) # Add channel dimension

# Create an AMA model with 8 filters
ama = AMAGauss(
    stimuli=x_train_ch,
    labels=y_train,
    n_filters=N_FILTERS,
    response_noise=RESPONSE_NOISE,
)

# Train AMA Gauss
start = time.time()
loss, _ = fit(
  model=ama,
  stimuli=x_train_ch,
  labels=y_train,
  max_epochs=100,
  lr=0.2,
  show_progress=True,
  return_loss=True,
  pairwise=True,
)
ama_time = time.time() - start
ama_filters = ama.filters.squeeze().detach()
np.save('filters/ama_filters.npy', np.array(ama_filters))
np.save('filters/ama_time.npy', np.array(ama_time))

# ------------------------------
# Train LDA 
# ------------------------------
shrinkage=0.7
lda = LinearDiscriminantAnalysis(n_components=N_FILTERS, solver='eigen',
                                 shrinkage=shrinkage)
start = time.time()
lda.fit(x_train, y_train)
lda_time = time.time() - start
lda_filters = lda.coef_[:N_FILTERS]
np.save('filters/lda_filters.npy', np.array(lda_filters))
np.save('filters/lda_time.npy', np.array(lda_time))


# ------------------------------
# Train FA
# ------------------------------
fa = FactorAnalysis(n_components=N_FILTERS)
start = time.time()
fa.fit(x_train)
fa_time = time.time() - start
fa_filters = torch.tensor(fa.components_).float()
np.save('filters/fa_filters.npy', np.array(fa_filters))
np.save('filters/fa_time.npy', np.array(fa_time))


# ------------------------------
# Train ICA
# ------------------------------
ica = FastICA(n_components=N_FILTERS)
start = time.time()
ica.fit(x_train)
ica_time = time.time() - start
ica_filters = torch.tensor(ica.components_).float()
np.save('filters/ica_filters.npy', np.array(ica_filters))
np.save('filters/ica_time.npy', np.array(ica_time))


# ------------------------------
# Train LMNN
# ------------------------------
pca_subsample = PCA(n_components=N_DIM_LMNN)
pca_subsample.fit(x_train)
x_transformed = pca_subsample.transform(x_train)

y_train_sub = y_train
x_transformed, y_train_sub = x_transformed[::N_SUBSAMPLE_LMNN], y_train[::N_SUBSAMPLE_LMNN]

lmnn = LMNN(n_neighbors=3, learn_rate=1e-6, n_components=9, init='pca', verbose=True,
           max_iter=2000, convergence_tol=1.0)
start = time.time()
lmnn.fit(x_transformed, y_train_sub)
lmnn_time = time.time() - start

lmnn_filters = pca_subsample.inverse_transform(lmnn.components_)
np.save('filters/lmnn_filters.npy', np.array(lmnn_filters))
np.save('filters/lmnn_time.npy', np.array(lmnn_time))

#############################
#
# PLOT FILTERS
#
#############################

# Load filters
filter_names = [
    'sqfa_filters.npy',
    'smsqfa_filters.npy',
    'bhattacharyya_filters.npy',
    'ama_filters.npy',
    'lda_filters.npy',
    'spca_filters.npy',
    'pca_filters.npy',
    'ica_filters.npy',
    'fa_filters.npy',
    'lmnn_filters.npy',
]

model_names = [
  "SQFA",
  "smSQFA",
  "Bhatt",
  "AMA",
  "LDA",
  "SPCA",
  "PCA",
  "ICA",
  "FA",
  "LMNN",
]


model_filters = []
model_times = []
for name in filter_names:
    model_filters.append(np.load(f'filters/{name}'))
    model_times.append(np.load(f'filters/{name.replace("filters", "time")}'))


# Function to plot filters
def plot_filters(filters, title):
    fig, ax = plt.subplots(1, N_FILTERS, figsize=(7, 1), constrained_layout=True)
    for i in range(N_FILTERS):
        ax[i].imshow(filters[i].reshape(15, 30), cmap='gray')
        ax[i].axis('off')
    plt.tight_layout()

for name, filters in zip(model_names, model_filters):
    plot_filters(filters, name)
    plt.savefig(
      f'figures/motion_{name.lower()}_filters.png', bbox_inches='tight', pad_inches=0
    )
    plt.close()


#############################
#
# PLOT TRAINING TIMES
#
#############################

fig, ax = plt.subplots(figsize=(7, 3))
plt.bar(range(len(model_times)), model_times)
plt.xticks(range(len(model_times)), model_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Training Time (s)", fontsize=14)
plt.xlabel("Model", fontsize=14)
# Make y axis logarithmic
plt.yscale('log')
# Print the times on top of the bars
for i, training_time in enumerate(model_times):
    plt.text(i, training_time * 1.5, f"{training_time:.1f}", ha='center', fontsize=12)
plt.tight_layout()
plt.ylim([min(model_times)*0.5, max(model_times) * 3.5])
# Save image
plt.savefig('figures/motion_training_times.pdf')


#############################
#
# PLOT QDA ACCURACIES
#
#############################
def get_qda_accuracy(x_train, labels_train, x_test, labels_test, filters):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    # Get the features
    filters = torch.as_tensor(filters, dtype=torch.float)
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(z_train, labels_train)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == labels_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []

for filters  in model_filters:
    with torch.no_grad():
        accuracy = get_qda_accuracy(x_train, y_train, x_test, y_test, filters)
        accuracies.append(accuracy * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(8, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), model_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("QDA Accuracy (%)", fontsize=14)
plt.xlabel("Features", fontsize=14)
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=12)
plt.tight_layout()
ax.set_ylim([0, 100])
# Save image
plt.savefig('figures/motion_accuracies.pdf')



#############################
#
# PLOT KNN ACCURACIES
#
#############################

def get_knn_accuracy(x_train, y_train, x_test, y_test, filters):
    """Fit KNN model to the training data and return the accuracy on the test data."""
    # Get the features
    filters = torch.as_tensor(filters, dtype=torch.float)
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Fit KNN model
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(z_train, y_train)
    y_pred = knn.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []
for name, filters in zip(model_names, model_filters):
    accuracy = get_knn_accuracy(x_train, y_train, x_test, y_test, filters)
    accuracies.append(accuracy.item() * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(7, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), model_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("KNN Accuracy (%)", fontsize=14)
plt.xlabel("Features", fontsize=14)
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=12)
plt.tight_layout()

ax.set_ylim([min(accuracies)*0.9, 100])  # Adjust if needed
# Save image
plt.savefig('figures/motion_accuracies_knn.pdf')


#############################
#
# PLOT QDA ACCURACIES WITH GAUSSIAN DATA
#
#############################

def get_qda_accuracy_gaussian(x_train, y_train, filters):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    filters = np.asarray(filters)
    filters = filters / np.linalg.norm(filters, axis=1, keepdims=True)
    # Get the features
    z_train = torch.matmul(x_train, torch.as_tensor(filters.T).float())
    # Add noise
    #z_train += torch.randn_like(z_train) * torch.sqrt(torch.tensor(RESPONSE_NOISE))
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(z_train, y_train)
    # Simulate Gaussian data for the testing set
    n_samples = 5000
    z_test = []
    y_test = []
    for i in range(qda.means_.shape[0]):
        mean = torch.tensor(qda.means_[i])
        cov = torch.tensor(qda.covariance_[i])
        dist = torch.distributions.MultivariateNormal(mean, cov)
        z_test.append(dist.sample((n_samples,)))
        y_test.append(torch.full((n_samples,), i))
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []
for name, filters in zip(model_names, model_filters):
    accuracy = get_qda_accuracy_gaussian(x_train, y_train, filters)
    accuracies.append(accuracy.item() * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(8, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), model_names, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("QDA Accuracy (%)", fontsize=14)
plt.xlabel("Features", fontsize=14)
# Print the accuracies on top of the bars
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f"{acc:.1f}%", ha='center', fontsize=12)
plt.tight_layout()
ax.set_ylim([0, 100])  # Adjust if needed
# Save image
plt.savefig('figures/motion_accuracies_gaussian.pdf')
plt.close()

