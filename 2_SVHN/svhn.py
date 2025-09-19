import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from metric_learn import LMNN
import sqfa
import time

import sys
sys.path.append('..')  # Add parent directory to path

from pkg_utils import (
  scale_and_center,
  train_val_split,
  qda_accuracy,
  validate_regularization,
  SupervisedPCA
)

N_FILTERS = 9
NOISE_VALS = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
n_subsample = 10
n_dim_lmnn = 100
N_REPS = 20
torch.manual_seed(2)

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

x_train, x_test = scale_and_center(x_train, x_test)


#############################
#
# TRAIN MODELS
#
#############################
x_train_reg, y_train_reg, x_val, y_val = train_val_split(x_train, y_train, val_size=0.15)

# ------------------------------
# Train PCA
# ------------------------------
pca = PCA(n_components=N_FILTERS)
start = time.time()
pca.fit(x_train)
pca_time = time.time() - start
pca_filters = pca.components_
np.save('filters/pca_filters.npy', np.array(pca_filters))
np.save('filters/pca_time.npy', np.array(pca_time))


# ------------------------------
# Train Supervised PCA
# ------------------------------
x_subsampled = x_train[::5]
y_subsampled = y_train[::5]
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
# Get noise hyperparameter via validation
smsqfa_val = sqfa.model.SecondMomentsSQFA(
    n_dim=x_train.shape[1],
    n_filters=N_FILTERS,
    feature_noise=0,
)

noise_accs = validate_regularization(smsqfa_val, x_train, y_train, x_val, y_val, NOISE_VALS)
smsqfa_noise = NOISE_VALS[torch.argmax(noise_accs)]

smsqfa_filter_list = []
smsqfa_times = []
for _rep in range(N_REPS):
    smsqfa_model = sqfa.model.SecondMomentsSQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=smsqfa_noise,
    )
    start = time.time()
    smsqfa_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    smsqfa_times.append(time.time() - start)
    smsqfa_filter_list.append(smsqfa_model.filters.detach().numpy())

np.save('filters/smsqfa_filters.npy', np.array(smsqfa_filter_list))
np.save('filters/smsqfa_time.npy', np.array(smsqfa_times))
np.save('filters/smsqfa_noise.npy', np.array(smsqfa_noise))


# ------------------------------
# Train SQFA
# ------------------------------
sqfa_val = sqfa.model.SQFA(
    n_dim=x_train.shape[1],
    n_filters=N_FILTERS,
    feature_noise=0,
)

sqfa_noise_accs = validate_regularization(sqfa_val, x_train, y_train, x_val, y_val, NOISE_VALS,
)
sqfa_noise = NOISE_VALS[torch.argmax(sqfa_noise_accs)]

sqfa_filter_list = []
sqfa_times = []
for _rep in range(N_REPS):
    sqfa_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=sqfa_noise,
    )
    start = time.time()
    sqfa_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    sqfa_times.append(time.time() - start)
    sqfa_filter_list.append(sqfa_model.filters.detach().numpy())

sqfa_filters = np.mean(np.stack(sqfa_filter_list, axis=0), axis=0)
sqfa_time = float(np.mean(sqfa_times))

np.save('filters/sqfa_filters.npy', sqfa_filters)
np.save('filters/sqfa_time.npy', np.array(sqfa_time))


# ------------------------------
# Train Bhattacharyya
# ------------------------------
bhattacharyya_val = sqfa.model.SQFA(
    n_dim=x_train.shape[1],
    n_filters=N_FILTERS,
    feature_noise=0,
    distance_fun=sqfa.distances.bhattacharyya,
)

bhattacharyya_noise_accs = validate_regularization(
    bhattacharyya_val,
    x_train,
    y_train,
    x_val,
    y_val,
    NOISE_VALS,
    max_epochs=300,
    show_progress=False,
)
bhattacharyya_noise = NOISE_VALS[torch.argmax(bhattacharyya_noise_accs)]

bhattacharyya_filter_list = []
bhattacharyya_times = []
for _rep in range(N_REPS):
    bhattacharyya_model = sqfa.model.SQFA(
        n_dim=x_train.shape[1],
        n_filters=N_FILTERS,
        feature_noise=bhattacharyya_noise,
        distance_fun=sqfa.distances.bhattacharyya,
    )
    start = time.time()
    bhattacharyya_model.fit(
        x_train,
        y_train,
        max_epochs=300,
        show_progress=False,
    )
    bhattacharyya_times.append(time.time() - start)
    bhattacharyya_filter_list.append(bhattacharyya_model.filters.detach().numpy())

bhattacharyya_filters = np.mean(np.stack(bhattacharyya_filter_list, axis=0), axis=0)
bhattacharyya_time = float(np.mean(bhattacharyya_times))

np.save('filters/bhattacharyya_filters.npy', bhattacharyya_filters)
np.save('filters/bhattacharyya_time.npy', np.array(bhattacharyya_time))


# ------------------------------
# Train LDA
# ------------------------------
shrinkage = 0.8  # Set to optimize LDA performance and have smoother filters
lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
start = time.time()
lda.fit(x_train, y_train)
lda_time = time.time() - start
lda_filters = lda.coef_[:N_FILTERS]
np.save('filters/lda_filters.npy', np.array(lda_filters))
np.save('filters/lda_time.npy', np.array(lda_time))


# ------------------------------
# Train ICA
# ------------------------------
ica = FastICA(n_components=N_FILTERS, random_state=0, max_iter=1000)
start = time.time()
ica.fit(x_train)
ica_time = time.time() - start
ica_filters = ica.components_
np.save('filters/ica_filters.npy', np.array(ica_filters))
np.save('filters/ica_time.npy', np.array(ica_time))


# ------------------------------
# Train Factor Analysis
# ------------------------------
fa = FactorAnalysis(n_components=N_FILTERS, random_state=0, max_iter=1000)
start = time.time()
fa.fit(x_train)
fa_time = time.time() - start
fa_filters = fa.components_
np.save('filters/fa_filters.npy', np.array(fa_filters))
np.save('filters/fa_time.npy', np.array(fa_time))


# ------------------------------
# Train LMNN
# ------------------------------
pca_subsample = PCA(n_components=n_dim_lmnn)
pca_subsample.fit(x_train)
x_transformed = pca_subsample.transform(x_train)

y_train_sub = y_train
x_transformed, y_train_sub = x_transformed[::n_subsample], y_train[::n_subsample]

lmnn = LMNN(n_neighbors=3, learn_rate=1e-6, n_components=9, init='pca',
            verbose=True, max_iter=2000, convergence_tol=1.0)
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
    model_times.append(np.load(
      f'filters/{name.replace("filters", "time")}')
    )

# Function to plot filters
def plot_filters(filters, title):
    fig, ax = plt.subplots(1, N_FILTERS, figsize=(7, 1), constrained_layout=True)
    for i in range(N_FILTERS):
        ax[i].imshow(filters[i].reshape(n_row, n_col), cmap='gray')
        ax[i].axis('off')
    plt.tight_layout()

for name, filters in zip(model_names, model_filters):
    plot_filters(filters, name)
    plt.savefig(
      f'figures/svhn_{name.lower()}_filters.png', bbox_inches='tight', pad_inches=0
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
    plt.text(i, training_time * 1.5, f"{training_time:.2f}", ha='center', fontsize=12)
plt.tight_layout()
plt.ylim([min(model_times)*0.5, max(model_times) * 5])
# Save image
plt.savefig('figures/svhn_training_times.pdf')


#############################
#
# PLOT QDA ACCURACIES
#
#############################

def get_qda_accuracy(x_train, y_train, x_test, y_test, filters):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    # Get the features
    filters = torch.as_tensor(filters, dtype=torch.float)
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(z_train, y_train)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy

accuracies = []

for name, filters in zip(model_names, model_filters):
    accuracy = get_qda_accuracy(x_train, y_train, x_test, y_test, filters)
    accuracies.append(accuracy.item() * 100)

# Plot accuracies
fig, ax = plt.subplots(figsize=(7, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), model_names, fontsize=12)
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
    knn = KNeighborsClassifier(n_neighbors=3)
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

ax.set_ylim([0, 80])  # Adjust if needed
# Save image
plt.savefig('figures/svhn_accuracies_knn.pdf')




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
    #z_train += torch.randn_like(z_train) * torch.sqrt(NOISE_FISHER)
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(z_train, y_train)
    # Simulate Gaussian data for the testing set
    n_samples = 20000
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
fig, ax = plt.subplots(figsize=(7, 3))
plt.bar(range(len(accuracies)), accuracies)
plt.xticks(range(len(accuracies)), model_names, fontsize=12)
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
