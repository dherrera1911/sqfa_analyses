"""Evaluation utilities for scoring learned filters with downstream classifiers."""

import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def qda_accuracy(
    x_train,
    y_train,
    x_test,
    y_test,
    filters,
    eval_qda_noise=0.0,
    eval_qda_reg=1.0e-5,
):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    # Get the features
    filters = torch.as_tensor(filters, dtype=x_train.dtype)
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Add noise
    if eval_qda_noise > 0:
        z_train += torch.randn_like(z_train) * torch.sqrt(torch.as_tensor(eval_qda_noise))
        z_test += torch.randn_like(z_test) * torch.sqrt(torch.as_tensor(eval_qda_noise))
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis(
        solver='eigen',
        shrinkage=eval_qda_reg,
        tol=1.0e-7,
    )
    qda.fit(z_train, y_train)
    y_pred = qda.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy


def qda_accuracy_gaussian(x_train, y_train, filters, eval_qda_reg=1.0e-4):
    """Fit QDA model to the training data and return the accuracy on the test data."""
    filters = np.asarray(filters)
    filters = filters / np.linalg.norm(filters, axis=1, keepdims=True)
    # Get the features
    z_train = torch.matmul(x_train, torch.as_tensor(filters.T).float())
    # Fit QDA model
    qda = QuadraticDiscriminantAnalysis(
        store_covariance=True,
        reg_param=eval_qda_reg,
    )
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


def knn_accuracy(x_train, y_train, x_test, y_test, filters, n_neighbors=3):
    """Fit KNN model to the training data and return the accuracy on the test data."""
    # Get the features
    filters = torch.as_tensor(filters, dtype=x_train.dtype)
    z_train = torch.matmul(x_train, filters.T)
    z_test = torch.matmul(x_test, filters.T)
    # Fit KNN model
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(z_train, y_train)
    y_pred = knn.predict(z_test)
    accuracy = torch.mean(torch.as_tensor(y_pred == y_test.numpy(), dtype=torch.float))
    return accuracy


def iterate_filter_sets(filter_bank):
    """Yield individual filter sets regardless of being single or batched."""
    if isinstance(filter_bank, np.ndarray):
        if filter_bank.ndim == 3:
            for filters in filter_bank:
                yield filters
        elif filter_bank.ndim == 2:
            yield filter_bank
        else:
            raise ValueError(
                f"Unexpected filter dimensionality: {filter_bank.ndim}, expected 2 or 3"
            )
    else:
        array = np.asarray(filter_bank)
        yield from iterate_filter_sets(array)


def collect_metric_across_runs(filter_bank, metric_fn):
    """Compute a metric for every set of filters associated with a model."""
    return np.array([metric_fn(filters) for filters in iterate_filter_sets(filter_bank)])
