import torch
from .evaluation import qda_accuracy


def validate_regularization(model, x_train, y_train, x_val, y_val, noise_vals, **kwargs):
    """
    Perform cross-validation for the regularization parameter
    """
    n_vals = len(noise_vals)
    accuracies = torch.zeros(n_vals)

    for i, noise in enumerate(noise_vals):
        model.noise_mat = torch.eye(model.filters.shape[0]) * noise
        # Reinitialize model weights
        model.filters = torch.randn_like(model.filters)

        model.fit(x_train, y_train, **kwargs)

        acc = qda_accuracy(x_train, y_train, x_val, y_val, model.filters.detach(), noise=0.001)
        accuracies[i] = acc
    return accuracies
