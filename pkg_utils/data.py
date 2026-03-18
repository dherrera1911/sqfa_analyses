"""Data preprocessing helpers for normalization and train/validation splitting."""

import torch
from sklearn.model_selection import StratifiedShuffleSplit

def scale_and_center(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    scale_factor: float=1.0,
):
    """Normalize data by global scale and mean.

    Parameters
    ----------
    x_train, x_test : torch.Tensor
        Flattened data tensors with shape ``(n_samples, n_features)``.
    scale_factor : float, optional
        Additional factor applied to the global standard deviation before
        scaling. Pass ``n_row`` to retain legacy behaviour of dividing by the
        number of image rows. Defaults to ``1.0`` when ``None``.

    Returns
    -------
    tuple of torch.Tensor
        Normalized training and test data.
    """
    if not isinstance(x_train, torch.Tensor) or not isinstance(x_test, torch.Tensor):
        raise TypeError("scale_and_center expects torch.Tensor inputs")

    std = x_train.std()
    if std == 0:
        raise ValueError("Training data has zero standard deviation")

    denom = std * float(scale_factor)
    x_train = x_train / denom
    x_test = x_test / denom

    mean = x_train.mean(dim=0, keepdim=True)
    x_train = x_train - mean
    x_test = x_test - mean
    return x_train, x_test


def train_val_split(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    val_size: float = 0.2,
):
    """Split training data into training and validation sets.

    Parameters
    ----------
    x_train : torch.Tensor
        Training data tensor with shape ``(n_samples, n_features)``.
    y_train : torch.Tensor
        Training labels tensor with shape ``(n_samples,)``.
    val_size : float, optional
        Proportion of the training data to use for validation. Default is ``0.1``.
    random_state : int, optional
        Random seed for reproducibility. Default is ``None``.

    Returns
    -------
    tuple of torch.Tensor
        Training and validation data and labels.
    """
    if not isinstance(x_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
        raise TypeError("train_val_split expects torch.Tensor inputs")

    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
    train_idx, val_idx = next(sss.split(x_train.numpy(), y_train.numpy()))

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    x_train_new = x_train[train_idx]
    y_train_new = y_train[train_idx]

    return x_train_new, y_train_new, x_val, y_val
