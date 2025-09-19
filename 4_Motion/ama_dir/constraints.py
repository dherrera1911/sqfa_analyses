"""
Constraints to keep the filters on a certain set (e.g. sphere), or to fix
some filter values during training.
"""

import torch
import torch.nn as nn

__all__ = ["Sphere", "FixedFilters"]


def __dir__():
    return __all__


def unit_norm(stimuli):
    """
    Normalize stimuli to have a norm less than or equal to 1.

    Channels are normalized by their aggregated offsetted norm
    (i.e., || sum of squares + c50 ||).

    Parameters
    ----------
    stimuli : torch.Tensor
        Stimuli tensor of shape (n_stim, n_channels, n_dim).
    c50 : torch.Tensor, optional
        Offset constant added to the sum of squares, by default `torch.as_tensor(0)`.

    Returns
    -------
    torch.Tensor
        Normalized stimuli tensor of shape (n_stim, n_channels, n_dim).
    """
    # Normalizing factor
    normalizing_factor = torch.sqrt(torch.sum(stimuli**2, dim=(-2, -1)))
    return stimuli / normalizing_factor[:, None, None]


# Define the sphere constraint
class Sphere(nn.Module):
    """
    Constrains the input tensor to lie on the sphere.
    """

    def forward(self, X):
        """
        Normalize the input tensor so that it lies on the sphere.

        The norm pooled across channels is computed and used to normalize the tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor in Euclidean space with shape (n_filters, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor lying on the sphere with shape
            (n_filters, n_channels, n_dim).
        """
        return unit_norm(X)

    def right_inverse(self, S):
        """
        Identity function to assign to parametrization.

        Parameters
        ----------
        S : torch.Tensor
            Input tensor. Should be different from zero.

        Returns
        -------
        torch.Tensor
            Returns the input tensor `S`.
        """
        return S


class FixedFilters(nn.Module):
    """Fix some of the filters to prevent updating with gradient descent."""

    def __init__(self, n_row_fixed):
        """
        Initialize the FixedFilters class.

        Parameters
        ----------
        value : torch.Tensor
            Value to fix the filters to.
        """
        super().__init__()
        self.n_row_fixed = n_row_fixed

    def forward(self, X):
        """
        Concatenate the fixed tensor with the input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Fixed value.
        """
        fixed_tensor = X[: self.n_row_fixed].detach()
        return torch.cat([fixed_tensor, X[self.n_row_fixed :]], dim=0)

    def right_inverse(self, X):
        """
        Return only the rows after the fixed tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Returns the non-fixed part of the tensor.
        """
        return X

