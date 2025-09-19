from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as tfun
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.utils.parametrizations import orthogonal

from _buffers_dict import BuffersDict
from constraints import Sphere
import inference


class AMAParent(ABC, nn.Module):
    """
    Abstract AMA parent class.
    """

    def __init__(self, priors, n_dim=None, n_filters=None, n_channels=1, filters=None,
                constraint="sphere"):
        """
        Initialize the AMA model.

        Parameters
        ----------
        priors : torch.Tensor
            Prior probabilities for each class, of shape (n_classes).
        n_dim : int, optional
            Number of dimensions of inputs.
        n_filters : int, optional
            Number of filters to use.
        n_channels : int, optional
            Number of channels of the stimuli, by default 1.
        filters : torch.Tensor, optional
            Initial filters to use, by default None.
        """
        super().__init__()
        self.register_buffer("priors", torch.as_tensor(priors))

        if filters is not None:
            filters = torch.as_tensor(filters)
        else:
            assert n_dim is not None, "n_dim must be provided if filters is not provided"
            assert n_filters is not None, "n_filters must be provided if filters is not provided"
            filters = torch.randn(n_filters, n_channels, n_dim)

        # Model parameters
        self.filters = nn.Parameter(filters)
        self.constraint = constraint
        self._add_constraint(constraint)

    #########################
    # PREPROCESSING
    #########################

    @abstractmethod
    def preprocess(self, stimuli):
        """
        Preprocess the stimuli before computing the responses.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Preprocessed stimuli of shape (n_stim, n_channels, n_dim).
        """
        pass

    #########################
    # INFERENCE
    #########################

    @abstractmethod
    def get_responses(self, stimuli):
        """
        Compute the response to each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Responses tensor of shape (n_stim, n_filters).
        """
        pass

    def get_log_likelihoods(self, stimuli):
        """
        Compute the log-likelihood of each class for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).
        """
        responses = self.get_responses(stimuli=stimuli)
        log_likelihoods = self.responses_2_log_likelihoods(responses)
        return log_likelihoods

    def get_posteriors(self, stimuli):
        """
        Compute the posterior of each class for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        log_likelihoods = self.get_log_likelihoods(stimuli=stimuli)
        posteriors = self.log_likelihoods_2_posteriors(log_likelihoods)
        return posteriors

    def get_estimates(self, stimuli):
        """
        Compute latent variable estimates for each stimulus.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Estimates tensor of shape (n_stim).
        """
        posteriors = self.get_posteriors(stimuli=stimuli)
        estimates = self.posteriors_2_estimates(posteriors=posteriors)
        return estimates

    @abstractmethod
    def responses_2_log_likelihoods(self, responses):
        """
        Compute log-likelihood of each class given the filter responses.

        Parameters
        ----------
        responses : torch.Tensor
            Filter responses tensor of shape (n_stim, n_filters).

        Returns
        -------
        torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).
        """
        pass

    def log_likelihoods_2_posteriors(self, log_likelihoods):
        """
        Compute the posterior of each class given the log-likelihoods.

        Parameters
        ----------
        log_likelihoods : torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        posteriors = tfun.softmax(log_likelihoods + torch.log(self.priors), dim=-1)
        return posteriors

    def posteriors_2_estimates(self, posteriors):
        """
        Convert posterior probabilities to estimates of the latent variable.

        Parameters
        ----------
        posteriors : torch.Tensor
            Posterior probabilities tensor of shape (n_stim, n_classes).

        Returns
        -------
        torch.Tensor
            Estimates tensor of shape (n_stim), containing the estimated latent
            variable for each stimulus.
        """
        # Get the index of the class with the highest posterior probability
        estimates = torch.argmax(posteriors, dim=-1)
        return estimates

    def forward(self, stimuli):
        """
        Compute the class posteriors for the stimuli.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Posteriors tensor of shape (n_stim, n_classes).
        """
        posteriors = self.get_posteriors(stimuli)
        return posteriors

    def _add_constraint(self, constraint="none"):
        """
        Add constraint to the filters.

        Parameters
        ----------
        constraint : str
            Constraint to apply to the filters. Can be 'none', 'sphere' or
            'orthogonal'. Default is 'none'.
        """
        if constraint == "sphere":
            register_parametrization(self, "filters", Sphere())
        elif constraint == "orthogonal":
            orthogonal(self, "filters")


class AMAGauss(AMAParent):
    """
    AMAGauss model.

    This model assumes that class-conditional responses are Gaussian distributed.
    """

    def __init__(
        self,
        stimuli,
        labels,
        n_filters=None,
        filters=None,
        priors=None,
        response_noise=0.0,
    ):
        """
        Initialize the AMAGauss model.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (n_stim, n_channels, n_dim).
        labels : torch.int64
            Label tensor of shape (n_stim).
        n_filters : int, optional
            Number of filters to use, by default 2.
        priors : torch.Tensor, optional
            Prior probabilities of each class, by default None.
        response_noise : float, optional
            Noise level in the responses, by default 0.0.
        """
        n_channels = stimuli.shape[-2]
        n_dim = stimuli.shape[-1]
        n_classes = torch.unique(labels).size()[0]

        if filters is not None:
            assert filters.shape[-2] == n_channels, "Channels of filters don't match stimuli."
            assert filters.shape[-1] == n_dim, "Dimensions of filters don't match stimuli."

        if priors is None:
            priors = torch.ones(n_classes) / n_classes

        super().__init__(
            priors=priors,
            n_dim=stimuli.shape[-1],
            n_filters=n_filters,
            n_channels=n_channels,
            filters=filters,
        )
        self.register_buffer("response_noise", torch.as_tensor(response_noise))

        # Store stimuli statistics
        stimulus_statistics = inference.class_statistics(
            points=torch.flatten(self.preprocess(stimuli), -2, -1),  # Collapse channels
            labels=labels,
        )
        self.stimulus_statistics = BuffersDict(stimulus_statistics)

    def preprocess(self, stimuli):
        """
        Does nothing, stands in for a preprocessing step.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (..., n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Processed stimuli tensor of shape (..., n_channels, n_dim).
        """
        return stimuli

    def get_responses(self, stimuli):
        """
        Compute the responses of the filters to the stimuli after
        pre-processing.

        Parameters
        ----------
        stimuli : torch.Tensor
            Stimulus tensor of shape (..., n_channels, n_dim).

        Returns
        -------
        torch.Tensor
            Responses tensor of shape (..., n_filters).
        """
        stimuli_processed = self.preprocess(stimuli)
        responses = torch.einsum("kcd,...cd->...k", self.filters, stimuli_processed)
        return responses

    def responses_2_log_likelihoods(self, responses):
        """
        Compute log-likelihood of each class given the filter responses.

        Parameters
        ----------
        responses : torch.Tensor
            Filter responses tensor of shape (n_stim, n_filters).

        Returns
        -------
        torch.Tensor
            Log-likelihoods tensor of shape (n_stim, n_classes).
        """
        log_likelihoods = inference.gaussian_log_likelihoods(
            responses,
            self.response_statistics["means"],
            self.response_statistics["covariances"],
        )
        return log_likelihoods

    @property
    def response_statistics(self):
        """
        Return the class-conditional response statistics.

        Returns
        -------
        dict
            A dictionary containing:
            - 'means': torch.Tensor of shape (n_classes, n_filters).
            - 'covariances': torch.Tensor of shape (n_classes, n_filters, n_filters).
        """
        flat_filters = torch.flatten(self.filters, -2, -1)
        dtype = flat_filters.dtype
        device = flat_filters.device
        n_filters = flat_filters.shape[0]

        response_means = torch.einsum(
            "cd,kd->ck", self.stimulus_statistics["means"], flat_filters
        )

        noise_covariance = (
            torch.eye(n_filters, dtype=dtype, device=device) * self.response_noise
        )
        response_covariances = torch.einsum(
            "kd,cdb,mb->ckm",
            flat_filters,
            self.stimulus_statistics["covariances"],
            flat_filters,
        )

        response_statistics = {
            "means": response_means,
            "covariances": response_covariances + noise_covariance,
        }
        return response_statistics

    @response_statistics.setter
    def response_statistics(self):
        """
        Prevent direct setting of the response statistics.

        Raises
        ------
        AttributeError
            Raised if trying to set response statistics directly.
        """
        raise AttributeError(
            "Response statistics can't be set directly. "
            "They are computed from the filters and the "
            "stimulus statistics."
        )

