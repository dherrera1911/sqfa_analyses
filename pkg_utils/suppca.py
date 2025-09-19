from typing import Optional, Union
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils.parametrizations import orthogonal


class SupervisedPCA:
    """Supervised Principal Component Analysis (Barshan et al., 2011)

    This implementation follows the formulation described in
    Ghojogh & Crowley (2022) Tutorial, Section 6.2–6.3.  The projection
    directions **U** are obtained as the leading eigenvectors of the
    matrix ::

        M = X^T H K_y H X,

    where ``H = I - 11^T/n`` is the centring matrix and ``K_y`` is a kernel
    over the target values.  When ``K_y`` is chosen to be the identity
    matrix, the method reduces to standard PCA.

    Parameters
    ----------
    n_components : int, optional
        Number of supervised principal components to keep.  If ``None``
        (default) all components are returned.
    label_kernel : str, optional {"auto", "delta", "linear", "rbf"}
        Which kernel to use for the labels *y*.

        * ``"delta"``  – Kronecker delta kernel k(y_i,y_j)=1[y_i==y_j].
        * ``"linear"`` – Linear kernel ``y y^T`` (sensible for regression).
        * ``"rbf"``    – RBF kernel computed with sklearn’s
                         :pyfunc:`sklearn.metrics.pairwise.rbf_kernel`.
        * ``"auto"``   – ``"delta"`` if *y* is 1‑D / categorical, otherwise
                         ``"linear"``.

    Notes
    -----
    Let *n* be the number of samples and *d* the number of features.
    The computational cost is dominated by the eigen‑decomposition of
    the (*d × d*) matrix *M*; when *d ≫ n* you may prefer the dual form
    (see tutorial, Sec. 6.4) which works in the *n × n* space.
    """

    def __init__(self, n_components: Optional[int] = None,
                 label_kernel: str = "auto") -> None:
        self.n_components = n_components
        self.label_kernel = label_kernel
        # learned attributes
        self.components_: Optional[np.ndarray] = None  # (p, d)
        self.explained_variance_: Optional[np.ndarray] = None  # (p,)
        self._Ky: Optional[np.ndarray] = None  # stored for inspection

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _build_label_kernel(self, y: np.ndarray) -> np.ndarray:
        """Return Ky ∈ ℝ^{n×n} given targets y (shape (n,) or (n,ℓ))."""
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n = y.shape[0]

        # choose kernel
        kernel = self.label_kernel
        if kernel == "auto":
            kernel = "delta" if y.shape[1] == 1 else "linear"

        if kernel == "delta":
            Ky = (y == y.T).astype(np.uint8)
        else:
            raise ValueError(f"Unknown label_kernel: {self.label_kernel}")

        return Ky

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Union[np.ndarray, list]) -> "SupervisedPCABarshan":
        """Estimate supervised principal components from data.

        Parameters
        ----------
        X : array‑like, shape (n_samples, n_features)
            Training data.
        y : array‑like, shape (n_samples,) or (n_samples, ℓ)
            Target values.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # centring matrix
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

        # Ky kernel over labels
        Ky = self._build_label_kernel(y)

        # core matrix M = X^T H Ky H X  (d × d, symmetric)
        M = X.T @ H @ Ky @ H @ X

        # eigen‑decomposition (since M is symmetric)
        eigvals, eigvecs = np.linalg.eigh(M)
        order = eigvals.argsort()[::-1]  # descending
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        p = n_features if self.n_components is None else self.n_components
        p = min(p, n_features)
        self.components_ = eigvecs[:, :p].T        # (p, d)
        self.explained_variance_ = eigvals[:p]
        self._Ky = Ky
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data into the learned supervised PC subspace."""
        if self.components_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        return np.asarray(X) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y: Union[np.ndarray, list]) -> np.ndarray:
        """Fit the model to *X, y* and return the projected data."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_spca: np.ndarray) -> np.ndarray:
        """Reconstruct data from its SPCA representation."""
        if self.components_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        return np.asarray(X_spca) @ self.components_

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def explained_variance_ratio(self) -> np.ndarray:
        """Return the proportion of HSIC captured by each component."""
        if self.explained_variance_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        return self.explained_variance_ / self.explained_variance_.sum()



class SparseRatioLDA(nn.Module):
    """Sparse Trace Ratio LDA"""

    def __init__(self, n_dim, n_filters, p=1.0, gamma=0.1, alpha=0.001):
        """
        Parameters
        ----------
        n_dim : int
            Dimension of the input data space.

        n_filters : int
            Number of filters to use.

        p : float
            Exponent in the sparsity norm.

        gamma : float
            Weight of sparseness penalty

        alpha : float
            Added identity to the within-class scatter matrix.
        """
        super().__init__()

        filters = torch.randn(n_filters, n_dim)
        self.filters = nn.Parameter(filters)
        orthogonal(self, "filters")
        self.register_buffer("p", torch.as_tensor(p))
        self.register_buffer("gamma", torch.as_tensor(gamma))
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("Id", torch.eye(n_filters, device=filters.device))


    def trace_ratio(self, S_b, S_w):
        """Compute the trace ratio."""
        # Transform the matrices to feature space
        S_b_features = self.filters @ S_b @ self.filters.T
        S_w_features = self.filters @ S_w @ self.filters.T
        # Add identity to the within-class scatter matrix
        S_w_features += self.alpha * self.Id

        return torch.trace(S_b_features) / torch.trace(S_w_features)


    def loss(self, S_b, S_w):
        """Compute the loss."""
        trace_ratio = self.trace_ratio(S_b, S_w)
        sparsity_penalty = self.gamma * torch.norm(self.filters, p=self.p)

        return -trace_ratio + sparsity_penalty


    def fit(
        self,
        S_b: torch.Tensor,
        S_w: torch.Tensor,
        n_iter: int = 1000,
        lr: float = 1.0,
        atol: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Optimize the projection matrix `self.filters` with LBFGS.

        Parameters
        ----------
        S_b, S_w : torch.Tensor
            Between‑ and within‑class scatter matrices (must be square
            with the same dimensionality as the input space).
        n_iter : int
            Maximum number of LBFGS iterations/line‑search steps.
        lr : float
            Initial step size for LBFGS.
        atol : float
            Absolute tolerance for early stopping on the loss.
        line_search_fn : str
            Line‑search strategy passed to `torch.optim.LBFGS`.

        Returns
        -------
        self
        """
        device = self.filters.device
        S_b = S_b.to(device, dtype=self.filters.dtype)
        S_w = S_w.to(device, dtype=self.filters.dtype)

        # LBFGS works with a *closure* that reevaluates the model.
        optimizer = optim.LBFGS(
            self.parameters(),
            lr=lr,
        )

        prev_loss = None

        def closure():
            optimizer.zero_grad()
            loss_val = self.loss(S_b, S_w)
            loss_val.backward()
            return loss_val

        for it in range(n_iter):
            loss = optimizer.step(closure)

            if verbose and (it % 5 == 0 or it == n_iter - 1):
                print(f"Iter {it:4d} | loss = {loss.item():.6e}")

            # Early stopping
            if prev_loss is not None and abs(prev_loss - loss.item()) < atol:
                if verbose:
                    print(f"Converged (|Δℓ| < {atol}) after {it+1} iterations.")
                break
            prev_loss = loss.item()

        return self

