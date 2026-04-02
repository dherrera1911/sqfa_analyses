from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import parametrizations


def _import_ot():
    try:
        import ot

        return ot
    except Exception as first_error:
        candidate_dirs = [
            Path(__file__).resolve().parent / "POT",
            Path(__file__).resolve().parent.parent / "wdatorch" / "POT",
        ]
        for pot_dir in candidate_dirs:
            if pot_dir.exists():
                sys.path.insert(0, str(pot_dir))
                try:
                    import ot

                    return ot
                except Exception:
                    continue
        raise ImportError(
            "Unable to import POT. Install POT in the active Python environment."
        ) from first_error


ot = _import_ot()


class WDATorch(nn.Module):
    """PyTorch implementation of Wasserstein Discriminant Analysis.

    The model keeps the projection matrix orthonormal through PyTorch's
    orthogonal parametrization and uses POT to compute the entropic OT terms.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        *,
        reg: float = 1.0,
        sinkhorn_method: str = "sinkhorn_log",
        sinkhorn_iterations: int = 50,
        sinkhorn_tolerance: float = 1e-9,
        gradient_mode: str = "autodiff",
        eps: float = 1e-12,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if output_dim > input_dim:
            raise ValueError("output_dim must be <= input_dim.")
        if sinkhorn_method not in {"sinkhorn", "sinkhorn_log"}:
            raise ValueError("sinkhorn_method must be 'sinkhorn' or 'sinkhorn_log'.")
        if gradient_mode not in {"autodiff", "last_step"}:
            raise ValueError("gradient_mode must be 'autodiff' or 'last_step'.")
        if gradient_mode == "last_step" and sinkhorn_method != "sinkhorn_log":
            raise ValueError(
                "gradient_mode='last_step' requires sinkhorn_method='sinkhorn_log'."
            )

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reg = reg
        self.sinkhorn_method = sinkhorn_method
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.gradient_mode = gradient_mode
        self.eps = eps

        self.projection = nn.Linear(input_dim, output_dim, bias=False, **factory_kwargs)
        nn.init.orthogonal_(self.projection.weight)
        parametrizations.orthogonal(self.projection, "weight")

        buffer_kwargs = {}
        if device is not None:
            buffer_kwargs["device"] = device
        if dtype is not None:
            buffer_kwargs["dtype"] = dtype
        self.register_buffer("mean_", torch.zeros((), **buffer_kwargs))
        self.is_fitted_ = False

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"reg={self.reg}, sinkhorn_method='{self.sinkhorn_method}', "
            f"sinkhorn_iterations={self.sinkhorn_iterations}, "
            f"gradient_mode='{self.gradient_mode}'"
        )

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self.projection.weight.transpose(0, 1)

    def _coerce_inputs(
        self, X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = self.projection.weight
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=weight.dtype, device=weight.device)
        else:
            X = X.to(device=weight.device, dtype=weight.dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected X with shape (n_samples, {self.input_dim}), got {tuple(X.shape)}."
            )

        if y is None:
            return X, None

        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=weight.device)
        else:
            y = y.to(device=weight.device)
        y = y.reshape(-1)

        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        return X, y

    def _split_classes(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> list[torch.Tensor]:
        classes = []
        for cls in torch.unique(y, sorted=True):
            cls_idx = torch.nonzero(y == cls, as_tuple=False).flatten()
            classes.append(X.index_select(0, cls_idx))
        return classes

    def _pairwise_cost(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return ot.dist(x1, x2, metric="sqeuclidean")

    def _uniform_weights(self, n: int, like: torch.Tensor) -> torch.Tensor:
        return torch.full((n,), 1.0 / n, dtype=like.dtype, device=like.device)

    def _sinkhorn_linear_cost(
        self, a: torch.Tensor, b: torch.Tensor, M: torch.Tensor
    ) -> torch.Tensor:
        if self.gradient_mode == "last_step":
            result = ot.solve(
                M,
                a,
                b,
                reg=self.reg,
                max_iter=self.sinkhorn_iterations,
                tol=self.sinkhorn_tolerance,
                grad="last_step",
            )
            return result.value_linear

        plan = ot.bregman.sinkhorn(
            a,
            b,
            M,
            reg=self.reg,
            method=self.sinkhorn_method,
            numItermax=self.sinkhorn_iterations,
            stopThr=self.sinkhorn_tolerance,
            warn=False,
        )
        return torch.sum(plan * M)

    def set_mean(self, X: torch.Tensor) -> torch.Tensor:
        X, _ = self._coerce_inputs(X)
        self.mean_.copy_(X.mean().detach())
        self.is_fitted_ = True
        return self.mean_

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X, _ = self._coerce_inputs(X)
        return self.projection(X - self.mean_)

    def loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        update_mean: bool = False,
    ) -> torch.Tensor:
        X, y = self._coerce_inputs(X, y)

        if update_mean or not self.is_fitted_:
            mean = X.mean()
        else:
            mean = self.mean_

        centered = X - mean
        class_batches = self._split_classes(centered, y)
        projected_batches = [self.projection(batch) for batch in class_batches]
        weights = [self._uniform_weights(batch.shape[0], batch) for batch in projected_batches]

        loss_within = X.new_zeros(())
        loss_between = X.new_zeros(())

        for i, xi in enumerate(projected_batches):
            wi = weights[i]
            for j in range(i, len(projected_batches)):
                xj = projected_batches[j]
                wj = weights[j]
                M = self._pairwise_cost(xi, xj)
                linear_cost = self._sinkhorn_linear_cost(wi, wj, M)
                if i == j:
                    loss_within = loss_within + linear_cost
                else:
                    loss_between = loss_between + linear_cost

        return loss_within / loss_between.clamp_min(self.eps)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        num_steps: int = 100,
        lr: float = 1.0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lbfgs_max_iter: int = 5,
        loss_change_tol: float = 1.0e-7,
        update_mean_each_step: bool = False,
        return_history: bool = False,
        verbose: bool = False,
        print_every: int = 10,
    ):
        X, y = self._coerce_inputs(X, y)
        self.mean_.copy_(X.mean().detach())
        self.is_fitted_ = True

        if optimizer is None:
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=lr,
                max_iter=lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )

        history = []
        for step in range(num_steps):
            if isinstance(optimizer, torch.optim.LBFGS):

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    objective = self.loss(
                        X,
                        y,
                        update_mean=update_mean_each_step,
                    )
                    objective.backward()
                    return objective

                objective = optimizer.step(closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                objective = self.loss(
                    X,
                    y,
                    update_mean=update_mean_each_step,
                )
                objective.backward()
                optimizer.step()

            loss_value = float(objective.detach().cpu())
            history.append(loss_value)
            if len(history) > 1:
                delta_loss = abs(history[-1] - history[-2])
                if delta_loss < loss_change_tol:
                    print(
                        "Stopping early after "
                        f"{step + 1} iterations: |delta_loss|={delta_loss:.3e} "
                        f"< {loss_change_tol:.3e}"
                    )
                    break
            if verbose and (
                step == 0
                or (step + 1) % max(print_every, 1) == 0
                or step + 1 == num_steps
            ):
                print(f"step={step + 1:04d} loss={loss_value:.6f}")

        return history if return_history else self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward(X)

    def fit_transform(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        **fit_kwargs,
    ) -> torch.Tensor:
        self.fit(X, y, **fit_kwargs)
        return self.transform(X)
