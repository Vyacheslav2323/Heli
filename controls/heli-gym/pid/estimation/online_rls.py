"""Online recursive least squares for angular dynamics identification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pid.models.angular import AngularDynamicsModel


@dataclass
class RecursiveLeastSquares:
    """Multivariate RLS: alpha = W @ phi with forgetting factor."""

    n_regressors: int
    n_outputs: int
    forgetting: float = 0.5
    W: np.ndarray = field(init=False)
    P: list[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0, 1], got {self.forgetting}")
        object.__setattr__(
            self,
            "W",
            np.zeros((self.n_outputs, self.n_regressors), dtype=np.float64),
        )
        object.__setattr__(
            self,
            "P",
            [np.eye(self.n_regressors, dtype=np.float64) * 1e3 for _ in range(self.n_outputs)],
        )

    def update(self, phi: np.ndarray, y: np.ndarray) -> None:
        """Update W given regressor phi and measured output y."""
        phi = np.asarray(phi, dtype=np.float64).reshape(self.n_regressors)
        y = np.asarray(y, dtype=np.float64).reshape(self.n_outputs)
        lam = self.forgetting

        for i in range(self.n_outputs):
            p = self.P[i]
            denom = lam + float(phi @ p @ phi)
            if denom < 1e-12:
                continue
            k = (p @ phi) / denom
            error = y[i] - float(self.W[i] @ phi)
            self.W[i] += error * k
            self.P[i] = (p - np.outer(k, phi @ p)) / lam

    @classmethod
    def from_model(
        cls,
        model: AngularDynamicsModel,
        *,
        forgetting: float = 0.5,
        initial_covariance: float = 1e3,
    ) -> RecursiveLeastSquares:
        """Warm-start from offline AngularDynamicsModel (b, B_ctrl)."""
        n_controls = model.B_ctrl.shape[0]
        n_outputs = model.b.shape[0]
        n_regressors = 1 + n_controls
        rls = cls(
            n_regressors=n_regressors,
            n_outputs=n_outputs,
            forgetting=forgetting,
        )
        rls.W[:, 0] = model.b
        rls.W[:, 1:] = model.B_ctrl.T
        for i in range(n_outputs):
            rls.P[i] = np.eye(n_regressors, dtype=np.float64) * initial_covariance
        return rls
