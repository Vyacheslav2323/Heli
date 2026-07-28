"""Linear angular dynamics model and inversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pid.estimation.online_rls import RecursiveLeastSquares


@dataclass
class AngularDynamicsModel:
    """Linear angular dynamics: w_{t+1} = w_t + dt * (b + B_ctrl' @ u_t)."""

    b: np.ndarray
    B_ctrl: np.ndarray
    _pinv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pinv", np.linalg.pinv(self.B_ctrl.T))

    def invert(self, w_t: np.ndarray, w_des: np.ndarray, dt: float) -> np.ndarray:
        """Compute control u to achieve w_des from w_t in one step."""
        alpha_des = (w_des - w_t) / dt
        return self._pinv @ (alpha_des - self.b)

    def apply_rls(
        self,
        rls: RecursiveLeastSquares,
        *,
        b_dev_limit: float = 10.0,
        b_ctrl_ratio_limit: float = 2.0,
    ) -> None:
        """Write clamped RLS estimates back into b, B_ctrl, and recompute _pinv."""
        if not hasattr(self, "_b_baseline"):
            object.__setattr__(self, "_b_baseline", self.b.copy())
            object.__setattr__(self, "_B_ctrl_baseline", self.B_ctrl.copy())

        b_new = np.clip(
            rls.W[:, 0],
            self._b_baseline - b_dev_limit,
            self._b_baseline + b_dev_limit,
        )
        b_rls = rls.W[:, 1:].T
        b_base = self._B_ctrl_baseline
        lo = np.minimum(b_base / b_ctrl_ratio_limit, b_base * b_ctrl_ratio_limit)
        hi = np.maximum(b_base / b_ctrl_ratio_limit, b_base * b_ctrl_ratio_limit)
        b_ctrl_new = np.clip(b_rls, lo, hi)

        self.b = b_new.copy()
        self.B_ctrl = b_ctrl_new.copy()
        object.__setattr__(self, "_pinv", np.linalg.pinv(self.B_ctrl.T))
