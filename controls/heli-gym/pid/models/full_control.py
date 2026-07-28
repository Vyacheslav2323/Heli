"""Full 4x4 linear control-effects model (angular + vertical force)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FullControlEffectsModel:
    """Linear control effects from control_effects_no_bias.csv.

    AV = B.T @ u   =>   u = pinv(B.T) @ AV

    B has shape (n_controls, n_outputs); rows are controls, columns are
    [roll_acceleration, pitch_acceleration, yaw_acceleration, vertical_force].
    The last output is delta_F_up (lb), not ft/s^2.
    """

    B: np.ndarray
    mass: float
    _pinv_bt: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_pinv_bt", np.linalg.pinv(self.B.T))

    def invert(self, accel_vector: np.ndarray) -> np.ndarray:
        """Map desired acceleration/force vector to control deltas."""
        return self._pinv_bt @ accel_vector
