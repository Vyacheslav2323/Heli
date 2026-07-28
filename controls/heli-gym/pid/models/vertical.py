"""Force-based vertical control coupling model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerticalControlEffects:
    """Force-based vertical model: ΔF_up = bias + k_col*d_col + ...  (lb/unit)

    a_up = ΔF_up / mass.  bias should be ~0 for well-trimmed hover.
    """

    mass: float
    bias: float
    k_col: float
    k_lon: float
    k_lat: float
    k_pedal: float

    def collective_for_accel(self, a_up: float) -> float:
        if abs(self.k_col) < 1e-9:
            raise ValueError(f"Invalid k_col={self.k_col}")
        return (a_up * self.mass - self.bias) / self.k_col

    @property
    def hover_collective_delta(self) -> float:
        return self.collective_for_accel(0.0)
