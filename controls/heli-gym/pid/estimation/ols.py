"""Ordinary least squares fitting utilities."""

from __future__ import annotations

import numpy as np


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the constant bias term."""
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_ols(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit [bias, B] = (X'X)^-1 X'Y and return coefficients plus per-output R²."""
    X_aug = add_bias_column(X)
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y
    B = np.linalg.solve(XtX, XtY)

    Y_hat = X_aug @ B
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 1.0)

    return B, r2


def fit_ols_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = bias + slope * x; return bias, slope, r2."""
    X = np.column_stack([np.ones(x.shape[0]), x])
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    bias = float(coeffs[0])
    slope = float(coeffs[1])

    y_hat = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return bias, slope, r2


def fit_ols_no_bias(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Y = X @ B (no intercept) and return coefficients plus per-output R²."""
    B = np.linalg.solve(X.T @ X, X.T @ Y)

    Y_hat = X @ B
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum(Y**2, axis=0)
    r2 = np.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 1.0)

    return B, r2


def fit_mixed_control_effects(
    X: np.ndarray,
    Y: np.ndarray,
    vertical_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit angular outputs with intercept; fit vertical force through origin."""
    B, r2 = fit_ols(X, Y)
    coeff = np.linalg.solve(X.T @ X, X.T @ Y[:, vertical_idx])
    B[0, vertical_idx] = 0.0
    B[1:, vertical_idx] = coeff

    y_vert = Y[:, vertical_idx]
    y_hat_vert = X @ coeff
    ss_res = float(np.sum((y_vert - y_hat_vert) ** 2))
    ss_tot = float(np.sum((y_vert - np.mean(y_vert)) ** 2))
    r2[vertical_idx] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return B, r2


def collective_delta_for_hover(bias: float, k_col: float) -> float:
    """Delta collective from trim that yields zero vertical acceleration."""
    if abs(k_col) < 1e-9:
        raise ValueError(f"Invalid k_col={k_col}; cannot compute hover collective delta.")
    return -bias / k_col
