"""Wind estimation and predictive control math for hover."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heligym.envs.helicopter import DT
from pid.controllers.position import MAX_TILT_RAD
from pid.models.full_control import FullControlEffectsModel
from pid.obs import FULL_OBS

MAX_WIND_EST_ACCEL = 100.0


@dataclass
class WindEstimate:
    """Estimated disturbance acceleration (ft/s^2) in N/E/Z axes."""

    n: float = 0.0
    e: float = 0.0
    z: float = 0.0


def update_wind_estimate(
    estimate: WindEstimate,
    *,
    residual_accel: tuple[float, float, float],
    alpha: float,
) -> WindEstimate:
    """EMA update for disturbance acceleration estimate."""
    if alpha <= 0.0:
        return estimate
    if alpha >= 1.0:
        return WindEstimate(*residual_accel)
    return WindEstimate(
        n=(1.0 - alpha) * estimate.n + alpha * residual_accel[0],
        e=(1.0 - alpha) * estimate.e + alpha * residual_accel[1],
        z=(1.0 - alpha) * estimate.z + alpha * residual_accel[2],
    )


def compute_damped_disturbance_innovation(
    *,
    d_pos: float,
    d_vel: float,
    horizon_s: float,
    coverage: float,
) -> float:
    """Damped discrepancy-domain innovation using coverage+horizon law."""
    if horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    t = horizon_s
    return 2.0 * (-coverage * d_pos - t * d_vel) / (t * t)


def apply_gaussian_spawn_perturbation(
    env,
    std: float,
) -> tuple[float, float, float]:
    """Perturb N/E position and altitude: offset ~ N(0, std) ft from spawn."""
    if std <= 0.0:
        return 0.0, 0.0, 0.0

    n_offset = float(env.np_random.normal(0.0, std))
    e_offset = float(env.np_random.normal(0.0, std))
    z_offset = float(env.np_random.normal(0.0, std))
    env.heli_dyn.state["xyz"][0] += n_offset
    env.heli_dyn.state["xyz"][1] += e_offset
    env.heli_dyn.state["xyz"][2] -= z_offset
    env.heli_dyn.dynamics(env.heli_dyn.state, set_observation=True)
    env._prev_npos = float(env.heli_dyn.observation[FULL_OBS.npos])
    env._prev_epos = float(env.heli_dyn.observation[FULL_OBS.epos])
    return n_offset, e_offset, z_offset


def compute_predictive_accelerations(
    *,
    npos: float,
    epos: float,
    alt: float,
    nvel: float,
    evel: float,
    vel_up: float,
    target_alt: float,
    horizon_s: float,
    coverage: float = 0.5,
    wind_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Kinematic predictive accelerations covering `coverage` of the error in horizon T."""
    if horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    t = horizon_s
    t2 = t * t
    a_n_raw = 2.0 * (-coverage * npos - t * nvel) / t2
    a_e_raw = 2.0 * (-coverage * epos - t * evel) / t2
    a_z_raw = 2.0 * (coverage * (target_alt - alt) - t * vel_up) / t2
    a_n = a_n_raw - wind_bias[0]
    a_e = a_e_raw - wind_bias[1]
    a_z = a_z_raw - wind_bias[2]
    return a_n, a_e, a_z


def compute_accel_vector(
    full_obs: np.ndarray,
    *,
    target_alt: float,
    gravity: float,
    mass: float,
    prev_alt: float,
    horizon_s: float,
    coverage: float = 0.5,
    wind_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build [roll, pitch, yaw, vertical_force] vector for B.T @ u inversion."""
    npos = float(full_obs[FULL_OBS.npos])
    epos = float(full_obs[FULL_OBS.epos])
    alt = float(full_obs[FULL_OBS.alt])
    nvel = float(full_obs[FULL_OBS.nvel])
    evel = float(full_obs[FULL_OBS.evel])
    vel_up = (alt - prev_alt) / DT

    a_n, a_e, a_z = compute_predictive_accelerations(
        npos=npos,
        epos=epos,
        alt=alt,
        nvel=nvel,
        evel=evel,
        vel_up=vel_up,
        target_alt=target_alt,
        horizon_s=horizon_s,
        coverage=coverage,
        wind_bias=wind_bias,
    )

    pitch_des = float(np.clip(-a_n / gravity, -MAX_TILT_RAD, MAX_TILT_RAD))
    roll_des = float(np.clip(a_e / gravity, -MAX_TILT_RAD, MAX_TILT_RAD))
    yaw_des = 0.0

    roll = float(full_obs[FULL_OBS.roll])
    pitch = float(full_obs[FULL_OBS.pitch])
    yaw = float(full_obs[FULL_OBS.yaw])
    rollrate = float(full_obs[FULL_OBS.rollrate])
    pitchrate = float(full_obs[FULL_OBS.pitchrate])
    yawrate = float(full_obs[FULL_OBS.yawrate])

    t = horizon_s
    t2 = t * t
    roll_acc = 2.0 * (roll_des - roll - t * rollrate) / t2
    pitch_acc = 2.0 * (pitch_des - pitch - t * pitchrate) / t2
    yaw_acc = 2.0 * (yaw_des - yaw - t * yawrate) / t2
    vertical_force = a_z * mass

    return np.array([roll_acc, pitch_acc, yaw_acc, vertical_force], dtype=np.float64)


def compute_predictive_action(
    full_obs: np.ndarray,
    *,
    target_alt: float,
    effects: FullControlEffectsModel,
    trim_action: np.ndarray,
    gravity: float,
    action_limit: float,
    prev_alt: float,
    horizon_steps: int,
    coverage: float = 0.5,
    wind_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """One control step: predictive accel vector -> u = pinv(B.T) @ AV."""
    horizon_s = horizon_steps * DT
    accel_vector = compute_accel_vector(
        full_obs,
        target_alt=target_alt,
        gravity=gravity,
        mass=effects.mass,
        prev_alt=prev_alt,
        horizon_s=horizon_s,
        coverage=coverage,
        wind_bias=wind_bias,
    )
    delta_u = effects.invert(accel_vector)
    action = trim_action + delta_u.astype(np.float32)
    return np.clip(action, -action_limit, action_limit).astype(np.float32)
