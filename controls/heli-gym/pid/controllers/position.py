"""Cascade position controller on top of attitude stabilization."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from simple_pid import PID

from heligym.envs.helicopter import DT
from pid.constants import (
    ACCEL_OUTPUT_LIMITS,
    DEFAULT_INVERT_DT_STEPS,
    RATE_PID_OUTPUT_LIMITS,
    VEL_OUTPUT_LIMITS,
)
from pid.estimation.online_rls import RecursiveLeastSquares
from pid.models.angular import AngularDynamicsModel
from pid.models.vertical import VerticalControlEffects
from pid.obs import FULL_OBS
from pid.pid_factory import make_axis_pids, reset_pids
from skills.training.envs import HeliStabilization

MAX_TILT_RAD = math.radians(15.0)
ATT_STABLE_THRESHOLD_RAD = 0.1
ATT_STABLE_WINDOW_STEPS = 50
HORIZ_AXES = ("n", "e")


@dataclass
class PositionEpisodeResult:
    total_reward: float
    steps_survived: int
    max_pos_error: float
    max_alt_error: float
    max_roll: float
    max_pitch: float
    max_yaw: float
    terminated: bool
    steps_to_pos_active: int


@dataclass
class AdaptiveGainConfig:
    """Gradient-free online gain adjustment from rolling position error."""

    window_size: int = 5
    adapt_every: int = 20
    adapt_rate: float = 0.02
    target_rms: float = 50.0
    kp_min: float = 0.01
    kp_max: float = 0.3
    scale_output_limits: bool = False
    max_output_scale: float = 3.0
    pos_kp_baseline: dict[str, float] = field(default_factory=dict)
    pos_ki_baseline: dict[str, float] = field(default_factory=dict)
    pos_kd_baseline: dict[str, float] = field(default_factory=dict)
    vel_kp_baseline: dict[str, float] = field(default_factory=dict)
    vel_ki_baseline: dict[str, float] = field(default_factory=dict)
    vel_kd_baseline: dict[str, float] = field(default_factory=dict)
    pos_out_baseline: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict,
    )
    vel_out_baseline: dict[str, tuple[float | None, float | None]] = field(
        default_factory=dict,
    )


def _scale_pid_output_limits(
    pid: PID,
    baseline_kp: float,
    baseline_limits: tuple[float | None, float | None],
    *,
    max_scale: float,
) -> None:
    """Optionally scale PID output limits with Kp, capped at max_scale × baseline."""
    if baseline_kp <= 0.0:
        return
    lo, hi = baseline_limits
    if lo is None or hi is None:
        return
    scale = float(np.clip(pid.Kp / baseline_kp, 1.0, max_scale))
    pid.output_limits = (lo * scale, hi * scale)


def _restore_pid_output_limits(
    pid: PID,
    baseline_limits: tuple[float | None, float | None],
) -> None:
    pid.output_limits = baseline_limits


def _action_saturated(action: np.ndarray, action_limit: float, tol: float = 1e-4) -> bool:
    return bool(np.any(np.abs(action) >= action_limit - tol))


def _allow_rls_update(
    env: HeliStabilization,
    *,
    pos_active: bool,
    action: np.ndarray,
    action_limit: float,
    max_turb_lvl: int,
) -> bool:
    if not pos_active:
        return False
    if _action_saturated(action, action_limit):
        return False
    return int(env.wind_dyn.turbulence_level) <= max_turb_lvl


def _capture_kp_baselines(
    pos_pids: dict[str, PID],
    vel_pids: dict[str, PID],
    cfg: AdaptiveGainConfig,
) -> None:
    if not cfg.pos_kp_baseline:
        cfg.pos_kp_baseline = {axis: pid.Kp for axis, pid in pos_pids.items()}
        cfg.pos_ki_baseline = {axis: pid.Ki for axis, pid in pos_pids.items()}
        cfg.pos_kd_baseline = {axis: pid.Kd for axis, pid in pos_pids.items()}
        cfg.vel_kp_baseline = {axis: pid.Kp for axis, pid in vel_pids.items()}
        cfg.vel_ki_baseline = {axis: pid.Ki for axis, pid in vel_pids.items()}
        cfg.vel_kd_baseline = {axis: pid.Kd for axis, pid in vel_pids.items()}
        cfg.pos_out_baseline = {axis: pid.output_limits for axis, pid in pos_pids.items()}
        cfg.vel_out_baseline = {axis: pid.output_limits for axis, pid in vel_pids.items()}


def _reset_adaptive_gains(
    pos_pids: dict[str, PID],
    vel_pids: dict[str, PID],
    cfg: AdaptiveGainConfig,
) -> None:
    _capture_kp_baselines(pos_pids, vel_pids, cfg)
    for axis, pid in pos_pids.items():
        pid.Kp = cfg.pos_kp_baseline[axis]
        pid.Ki = cfg.pos_ki_baseline[axis]
        pid.Kd = cfg.pos_kd_baseline[axis]
        if cfg.scale_output_limits:
            _scale_pid_output_limits(
                pid,
                cfg.pos_kp_baseline[axis],
                cfg.pos_out_baseline[axis],
                max_scale=cfg.max_output_scale,
            )
        else:
            _restore_pid_output_limits(pid, cfg.pos_out_baseline[axis])
    for axis, pid in vel_pids.items():
        pid.Kp = cfg.vel_kp_baseline[axis]
        pid.Ki = cfg.vel_ki_baseline[axis]
        pid.Kd = cfg.vel_kd_baseline[axis]
        if cfg.scale_output_limits:
            _scale_pid_output_limits(
                pid,
                cfg.vel_kp_baseline[axis],
                cfg.vel_out_baseline[axis],
                max_scale=cfg.max_output_scale,
            )
        else:
            _restore_pid_output_limits(pid, cfg.vel_out_baseline[axis])


def _adjust_gains(
    pos_pids: dict[str, PID],
    vel_pids: dict[str, PID],
    rms: float,
    current_error: float,
    cfg: AdaptiveGainConfig,
) -> None:
    """Nudge horizontal position/velocity Kp from rolling position error."""
    effective_err = max(rms, current_error)
    ratio = effective_err / cfg.target_rms
    if ratio > 1.0:
        kp_factor = 1.0 + cfg.adapt_rate * (ratio - 1.0)
    elif ratio < 0.7:
        kp_factor = 1.0 - cfg.adapt_rate * (1.0 - ratio)
    else:
        return

    kp_factor = float(np.clip(kp_factor, 0.95, 1.05))

    for axis in HORIZ_AXES:
        pos_pid = pos_pids[axis]
        pos_pid.Kp = float(np.clip(pos_pid.Kp * kp_factor, cfg.kp_min, cfg.kp_max))
        if cfg.scale_output_limits:
            _scale_pid_output_limits(
                pos_pid,
                cfg.pos_kp_baseline[axis],
                cfg.pos_out_baseline[axis],
                max_scale=cfg.max_output_scale,
            )

        vel_pid = vel_pids[axis]
        vel_pid.Kp = float(np.clip(vel_pid.Kp * kp_factor, cfg.kp_min, cfg.kp_max))
        if cfg.scale_output_limits:
            _scale_pid_output_limits(
                vel_pid,
                cfg.vel_kp_baseline[axis],
                cfg.vel_out_baseline[axis],
                max_scale=cfg.max_output_scale,
            )


def spawn_target_alt(env: HeliStabilization) -> float:
    """Absolute altitude (ft) from the current dynamics observation."""
    return float(env.heli_dyn.observation[FULL_OBS.alt])


def resolve_target_alt(env: HeliStabilization, target_alt: float | None) -> float:
    if target_alt is not None:
        return target_alt
    return spawn_target_alt(env)


def preview_target_alt(
    env: HeliStabilization,
    *,
    target_alt: float | None,
    seed: int,
) -> float:
    if target_alt is not None:
        return target_alt
    env.reset(seed=seed)
    apply_initial_position_perturbation(env)
    return spawn_target_alt(env)


def apply_initial_position_perturbation(env: HeliStabilization) -> None:
    init_pos_range = getattr(env, "_init_pos_range", 0.0)
    if init_pos_range <= 0.0:
        return

    n_offset = float(env.np_random.uniform(-init_pos_range, init_pos_range))
    e_offset = float(env.np_random.uniform(-init_pos_range, init_pos_range))
    env.heli_dyn.state["xyz"][0] += n_offset
    env.heli_dyn.state["xyz"][1] += e_offset
    env.heli_dyn.dynamics(env.heli_dyn.state, set_observation=True)
    env._prev_npos = float(env.heli_dyn.observation[FULL_OBS.npos])
    env._prev_epos = float(env.heli_dyn.observation[FULL_OBS.epos])


def compute_attitude_control_deltas(
    full_obs: np.ndarray,
    att_pids: dict[str, PID],
    att_model: AngularDynamicsModel,
    *,
    roll_setpoint: float,
    pitch_setpoint: float,
    yaw_setpoint: float,
    invert_dt: float,
) -> tuple[float, float, float]:
    """Map angle PIDs -> desired rates -> lon/lat/pedal via model inversion."""
    # Full dynamics observation indices (FullObs).
    roll = float(full_obs[FULL_OBS.roll])
    pitch = float(full_obs[FULL_OBS.pitch])
    yaw = float(full_obs[FULL_OBS.yaw])
    w_t = np.array(
        [
            float(full_obs[FULL_OBS.rollrate]),
            float(full_obs[FULL_OBS.pitchrate]),
            float(full_obs[FULL_OBS.yawrate]),
        ],
        dtype=np.float64,
    )

    att_pids["roll"].setpoint = roll_setpoint
    att_pids["pitch"].setpoint = pitch_setpoint
    att_pids["yaw"].setpoint = yaw_setpoint

    w_des = np.array(
        [
            att_pids["roll"](roll),
            att_pids["pitch"](pitch),
            att_pids["yaw"](yaw),
        ],
        dtype=np.float64,
    )
    d_lon, d_lat, d_pedal = att_model.invert(w_t, w_des, invert_dt)
    return float(d_lon), float(d_lat), float(d_pedal)


def compute_cascade_action(
    full_obs: np.ndarray,
    *,
    target_alt: float,
    pos_pids: dict[str, PID],
    vel_pids: dict[str, PID],
    att_pids: dict[str, PID],
    vertical_effects: VerticalControlEffects,
    att_model: AngularDynamicsModel,
    trim_action: np.ndarray,
    gravity: float,
    action_limit: float,
    invert_dt: float,
    prev_alt: float,
) -> np.ndarray:
    # Full dynamics observation indices (FullObs).
    npos = float(full_obs[FULL_OBS.npos])
    epos = float(full_obs[FULL_OBS.epos])
    alt = float(full_obs[FULL_OBS.alt])
    nvel = float(full_obs[FULL_OBS.nvel])
    evel = float(full_obs[FULL_OBS.evel])
    vel_up = (alt - prev_alt) / DT

    pos_pids["n"].setpoint = 0.0
    pos_pids["e"].setpoint = 0.0
    pos_pids["alt"].setpoint = target_alt

    vn_des = pos_pids["n"](npos)
    ve_des = pos_pids["e"](epos)
    vz_des = pos_pids["alt"](alt)

    vel_pids["n"].setpoint = vn_des
    vel_pids["e"].setpoint = ve_des
    vel_pids["alt"].setpoint = vz_des

    an_des = vel_pids["n"](nvel)
    ae_des = vel_pids["e"](evel)
    az_des = vel_pids["alt"](vel_up)

    pitch_des = float(np.clip(math.atan2(-an_des, gravity), -MAX_TILT_RAD, MAX_TILT_RAD))
    roll_des = float(np.clip(math.atan2(ae_des, gravity), -MAX_TILT_RAD, MAX_TILT_RAD))

    d_lon, d_lat, d_pedal = compute_attitude_control_deltas(
        full_obs,
        att_pids,
        att_model,
        roll_setpoint=roll_des,
        pitch_setpoint=pitch_des,
        yaw_setpoint=0.0,
        invert_dt=invert_dt,
    )
    force_residual = (
        az_des * vertical_effects.mass
        - vertical_effects.bias
        - vertical_effects.k_lon * d_lon
        - vertical_effects.k_lat * d_lat
        - vertical_effects.k_pedal * d_pedal
    )
    if abs(vertical_effects.k_col) < 1e-9:
        raise ValueError(f"Invalid vertical k_col={vertical_effects.k_col}")
    d_col = force_residual / vertical_effects.k_col

    action = trim_action + np.array([d_col, d_lon, d_lat, d_pedal], dtype=np.float32)
    return np.clip(action, -action_limit, action_limit).astype(np.float32)


def compute_att_only_action(
    full_obs: np.ndarray,
    att_pids: dict[str, PID],
    att_model: AngularDynamicsModel,
    trim_action: np.ndarray,
    action_limit: float,
    invert_dt: float,
) -> np.ndarray:
    d_lon, d_lat, d_pedal = compute_attitude_control_deltas(
        full_obs,
        att_pids,
        att_model,
        roll_setpoint=0.0,
        pitch_setpoint=0.0,
        yaw_setpoint=0.0,
        invert_dt=invert_dt,
    )
    action = trim_action + np.array([0.0, d_lon, d_lat, d_pedal], dtype=np.float32)
    return np.clip(action, -action_limit, action_limit).astype(np.float32)


def run_episode(
    env: HeliStabilization,
    pos_pids: dict[str, PID],
    vel_pids: dict[str, PID],
    att_pids: dict[str, PID],
    vertical_effects: VerticalControlEffects,
    att_model: AngularDynamicsModel,
    *,
    target_alt: float | None = None,
    action_limit: float,
    max_steps: int,
    seed: int | None = None,
    att_stable_threshold: float = ATT_STABLE_THRESHOLD_RAD,
    att_stable_window: int = ATT_STABLE_WINDOW_STEPS,
    invert_dt: float = DEFAULT_INVERT_DT_STEPS * DT,
    render: bool = False,
    verbose: bool = False,
    adaptive_gain_cfg: AdaptiveGainConfig | None = None,
    rls: RecursiveLeastSquares | None = None,
    rls_apply_model: bool = False,
    rls_max_turb_lvl: int = 2,
) -> PositionEpisodeResult:
    env.reset(seed=seed)
    apply_initial_position_perturbation(env)
    effective_target_alt = resolve_target_alt(env, target_alt)
    env.set_position_target(npos=0.0, epos=0.0, alt=effective_target_alt)
    trim_action = env.heli_dyn.action.astype(np.float32).copy()
    prev_alt = float(env.heli_dyn.observation[FULL_OBS.alt])

    total_reward = 0.0
    max_pos_error = 0.0
    max_alt_error = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    max_yaw = 0.0
    steps = 0
    terminated = False
    pos_active = False
    att_stable_count = 0
    steps_to_pos_active = 0
    err_window: deque[float] | None = (
        deque(maxlen=adaptive_gain_cfg.window_size) if adaptive_gain_cfg is not None else None
    )
    if adaptive_gain_cfg is not None:
        _reset_adaptive_gains(pos_pids, vel_pids, adaptive_gain_cfg)
    w_t_prev = np.array(
        [
            float(env.heli_dyn.observation[FULL_OBS.rollrate]),
            float(env.heli_dyn.observation[FULL_OBS.pitchrate]),
            float(env.heli_dyn.observation[FULL_OBS.yawrate]),
        ],
        dtype=np.float64,
    )

    for _ in range(max_steps):
        if render and env.renderer is not None and env.renderer.is_close():
            break

        full_obs = env.heli_dyn.observation
        npos = float(full_obs[FULL_OBS.npos])
        epos = float(full_obs[FULL_OBS.epos])
        alt = float(full_obs[FULL_OBS.alt])
        roll = float(full_obs[FULL_OBS.roll])
        pitch = float(full_obs[FULL_OBS.pitch])
        yaw = float(full_obs[FULL_OBS.yaw])

        pos_error = math.hypot(npos, epos)
        alt_error = abs(alt - effective_target_alt)
        max_pos_error = max(max_pos_error, pos_error)
        max_alt_error = max(max_alt_error, alt_error)
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))
        max_yaw = max(max_yaw, abs(yaw))

        if not pos_active:
            if (
                abs(roll) < att_stable_threshold
                and abs(pitch) < att_stable_threshold
                and abs(yaw) < att_stable_threshold
            ):
                att_stable_count += 1
            else:
                att_stable_count = 0

            if att_stable_count >= att_stable_window:
                pos_active = True
                steps_to_pos_active = steps
                reset_pids(pos_pids, vel_pids)
                if verbose:
                    print(
                        f"step={steps} POSITION CONTROL ACTIVATED "
                        f"(attitude stable for {att_stable_window} steps)"
                    )

        if pos_active:
            action = compute_cascade_action(
                full_obs,
                target_alt=effective_target_alt,
                pos_pids=pos_pids,
                vel_pids=vel_pids,
                att_pids=att_pids,
                vertical_effects=vertical_effects,
                att_model=att_model,
                trim_action=trim_action,
                gravity=float(env.heli_dyn.ENV["GRAV"]),
                action_limit=action_limit,
                invert_dt=invert_dt,
                prev_alt=prev_alt,
            )
        else:
            action = compute_att_only_action(
                full_obs,
                att_pids,
                att_model,
                trim_action,
                action_limit,
                invert_dt,
            )

        prev_alt = alt
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

        if rls is not None:
            new_full_obs = env.heli_dyn.observation
            w_t_new = np.array(
                [
                    float(new_full_obs[FULL_OBS.rollrate]),
                    float(new_full_obs[FULL_OBS.pitchrate]),
                    float(new_full_obs[FULL_OBS.yawrate]),
                ],
                dtype=np.float64,
            )
            if _allow_rls_update(
                env,
                pos_active=pos_active,
                action=action,
                action_limit=action_limit,
                max_turb_lvl=rls_max_turb_lvl,
            ):
                alpha_measured = (w_t_new - w_t_prev) / DT
                u_ctrl = action[1:4] - trim_action[1:4]
                phi = np.concatenate([[1.0], u_ctrl.astype(np.float64)])
                rls.update(phi, alpha_measured)
                if rls_apply_model:
                    att_model.apply_rls(rls)
            w_t_prev = w_t_new

        if err_window is not None and pos_active:
            err_window.append(pos_error)
            if (
                adaptive_gain_cfg is not None
                and steps % adaptive_gain_cfg.adapt_every == 0
                and len(err_window) >= 1
            ):
                rms = math.sqrt(sum(e * e for e in err_window) / len(err_window))
                _adjust_gains(pos_pids, vel_pids, rms, pos_error, adaptive_gain_cfg)

        if render:
            env.render()
        if verbose:
            print(
                f"step={steps} reward={reward:.4f} "
                f"npos={npos:.2f} epos={epos:.2f} alt={alt:.2f} "
                f"roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f} "
                f"action={action.tolist()}"
            )

        if terminated or truncated:
            break

    if not pos_active:
        steps_to_pos_active = steps

    return PositionEpisodeResult(
        total_reward=total_reward,
        steps_survived=steps,
        max_pos_error=max_pos_error,
        max_alt_error=max_alt_error,
        max_roll=max_roll,
        max_pitch=max_pitch,
        max_yaw=max_yaw,
        terminated=terminated,
        steps_to_pos_active=steps_to_pos_active,
    )


def make_position_pids(
    pos_kp: float,
    pos_ki: float,
    pos_kd: float,
    vel_kp: float,
    vel_ki: float,
    vel_kd: float,
    att_kp: float,
    att_ki: float,
    att_kd: float,
    *,
    sample_time: float,
) -> tuple[dict[str, PID], dict[str, PID], dict[str, PID]]:
    pos_pids = make_axis_pids(
        pos_kp, pos_ki, pos_kd,
        sample_time=sample_time,
        output_limits=VEL_OUTPUT_LIMITS,
        axes=("n", "e", "alt"),
    )
    vel_pids = make_axis_pids(
        vel_kp, vel_ki, vel_kd,
        sample_time=sample_time,
        output_limits=ACCEL_OUTPUT_LIMITS,
        axes=("n", "e", "alt"),
    )
    att_pids = make_axis_pids(
        att_kp, att_ki, att_kd,
        sample_time=sample_time,
        output_limits=RATE_PID_OUTPUT_LIMITS,
        axes=("roll", "pitch", "yaw"),
    )
    return pos_pids, vel_pids, att_pids
