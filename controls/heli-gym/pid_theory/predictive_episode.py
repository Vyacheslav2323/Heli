"""Episode rollout for predictive hover."""

from __future__ import annotations

import argparse
import math

import numpy as np

from heligym.envs.helicopter import DT, FPS
from pid.models.full_control import FullControlEffectsModel
from pid.obs import FULL_OBS
from pid_theory.predictive_control import (
    MAX_WIND_EST_ACCEL,
    WindEstimate,
    apply_gaussian_spawn_perturbation,
    compute_damped_disturbance_innovation,
    compute_predictive_accelerations,
    compute_predictive_action,
    update_wind_estimate,
)


def _run_episode(
    args: argparse.Namespace,
    *,
    env,
    effects: FullControlEffectsModel,
    gravity: float,
    seed: int,
    episode: int,
) -> dict[str, float | int | bool]:
    env.reset(seed=seed)
    spawn_alt = float(env.heli_dyn.observation[FULL_OBS.alt])
    n_off, e_off, z_off = apply_gaussian_spawn_perturbation(env, args.init_pos_std)
    target_alt = args.target_alt if args.target_alt is not None else spawn_alt
    env.set_position_target(npos=0.0, epos=0.0, alt=target_alt)

    trim_action = env.heli_dyn.action.astype(np.float32).copy()
    prev_alt = float(env.heli_dyn.observation[FULL_OBS.alt])
    prev_npos = float(env.heli_dyn.observation[FULL_OBS.npos])
    prev_epos = float(env.heli_dyn.observation[FULL_OBS.epos])
    prev_nvel = float(env.heli_dyn.observation[FULL_OBS.nvel])
    prev_evel = float(env.heli_dyn.observation[FULL_OBS.evel])
    prev_vel_up = 0.0
    wind_est = WindEstimate()
    prev_cmd_accel = (0.0, 0.0, 0.0)
    prev_action_saturated = False
    has_prev = False
    last_d_pos = (0.0, 0.0, 0.0)
    last_d_vel = (0.0, 0.0, 0.0)
    wind_est_coverage = args.wind_est_coverage if args.wind_est_coverage is not None else args.coverage
    wind_est_horizon_s = args.wind_est_horizon_scale * args.horizon * DT
    max_steps = int(args.max_time * FPS)

    episode_prefix = f"[episode {episode + 1}/{args.n_episodes}] " if args.n_episodes > 1 else ""
    print(
        f"{episode_prefix}Predictive hover: seed={seed}, "
        f"horizon={args.horizon} steps ({args.horizon * DT:.3f} s), "
        f"coverage={args.coverage:.3f}, "
        f"wind_est_alpha={args.wind_est_alpha:.3f}, "
        f"wind_est_cov={wind_est_coverage:.3f}, "
        f"wind_est_hscale={args.wind_est_horizon_scale:.3f}, "
        f"init_pos_std={args.init_pos_std:.1f} ft, "
        f"spawn_offset=(n={n_off:+.2f}, e={e_off:+.2f}, z={z_off:+.2f}) ft, "
        f"target_alt={target_alt:.1f} ft, trim={np.round(trim_action, 4).tolist()}, "
        f"csv={args.control_effects_csv}"
    )

    total_reward = 0.0
    max_pos_err = 0.0
    max_alt_err = 0.0
    steps = 0
    terminated = False

    for step in range(max_steps):
        if args.render and env.renderer is not None and env.renderer.is_close():
            break

        full_obs = env.heli_dyn.observation
        npos = float(full_obs[FULL_OBS.npos])
        epos = float(full_obs[FULL_OBS.epos])
        alt = float(full_obs[FULL_OBS.alt])
        nvel = float(full_obs[FULL_OBS.nvel])
        evel = float(full_obs[FULL_OBS.evel])
        vel_up = (alt - prev_alt) / DT

        if has_prev and args.wind_est_alpha > 0.0 and not prev_action_saturated:
            pred_npos = prev_npos + prev_nvel * DT + 0.5 * prev_cmd_accel[0] * DT * DT
            pred_epos = prev_epos + prev_evel * DT + 0.5 * prev_cmd_accel[1] * DT * DT
            pred_alt = prev_alt + prev_vel_up * DT + 0.5 * prev_cmd_accel[2] * DT * DT
            pred_nvel = prev_nvel + prev_cmd_accel[0] * DT
            pred_evel = prev_evel + prev_cmd_accel[1] * DT
            pred_vel_up = prev_vel_up + prev_cmd_accel[2] * DT
            last_d_pos = (
                npos - pred_npos,
                epos - pred_epos,
                alt - pred_alt,
            )
            last_d_vel = (
                nvel - pred_nvel,
                evel - pred_evel,
                vel_up - pred_vel_up,
            )
            innovation = (
                compute_damped_disturbance_innovation(
                    d_pos=last_d_pos[0],
                    d_vel=last_d_vel[0],
                    horizon_s=wind_est_horizon_s,
                    coverage=wind_est_coverage,
                ),
                compute_damped_disturbance_innovation(
                    d_pos=last_d_pos[1],
                    d_vel=last_d_vel[1],
                    horizon_s=wind_est_horizon_s,
                    coverage=wind_est_coverage,
                ),
                compute_damped_disturbance_innovation(
                    d_pos=last_d_pos[2],
                    d_vel=last_d_vel[2],
                    horizon_s=wind_est_horizon_s,
                    coverage=wind_est_coverage,
                ),
            )
            innovation = (
                float(np.clip(innovation[0], -MAX_WIND_EST_ACCEL, MAX_WIND_EST_ACCEL)),
                float(np.clip(innovation[1], -MAX_WIND_EST_ACCEL, MAX_WIND_EST_ACCEL)),
                float(np.clip(innovation[2], -MAX_WIND_EST_ACCEL, MAX_WIND_EST_ACCEL)),
            )
            wind_est = update_wind_estimate(
                wind_est,
                residual_accel=innovation,
                alpha=args.wind_est_alpha,
            )

        raw_a_n, raw_a_e, raw_a_z = compute_predictive_accelerations(
            npos=npos,
            epos=epos,
            alt=alt,
            nvel=nvel,
            evel=evel,
            vel_up=vel_up,
            target_alt=target_alt,
            horizon_s=args.horizon * DT,
            coverage=args.coverage,
        )
        wind_bias = (wind_est.n, wind_est.e, wind_est.z)
        cmd_accel = (
            raw_a_n - wind_bias[0],
            raw_a_e - wind_bias[1],
            raw_a_z - wind_bias[2],
        )

        action = compute_predictive_action(
            full_obs,
            target_alt=target_alt,
            effects=effects,
            trim_action=trim_action,
            gravity=gravity,
            action_limit=args.action_limit,
            prev_alt=prev_alt,
            horizon_steps=args.horizon,
            coverage=args.coverage,
            wind_bias=wind_bias,
        )

        prev_npos = npos
        prev_epos = epos
        prev_nvel = nvel
        prev_evel = evel
        prev_vel_up = vel_up
        prev_cmd_accel = cmd_accel
        prev_action_saturated = bool(np.any(np.abs(action) >= args.action_limit - 1e-4))
        has_prev = True
        prev_alt = alt
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        max_pos_err = max(max_pos_err, math.hypot(npos, epos))
        max_alt_err = max(max_alt_err, abs(alt - target_alt))

        if args.render:
            env.render()

        if step % args.print_every == 0:
            print(
                f"{episode_prefix}step={step:5d}  npos={npos:+9.2f}  epos={epos:+9.2f}  "
                f"alt={alt:9.2f}  wind_est=({wind_est.n:+7.2f}, {wind_est.e:+7.2f}, {wind_est.z:+7.2f})  "
                f"d_pos=({last_d_pos[0]:+7.2f}, {last_d_pos[1]:+7.2f}, {last_d_pos[2]:+7.2f})  "
                f"d_vel=({last_d_vel[0]:+7.2f}, {last_d_vel[1]:+7.2f}, {last_d_vel[2]:+7.2f})  "
                f"action={np.round(action, 4).tolist()}"
            )

        if terminated or truncated:
            break

    print(
        f"{episode_prefix}Done: steps={steps}, reward={total_reward:.2f}, "
        f"max_pos_err={max_pos_err:.2f} ft, max_alt_err={max_alt_err:.2f} ft, "
        f"wind_est=({wind_est.n:+.2f}, {wind_est.e:+.2f}, {wind_est.z:+.2f}), terminated={terminated}"
    )
    return {
        "reward": total_reward,
        "max_pos_err": max_pos_err,
        "max_alt_err": max_alt_err,
        "steps": steps,
        "terminated": terminated,
    }
