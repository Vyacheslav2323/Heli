"""Predictive hover controller using kinematic acceleration + model inversion.

Control law:
    T = N * DT
    c = coverage in (0, 1]  (fraction of remaining error covered within T)
    a_n = 2 * (-c * npos - T * nvel) / T**2
    a_e = 2 * (-c * epos - T * evel) / T**2
    a_z = 2 * (c * (target_alt - alt) - T * vel_up) / T**2
    a_cmd = a_raw - wind_est   (per-axis disturbance compensation)

Coverage receding-horizon planning:
    Each step solves for error(T) = (1 - c) * error(0).
    Default c = 0.5 yields geometric D, D/2, D/4, ... with fixed T.
    Keep T large enough for stable attitude tracking (empirically N >= 20).

    pitch_des = clip(a_n / g, -MAX_TILT, MAX_TILT)
    roll_des  = clip(-a_e / g, -MAX_TILT, MAX_TILT)
    yaw_des   = 0

    AV = [roll_acc, pitch_acc, yaw_acc, a_z * mass]  (kinematic, horizon T)
    u  = pinv(B.T) @ AV   where B is control_effects_no_bias.csv

Run from heli-gym/:
    python -m pid_theory.predictive_hover
    python -m pid_theory.predictive_hover --render
    python -m pid_theory.predictive_hover --coverage 0.75 --horizon 25
    python -m pid_theory.predictive_hover --wind-est-alpha 0.1 --wind-spd 20 --wind-dir 90
    python -m pid_theory.predictive_hover --wind-spd 20 --wind-dir 90 --turb-lvl 3
    python -m pid_theory.predictive_hover --n-episodes 20 --init-pos-std 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pid.bootstrap import ensure_heli_gym_path
from pid.constants import CONTROL_EFFECTS_NO_BIAS_CSV
from pid.env_factory import make_stabilization_env
from pid.io.control_no_bias import load_full_control_effects
from pid.wind_config import add_wind_args, describe_wind
from pid_theory.predictive_control import (
    WindEstimate,
    apply_gaussian_spawn_perturbation,
    compute_accel_vector,
    compute_damped_disturbance_innovation,
    compute_predictive_accelerations,
    compute_predictive_action,
    update_wind_estimate,
)
from pid_theory.predictive_episode import _run_episode
from skills.training.common import ensure_heligym_env

ensure_heli_gym_path()

__all__ = [
    "WindEstimate",
    "apply_gaussian_spawn_perturbation",
    "compute_accel_vector",
    "compute_damped_disturbance_innovation",
    "compute_predictive_accelerations",
    "compute_predictive_action",
    "update_wind_estimate",
    "parse_args",
    "run",
    "main",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predictive hover: kinematic accel + full B.T inversion.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help=(
            "Prediction horizon N in steps (T = N * DT). Default 50. "
            "Lower values (<~15-20) can destabilize attitude due to high 1/T^2 gain."
        ),
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.5,
        help=(
            "Fraction of remaining position/altitude error to cover within T. "
            "Default 0.5 (half-distance). Higher is more aggressive (e.g. 0.75)."
        ),
    )
    parser.add_argument("--target-alt", type=float, default=None, help="Altitude setpoint (ft).")
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument(
        "--init-pos-std",
        type=float,
        default=20.0,
        help="Gaussian spawn perturbation std (ft) for N/E/alt. 0 disables.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of randomized episodes to run (seed, seed+1, ...).",
    )
    parser.add_argument(
        "--control-effects-csv",
        type=Path,
        default=CONTROL_EFFECTS_NO_BIAS_CSV,
        help="Control effects CSV (default: no-bias file).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--wind-est-alpha",
        type=float,
        default=0.1,
        help="EMA alpha for expected-vs-realized disturbance estimate; 0 disables compensation.",
    )
    parser.add_argument(
        "--wind-est-coverage",
        type=float,
        default=None,
        help="Coverage used by discrepancy damping. Defaults to --coverage when omitted.",
    )
    parser.add_argument(
        "--wind-est-horizon-scale",
        type=float,
        default=1.0,
        help="Scale factor for discrepancy damping horizon (T_w = scale * T).",
    )
    add_wind_args(parser)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1")
    if not (0.0 < args.coverage <= 1.0):
        raise SystemExit("--coverage must be in (0, 1]")
    if not (0.0 <= args.wind_est_alpha <= 1.0):
        raise SystemExit("--wind-est-alpha must be in [0, 1]")
    if args.wind_est_coverage is not None and not (0.0 < args.wind_est_coverage <= 1.0):
        raise SystemExit("--wind-est-coverage must be in (0, 1]")
    if args.wind_est_horizon_scale <= 0.0:
        raise SystemExit("--wind-est-horizon-scale must be > 0")
    if args.n_episodes < 1:
        raise SystemExit("--n-episodes must be >= 1")

    episode_summaries: list[dict[str, float | int | bool]] = []

    env = make_stabilization_env(
        action_limit=args.action_limit,
        max_time=args.max_time,
        init_attitude_range=0.0,
        target_alt=args.target_alt,
        render_enabled=args.render,
        wind_spd=args.wind_spd,
        wind_dir=args.wind_dir,
        turb_lvl=args.turb_lvl,
    )
    mass = float(env.heli_dyn.HELI["M"])
    gravity = float(env.heli_dyn.ENV["GRAV"])
    effects = load_full_control_effects(args.control_effects_csv, mass)
    print(f"Wind: {describe_wind(env)}")

    for episode in range(args.n_episodes):
        if args.render and env.renderer is not None and env.renderer.is_close():
            break
        episode_seed = args.seed + episode
        summary = _run_episode(
            args,
            env=env,
            effects=effects,
            gravity=gravity,
            seed=episode_seed,
            episode=episode,
        )
        episode_summaries.append(summary)

    if len(episode_summaries) > 1:
        rewards = [float(s["reward"]) for s in episode_summaries]
        max_pos_errs = [float(s["max_pos_err"]) for s in episode_summaries]
        max_alt_errs = [float(s["max_alt_err"]) for s in episode_summaries]
        terminated_count = sum(1 for s in episode_summaries if s["terminated"])
        n_done = len(episode_summaries)
        print(
            f"\nSummary over {n_done} episodes: "
            f"reward mean={np.mean(rewards):.2f} std={np.std(rewards):.2f}, "
            f"max_pos_err mean={np.mean(max_pos_errs):.2f} ft, "
            f"max_alt_err mean={np.mean(max_alt_errs):.2f} ft, "
            f"terminated={terminated_count}/{n_done}"
        )


def main() -> None:
    ensure_heligym_env()
    run(parse_args())


if __name__ == "__main__":
    main()
