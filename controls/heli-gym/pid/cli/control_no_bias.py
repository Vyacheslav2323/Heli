"""Estimate linear control effects on accelerations without a bias term.

Run from heli-gym/:
    python -m pid.control_effects_no_bias_estimation
    python -m pid.control_effects_no_bias_estimation --n-sweep 100 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pid.bootstrap import ensure_heli_gym_path

ensure_heli_gym_path()

from pid.constants import (  # noqa: E402
    ACCEL_NAMES,
    CONTROL_REGRESSORS,
    DEFAULT_EFFECT_SWEEP_LIMIT,
    DEFAULT_N_SWEEP,
    RESULTS_DIR,
)
from pid.env_factory import make_stabilization_env  # noqa: E402
from pid.estimation.ols import fit_ols_no_bias  # noqa: E402
from pid.estimation.sweep import collect_control_effects_data  # noqa: E402
from pid.io.control_no_bias import save_control_effects_no_bias_csv  # noqa: E402
from skills.training.common import ensure_heligym_env  # noqa: E402


def print_results(B, r2, *, n_samples: int) -> None:
    col_width = 20
    header = f"{'term':<12}" + "".join(f"{name:>{col_width}}" for name in ACCEL_NAMES)
    print(f"\nCollected {n_samples} samples ({len(CONTROL_REGRESSORS)} controls x sweep points)")
    print("\nControl Effect Matrix (rows=regressors, cols=accelerations, no bias):")
    print(header)
    print("-" * len(header))

    for regressor_name, row in zip(CONTROL_REGRESSORS, B):
        values = "".join(f"{value:>{col_width}.6f}" for value in row)
        print(f"{regressor_name:<12}{values}")

    print("\nR²:")
    for accel_name, value in zip(ACCEL_NAMES, r2):
        print(f"  {accel_name}: {value:.4f}")

    col_idx = CONTROL_REGRESSORS.index("col_t")
    vert_idx = ACCEL_NAMES.index("vertical_acceleration")
    k_col = float(B[col_idx, vert_idx])
    print("\nVertical force (direct dynamics acceleration, lb/unit):")
    print(f"  Delta F_up = {k_col:.6f} * d_col (+ other control terms)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate linear control effects on angular/vertical accelerations (no bias).",
    )
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument(
        "--effect-sweep-limit",
        type=float,
        default=DEFAULT_EFFECT_SWEEP_LIMIT,
    )
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--init-attitude-range", type=float, default=0.0)
    parser.add_argument("--n-sweep", type=int, default=DEFAULT_N_SWEEP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_DIR / "control_effects_no_bias.csv",
    )
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_heligym_env()

    env = make_stabilization_env(
        action_limit=args.action_limit,
        max_time=args.max_time,
        init_attitude_range=args.init_attitude_range,
    )

    print(
        f"Control effects estimation (no bias): n_sweep={args.n_sweep}, "
        f"action_limit=±{args.action_limit}, "
        f"effect_sweep_limit=±{args.effect_sweep_limit}, seed={args.seed}"
    )

    X, Y = collect_control_effects_data(
        env,
        n_sweep=args.n_sweep,
        action_limit=args.action_limit,
        effect_sweep_limit=args.effect_sweep_limit,
        seed=args.seed,
    )
    B, r2 = fit_ols_no_bias(X, Y)
    print_results(B, r2, n_samples=X.shape[0])

    if not args.no_csv:
        save_control_effects_no_bias_csv(B, r2, args.csv)
        print(f"\nSaved results to {args.csv.resolve()}")


if __name__ == "__main__":
    main()
