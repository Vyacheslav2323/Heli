"""Shared utilities for skill training."""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
import yaml
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize

from skills.training.bc_anchor import make_bc_anchor_callback, restore_ppo_train

from skills.training.envs import HeliPosition, HeliStabilization, N_FRAME_STACK, STABLE_BAND

SKILLS_ROOT = Path(__file__).resolve().parents[1]
PARAMS_ROOT = SKILLS_ROOT / "params"
ATTEMPT_PATTERN = re.compile(r"attempt_(\d+)\.yaml$")

_ACTIVATION_FNS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


def ensure_heligym_env() -> None:
    """Set HELIGYM_RESOURCE_DIR when training headless (no renderer import)."""
    if "HELIGYM_RESOURCE_DIR" in os.environ:
        return

    renderer_dir = Path(__file__).resolve().parents[2] / "heligym" / "envs" / "renderer"
    resource_dir = renderer_dir / "resources"
    libs_dir = renderer_dir / "libs"
    python_dir = renderer_dir / "bin"

    os.environ["HELIGYM_RESOURCE_DIR"] = str(resource_dir)
    os.environ["HELIGYM_LIBS_DIR"] = str(libs_dir)
    os.environ["HELIGYM_PYTHON_DIR"] = str(python_dir)

    if sys.platform == "win32":
        os.environ["PATH"] += os.pathsep + str(libs_dir)
        os.environ["PATH"] += os.pathsep + str(python_dir)
        os.environ["PATH"] += os.pathsep + str(resource_dir)


def attempt_name(attempt: int) -> str:
    return f"attempt_{attempt:03d}"


def params_path(skill: str, attempt: int) -> Path:
    return PARAMS_ROOT / skill / f"{attempt_name(attempt)}.yaml"


def checkpoint_dir(skill: str, attempt: int) -> Path:
    return PARAMS_ROOT / skill / "checkpoints" / attempt_name(attempt)


def log_dir(skill: str, attempt: int) -> Path:
    return PARAMS_ROOT / skill / "logs" / attempt_name(attempt)


def next_attempt_number(skill: str) -> int:
    skill_dir = PARAMS_ROOT / skill
    if not skill_dir.exists():
        return 1

    attempts = []
    for path in skill_dir.glob("attempt_*.yaml"):
        match = ATTEMPT_PATTERN.match(path.name)
        if match:
            attempts.append(int(match.group(1)))

    return max(attempts, default=0) + 1


def latest_attempt_number(skill: str) -> int:
    """Return the highest existing attempt number for a skill."""
    latest = next_attempt_number(skill) - 1
    if latest < 1:
        raise FileNotFoundError(f"No attempts found for skill '{skill}'.")
    return latest


def load_params(skill: str, attempt: int) -> dict[str, Any]:
    path = params_path(skill, attempt)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_policy_kwargs(ppo_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build Stable Baselines3 policy_kwargs from the ppo.policy YAML section."""
    policy_cfg = ppo_cfg.get("policy")
    if not policy_cfg:
        return None

    policy_kwargs: dict[str, Any] = {}

    net_arch = policy_cfg.get("net_arch")
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    activation_fn = policy_cfg.get("activation_fn")
    if activation_fn is not None:
        key = activation_fn.lower()
        if key not in _ACTIVATION_FNS:
            valid = ", ".join(sorted(_ACTIVATION_FNS))
            raise ValueError(f"Unknown activation_fn '{activation_fn}'. Expected one of: {valid}")
        policy_kwargs["activation_fn"] = _ACTIVATION_FNS[key]

    for key in ("log_std_init", "ortho_init"):
        if key in policy_cfg:
            policy_kwargs[key] = policy_cfg[key]

    return policy_kwargs or None


def build_ppo_kwargs(ppo_cfg: dict[str, Any], *, tensorboard_log: str | None = None) -> dict[str, Any]:
    """Build Stable Baselines3 PPO constructor kwargs from YAML config."""
    ppo_kwargs: dict[str, Any] = {
        "learning_rate": ppo_cfg["learning_rate"],
        "n_steps": ppo_cfg["n_steps"],
        "batch_size": ppo_cfg["batch_size"],
        "n_epochs": ppo_cfg["n_epochs"],
        "gamma": ppo_cfg["gamma"],
        "gae_lambda": ppo_cfg["gae_lambda"],
        "clip_range": ppo_cfg["clip_range"],
        "ent_coef": ppo_cfg["ent_coef"],
        "verbose": ppo_cfg.get("verbose", 1),
    }
    if tensorboard_log is not None:
        ppo_kwargs["tensorboard_log"] = tensorboard_log

    for key in ("target_kl", "vf_coef", "max_grad_norm"):
        if key in ppo_cfg and ppo_cfg[key] is not None:
            ppo_kwargs[key] = ppo_cfg[key]

    policy_kwargs = build_policy_kwargs(ppo_cfg)
    if policy_kwargs is not None:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    return ppo_kwargs


def apply_ppo_runtime_hyperparams(model: PPO, ppo_cfg: dict[str, Any]) -> None:
    """Apply YAML PPO hyperparameters to a loaded checkpoint."""
    if "learning_rate" in ppo_cfg:
        model.learning_rate = ppo_cfg["learning_rate"]
        model.lr_schedule = FloatSchedule(ppo_cfg["learning_rate"])
    if "ent_coef" in ppo_cfg:
        model.ent_coef = ppo_cfg["ent_coef"]
    if "clip_range" in ppo_cfg:
        model.clip_range = FloatSchedule(ppo_cfg["clip_range"])
    # None disables SB3 early stop on approx KL (omit or set null in YAML).
    model.target_kl = ppo_cfg.get("target_kl")

    for key in ("n_epochs", "batch_size", "gamma", "gae_lambda", "vf_coef", "max_grad_norm"):
        if key in ppo_cfg and ppo_cfg[key] is not None:
            setattr(model, key, ppo_cfg[key])


def save_params(skill: str, attempt: int, data: dict[str, Any]) -> Path:
    path = params_path(skill, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return path


def default_params(skill: str, attempt: int, parent_attempt: int | None = None) -> dict[str, Any]:
    ckpt_dir = checkpoint_dir(skill, attempt)
    params: dict[str, Any] = {
        "skill": skill,
        "attempt": attempt,
        "parent_attempt": parent_attempt,
        "env": {
            "heli_name": "aw109",
            "render_enabled": False,
            "max_time": 40.0,
            "trim_cond": {
                "yaw": 0.0,
                "yaw_rate": 0.0,
                "ned_vel": [0.0, 0.0, 0.0],
                "gr_alt": 100.0,
                "xy": [0.0, 0.0],
                "psi_mr": 0.0,
                "psi_tr": 0.0,
            },
            "action_limit": 10,
            "init_attitude_range": 0.0,
        },
        "training": {
            "algorithm": "PPO",
            "total_timesteps": 10_000,
            "n_envs": 8,
            "max_episode_steps": 200,
            "n_frame_stack": N_FRAME_STACK,
            "init_from": None,
        },
        "ppo": {
            "learning_rate": 1.0e-3,
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 5,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_range": 0.01,
            "ent_coef": 0.01,
            "verbose": 1,
            "policy": {
                "net_arch": {"pi": [32, 32], "vf": [32, 32]},
                "activation_fn": "relu",
            },
        },
        "checkpoints": {
            "dir": str(ckpt_dir.relative_to(SKILLS_ROOT.parent)).replace("\\", "/"),
        },
    }
    if skill == "position":
        params["env"]["env_class"] = "position"
        params["env"]["action_limit"] = 1.0
        params["env"]["init_pos_range"] = 50.0
        params["training"]["total_timesteps"] = 150_000
        params["training"]["init_from"] = "skills/params/position/checkpoints/bc"
        params["training"]["bc_anchor"] = {
            "demos": "pid/results/demos.npz",
            "steps": 1,
            "batch_size": 256,
            "learning_rate": 1.0e-4,
        }
        params["ppo"]["learning_rate"] = 1.0e-3
        params["ppo"]["clip_range"] = 0.01
        params["ppo"]["ent_coef"] = 0.01
        params["ppo"]["policy"]["log_std_init"] = -2.0
    return params


def resolve_init_checkpoint(skill: str, init_from: str | int | None) -> tuple[Path, Path] | None:
    if init_from is None:
        return None

    if isinstance(init_from, int):
        ckpt = checkpoint_dir(skill, init_from)
    else:
        ckpt = Path(init_from)
        if not ckpt.is_absolute():
            ckpt = SKILLS_ROOT.parent / ckpt

    model_path = ckpt / "model.zip"
    vec_path = ckpt / "vecnormalize.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing vecnormalize checkpoint: {vec_path}")

    return model_path, vec_path


def make_skill_env(params: dict[str, Any]):
    env_cfg = params["env"]
    env_class = env_cfg.get("env_class", "stabilization")
    common_kwargs = {
        "heli_name": env_cfg.get("heli_name", "aw109"),
        "render_enabled": env_cfg.get("render_enabled", False),
        "max_time": env_cfg.get("max_time"),
        "trim_cond": env_cfg.get("trim_cond"),
        "action_limit": env_cfg.get("action_limit", 0.05),
        "init_attitude_range": env_cfg.get("init_attitude_range", 0.0),
        "instability_limit": env_cfg.get("instability_limit", 4.0),
        "instability_penalty": env_cfg.get("instability_penalty", -10.0),
        "survival_bonus": env_cfg.get("survival_bonus", 20.0),
        "landing_penalty": env_cfg.get("landing_penalty", 0.0),
        "reward_hypers": env_cfg.get("reward_hypers"),
    }
    if env_class == "position":
        return HeliPosition(
            **common_kwargs,
            init_pos_range=env_cfg.get("init_pos_range", 50.0),
            pos_fail_limit=env_cfg.get("pos_fail_limit", 200.0),
        )
    return HeliStabilization(**common_kwargs)


def unwrap_heli_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def theoretical_max_reward(env: HeliStabilization, max_steps: int) -> dict[str, float]:
    """Episode reward if every step has zero tracking error."""
    per_step = 0.0
    return {
        "per_step": per_step,
        "episode": per_step * max_steps + env.survival_bonus,
    }


_ERROR_KEYS = ("roll", "pitch", "yaw", "action", "pos", "alt")
_POSITION_ERROR_KEYS = ("npos", "epos", "alt")


def _error_keys_for_skill(skill: str, params: dict[str, Any]) -> tuple[str, ...]:
    if skill == "position" or params.get("env", {}).get("env_class") == "position":
        return _POSITION_ERROR_KEYS
    return _ERROR_KEYS


def _step_squared_errors(
    heli_env: HeliStabilization,
    prev_npos: float,
    prev_epos: float,
) -> dict[str, float]:
    roll, pitch, yaw = heli_env.heli_dyn.state["euler"]
    npos, epos = heli_env.heli_dyn.observation[13:15]
    action = heli_env._current_agent_action

    if isinstance(heli_env, HeliPosition):
        alt_dev = heli_env._alt_agl() - heli_env.trim_cond["gr_alt"]
        return {
            "npos": float(npos ** 2),
            "epos": float(epos ** 2),
            "alt": float(alt_dev ** 2) / 1000.0,
        }

    alt = float(heli_env.heli_dyn.observation[15])
    n_dev = float(npos) - prev_npos
    e_dev = float(epos) - prev_epos
    alt_dev = alt - heli_env.trim_cond["gr_alt"]
    return {
        "roll": float(roll ** 2),
        "pitch": float(pitch ** 2),
        "yaw": float(yaw ** 2),
        "action": float(np.sum(action ** 2)) / 1000.0,
        "pos": float(n_dev ** 2 + e_dev ** 2) / 1000.0,
        "alt": float(alt_dev ** 2) / 1000.0,
    }


def _make_eval_vec_env(params: dict[str, Any], vec_path: Path) -> VecNormalize:
    max_episode_steps = params["training"].get("max_episode_steps", 2000)
    eval_params = dict(params)
    eval_params["env"] = dict(params["env"])
    eval_params["env"]["render_enabled"] = False

    def make_env():
        return TimeLimit(make_skill_env(eval_params), max_episode_steps=max_episode_steps)

    vec_env = DummyVecEnv([make_env])
    vec_env = apply_frame_stack(vec_env, eval_params)
    vec_env = VecNormalize.load(str(vec_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _save_drawdown_plot(
    ckpt_path: Path,
    series_names: tuple[str, ...],
    series_trajs: tuple[list[list[float]], ...],
    *,
    ylabel: str,
    reference_line: float | None,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping eval_drawdown.png")
        return

    max_len = max((len(t) for trajs in series_trajs for t in trajs), default=0)
    if max_len == 0:
        return

    def to_array(trajs: list[list[float]]) -> np.ndarray:
        arr = np.full((len(trajs), max_len), np.nan, dtype=np.float64)
        for i, traj in enumerate(trajs):
            arr[i, : len(traj)] = traj
        return arr

    fig, axes = plt.subplots(len(series_names), 1, figsize=(10, 8), sharex=True)
    if len(series_names) == 1:
        axes = [axes]
    x = np.arange(max_len)
    for ax, name, trajs in zip(axes, series_names, series_trajs, strict=True):
        arr = to_array(trajs)
        median = np.nanmedian(arr, axis=0)
        p10 = np.nanpercentile(arr, 10, axis=0)
        p90 = np.nanpercentile(arr, 90, axis=0)
        ax.plot(x, median, label="median")
        ax.fill_between(x, p10, p90, alpha=0.25, label="10th-90th pct")
        if reference_line is not None:
            ax.axhline(
                reference_line,
                color="red",
                linestyle="--",
                linewidth=1,
                label="target",
            )
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = ckpt_path / "eval_drawdown.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved drawdown plot: {out_path}")


def evaluate_skill(
    model_path: Path,
    vec_path: Path,
    params: dict[str, Any],
    ckpt_path: Path | None = None,
    *,
    skill: str,
    attempt: int,
    n_episodes: int = 5,
    print_report: bool = True,
) -> dict[str, Any]:
    """Run deterministic rollouts and return reward / error metrics."""
    max_episode_steps = params["training"].get("max_episode_steps", 2000)
    ref_env = make_skill_env(params)
    theory = theoretical_max_reward(ref_env, max_episode_steps)

    vec_env = _make_eval_vec_env(params, vec_path)
    model = PPO.load(str(model_path), env=vec_env)
    heli_env = unwrap_heli_env(vec_env.envs[0])

    is_position = skill == "position" or params.get("env", {}).get("env_class") == "position"
    error_keys = _error_keys_for_skill(skill, params)

    episode_rewards: list[float] = []
    sq_error_sums = {key: 0.0 for key in error_keys}
    total_steps = 0
    drawdown_trajs: tuple[list[list[float]], list[list[float]], list[list[float]]]

    if is_position:
        npos_trajs: list[list[float]] = []
        epos_trajs: list[list[float]] = []
        alt_trajs: list[list[float]] = []
        drawdown_trajs = (npos_trajs, epos_trajs, alt_trajs)
    else:
        roll_trajs: list[list[float]] = []
        pitch_trajs: list[list[float]] = []
        yaw_trajs: list[list[float]] = []
        drawdown_trajs = (roll_trajs, pitch_trajs, yaw_trajs)

    for _ in range(n_episodes):
        obs = vec_env.reset()
        ep_reward = 0.0
        prev_npos = float(heli_env.heli_dyn.observation[13])
        prev_epos = float(heli_env.heli_dyn.observation[14])
        if is_position:
            ep_npos: list[float] = []
            ep_epos: list[float] = []
            ep_alt: list[float] = []
        else:
            ep_roll: list[float] = []
            ep_pitch: list[float] = []
            ep_yaw: list[float] = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = vec_env.step(action)
            if is_position:
                npos, epos = heli_env.heli_dyn.observation[13:15]
                alt_dev = (
                    heli_env._alt_agl() - heli_env.trim_cond["gr_alt"]
                    if isinstance(heli_env, HeliPosition)
                    else float(heli_env.heli_dyn.observation[15]) - heli_env.trim_cond["gr_alt"]
                )
                ep_npos.append(abs(float(npos)))
                ep_epos.append(abs(float(epos)))
                ep_alt.append(abs(float(alt_dev)))
            else:
                roll, pitch, yaw = heli_env.heli_dyn.state["euler"]
                ep_roll.append(abs(float(roll)))
                ep_pitch.append(abs(float(pitch)))
                ep_yaw.append(abs(float(yaw)))

            errors = _step_squared_errors(heli_env, prev_npos, prev_epos)
            for key, value in errors.items():
                sq_error_sums[key] += value
            prev_npos = float(heli_env.heli_dyn.observation[13])
            prev_epos = float(heli_env.heli_dyn.observation[14])

            ep_reward += float(reward[0])
            total_steps += 1
            if done[0]:
                break

        episode_rewards.append(ep_reward)
        if is_position:
            npos_trajs.append(ep_npos)
            epos_trajs.append(ep_epos)
            alt_trajs.append(ep_alt)
        else:
            roll_trajs.append(ep_roll)
            pitch_trajs.append(ep_pitch)
            yaw_trajs.append(ep_yaw)

    vec_env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    pct_of_max = 100.0 * mean_reward / theory["episode"] if theory["episode"] else 0.0
    mean_squared_error = {
        key: sq_error_sums[key] / total_steps if total_steps else 0.0
        for key in error_keys
    }
    total_mean_squared_error = float(sum(mean_squared_error.values()))

    if print_report:
        print(f"\n=== {skill.title()} Eval ({attempt_name(attempt)}) ===")
        print(f"Episodes: {n_episodes}   Max steps/ep: {max_episode_steps}")
        print()
        print(f"Theoretical max reward : {theory['episode']:.1f}  (per step: {theory['per_step']:.1f})")
        print()
        print(f"{'':26} Mean ± Std")
        print(
            f"  Realized reward        : {mean_reward:7.1f} ± {std_reward:.1f}   "
            f"({pct_of_max:5.1f}% of max)"
        )
        print()
        print("Mean squared error:")
        for key in error_keys:
            print(f"  {key:23}: {mean_squared_error[key]:.6f}")
        print(f"  {'total':23}: {total_mean_squared_error:.6f}")

        if ckpt_path is not None:
            if is_position:
                _save_drawdown_plot(
                    ckpt_path,
                    ("N offset", "E offset", "Alt deviation"),
                    drawdown_trajs,
                    ylabel="abs deviation (ft)",
                    reference_line=0.0,
                    title="Position drawdown (abs N/E/alt error vs timestep)",
                )
            else:
                _save_drawdown_plot(
                    ckpt_path,
                    ("Roll", "Pitch", "Yaw"),
                    drawdown_trajs,
                    ylabel="abs angle (rad)",
                    reference_line=STABLE_BAND,
                    title="Attitude drawdown (abs angle vs timestep)",
                )

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "pct_of_max": pct_of_max,
        "theoretical_episode_reward": theory["episode"],
        "theoretical_per_step_reward": theory["per_step"],
        "mean_squared_error": mean_squared_error,
        "total_mean_squared_error": total_mean_squared_error,
        "total_steps": total_steps,
        "n_episodes": n_episodes,
    }


def evaluate_after_training(
    model_path: Path,
    vec_path: Path,
    params: dict[str, Any],
    ckpt_path: Path,
    *,
    skill: str,
    attempt: int,
    n_episodes: int = 5,
) -> dict[str, Any]:
    """Run deterministic rollouts and print theoretical vs realized reward breakdown."""
    return evaluate_skill(
        model_path,
        vec_path,
        params,
        ckpt_path,
        skill=skill,
        attempt=attempt,
        n_episodes=n_episodes,
        print_report=True,
    )


def apply_frame_stack(vec_env: VecEnv, params: dict[str, Any]) -> VecEnv:
    n_stack = params["training"].get("n_frame_stack", 1)
    if n_stack > 1:
        return VecFrameStack(vec_env, n_stack=n_stack)
    return vec_env


def apply_vec_wrappers(
    vec_env: VecEnv,
    params: dict[str, Any],
    *,
    norm_obs: bool = True,
    norm_reward: bool = True,
) -> VecNormalize:
    """Wrap an already frame-stacked vec env with observation/reward normalization."""
    return VecNormalize(vec_env, norm_obs=norm_obs, norm_reward=norm_reward)


def make_base_vec_env(params: dict[str, Any]) -> VecEnv:
    """Frame-stacked env with no normalization wrapper."""
    training_cfg = params["training"]
    n_envs = training_cfg.get("n_envs", 4)
    max_episode_steps = training_cfg.get("max_episode_steps", 2000)

    env_factory = partial(make_skill_env, params)
    vec_env = make_vec_env(
        env_factory,
        n_envs=n_envs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": max_episode_steps},
    )
    return apply_frame_stack(vec_env, params)


def make_training_env(params: dict[str, Any]) -> VecNormalize:
    return apply_vec_wrappers(make_base_vec_env(params), params)


def train_skill(
    skill: str,
    attempt: int,
    params: dict[str, Any],
    *,
    print_eval: bool = True,
    n_eval_episodes: int = 5,
) -> dict[str, Any]:
    ensure_heligym_env()

    training_cfg = params["training"]
    ppo_cfg = params["ppo"]
    ckpt_path = checkpoint_dir(skill, attempt)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    tb_path = log_dir(skill, attempt)
    tb_path.mkdir(parents=True, exist_ok=True)

    init_paths = resolve_init_checkpoint(skill, training_cfg.get("init_from"))

    if init_paths is not None:
        model_path, vec_path = init_paths
        base_env = make_base_vec_env(params)
        env = VecNormalize.load(str(vec_path), base_env)
        env.training = True
        env.norm_reward = True
        model = PPO.load(str(model_path), env=env)
        model.tensorboard_log = str(tb_path)
        apply_ppo_runtime_hyperparams(model, ppo_cfg)
    else:
        env = make_training_env(params)
        model = PPO("MlpPolicy", env, **build_ppo_kwargs(ppo_cfg, tensorboard_log=str(tb_path)))

    total_timesteps = training_cfg["total_timesteps"]
    n_envs = training_cfg.get("n_envs", 4)
    verbose = ppo_cfg.get("verbose", 1)
    progress_bar = training_cfg.get("progress_bar", True)
    model.verbose = verbose

    bc_cfg = training_cfg.get("bc_anchor")
    callbacks = make_bc_anchor_callback(env, bc_cfg, verbose=verbose) if bc_cfg else None

    print(
        f"PPO training: {total_timesteps:,} timesteps, "
        f"{n_envs} envs, verbose={verbose}, progress_bar={progress_bar}"
    )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            callback=callbacks,
        )
    finally:
        restore_ppo_train(model)

    model_path = ckpt_path / "model.zip"
    vec_path = ckpt_path / "vecnormalize.pkl"
    model.save(str(model_path))
    env.save(str(vec_path))

    eval_metrics = evaluate_skill(
        model_path,
        vec_path,
        params,
        ckpt_path,
        skill=skill,
        attempt=attempt,
        n_episodes=n_eval_episodes,
        print_report=print_eval,
    )

    env.close()

    rel_ckpt = ckpt_path.relative_to(SKILLS_ROOT.parent).as_posix()
    params["checkpoints"] = {
        "dir": rel_ckpt,
        "model": f"{rel_ckpt}/model.zip",
        "vecnormalize": f"{rel_ckpt}/vecnormalize.pkl",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": total_timesteps,
    }
    save_params(skill, attempt, params)
    return {"params": params, **eval_metrics}
