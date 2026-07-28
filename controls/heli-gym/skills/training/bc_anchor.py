"""BC anchoring during PPO fine-tuning to limit drift from demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from skills.training.envs import N_FRAME_STACK, N_SINGLE_FRAME_OBS

SKILLS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMOS = SKILLS_ROOT.parent / "pid" / "results" / "demos.npz"


def load_demos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing demonstrations at {path}. "
            "Run: python -m pid.record_demonstrations"
        )
    data = np.load(path)
    obs = np.asarray(data["obs"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != N_SINGLE_FRAME_OBS:
        raise ValueError(
            f"Expected obs shape (N, {N_SINGLE_FRAME_OBS}), got {obs.shape}"
        )
    if actions.ndim != 2 or actions.shape[0] != obs.shape[0]:
        raise ValueError(f"Mismatched obs/actions shapes: {obs.shape} vs {actions.shape}")
    return obs, actions


def stack_observations(obs: np.ndarray) -> np.ndarray:
    """Tile single-frame obs to match VecFrameStack input size."""
    return np.tile(obs, (1, N_FRAME_STACK))


def resolve_demos_path(demos: str | Path) -> Path:
    path = Path(demos)
    if not path.is_absolute():
        path = SKILLS_ROOT.parent / path
    return path


def _predict_actions(model: PPO, obs_batch: torch.Tensor) -> torch.Tensor:
    features = model.policy.extract_features(obs_batch)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    return model.policy.action_net(latent_pi)


def restore_ppo_train(model: PPO) -> None:
    """Remove any patched train override so SB3 can save/load the model."""
    model.__dict__.pop("train", None)


class BcAnchorCallback(BaseCallback):
    """Run BC gradient steps after each PPO update via a temporary train() wrapper."""

    def __init__(
        self,
        vec_env: VecNormalize,
        bc_cfg: dict[str, Any],
        *,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.vec_env = vec_env
        self.bc_cfg = bc_cfg
        self._original_train = None
        self._optimizer: torch.optim.Adam | None = None
        self._obs_stacked: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._steps = 0
        self._batch_size = 0
        self._n_samples = 0

    def _init_callback(self) -> None:
        demos_path = resolve_demos_path(self.bc_cfg.get("demos", DEFAULT_DEMOS))
        obs, actions = load_demos(demos_path)
        self._obs_stacked = stack_observations(obs)
        self._actions = actions
        self._steps = int(self.bc_cfg.get("steps", 1))
        self._batch_size = int(self.bc_cfg.get("batch_size", 256))
        learning_rate = float(self.bc_cfg.get("learning_rate", 1.0e-4))
        self._n_samples = len(self._obs_stacked)

        self._optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=learning_rate)
        self._original_train = self.model.train
        self.model.train = self._train_with_bc_anchor  # type: ignore[method-assign]

        if self.verbose >= 1:
            print(
                f"BC anchor: {self._n_samples} demos from {demos_path}, "
                f"{self._steps} step(s)/update, batch_size={self._batch_size}, "
                f"lr={learning_rate}"
            )

    def _on_training_end(self) -> None:
        restore_ppo_train(self.model)

    def _on_step(self) -> bool:
        return True

    def _train_with_bc_anchor(self) -> None:
        assert self._original_train is not None
        self._original_train()
        if self._steps <= 0 or self._n_samples == 0:
            return
        assert self._optimizer is not None
        assert self._obs_stacked is not None
        assert self._actions is not None

        device = self.model.device
        actions_tensor = torch.as_tensor(self._actions, dtype=torch.float32, device=device)
        self.model.policy.set_training_mode(True)

        total_loss = 0.0
        for _ in range(self._steps):
            batch_idx = np.random.randint(
                0, self._n_samples, size=min(self._batch_size, self._n_samples)
            )
            obs_batch = self.vec_env.normalize_obs(self._obs_stacked[batch_idx])
            obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            act_batch = actions_tensor[batch_idx]

            pred_actions = _predict_actions(self.model, obs_tensor)
            loss = F.mse_loss(pred_actions, act_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += float(loss.item())

        self.logger.record("train/bc_anchor_loss", total_loss / self._steps)


def make_bc_anchor_callback(
    vec_env: VecNormalize,
    bc_cfg: dict[str, Any],
    *,
    verbose: int = 1,
) -> BcAnchorCallback:
    return BcAnchorCallback(vec_env, bc_cfg, verbose=verbose)
