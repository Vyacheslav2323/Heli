"""Skill-specific helicopter environments."""

import numpy as np
from gymnasium import spaces
from heligym.envs.helicopter import DT, Heli

OBS_NAMES = [
    "power", "lonvel", "latvel", "dwnvel", "nvel", "evel", "desrate",
    "roll", "pitch", "yaw", "rollrate", "pitchrate", "yawrate",
    "npos", "epos", "alt", "gralt",
]

STAB_OBS_INDICES = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Index 15 = -xyz[2] (MSL-like); index 16 = alt_gr (AGL above terrain).
ALT_AGL_INDEX = 16
POSITION_OBS_INDICES = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, ALT_AGL_INDEX]
ACTION_NAMES = ["collective", "loncyclic", "latcyclic", "pedal"]
AGENT_ACTION_INDICES = [0, 1, 2, 3]
STAB_OBS_NAMES = [
    "lonvel", "latvel", "dwnvel",
    "roll", "pitch", "yaw",
    "rollrate", "pitchrate", "yawrate",
    "npos", "epos", "alt",
    "collective", "loncyclic", "latcyclic", "pedal",
]
N_BASE_OBS = len(STAB_OBS_INDICES)
N_LAST_ACTION_OBS = len(AGENT_ACTION_INDICES)
N_SINGLE_FRAME_OBS = N_BASE_OBS + N_LAST_ACTION_OBS
N_FRAME_STACK = 3
N_STACKED_OBS = N_SINGLE_FRAME_OBS * N_FRAME_STACK

DEFAULT_TRIM_COND = {
    "yaw": 0.0,
    "yaw_rate": 0.0,
    "ned_vel": [0.0, 0.0, 0.0],
    "gr_alt": 100.0,
    "xy": [0.0, 0.0],
    "psi_mr": 0.0,
    "psi_tr": 0.0,
}

REWARD_HYPER_KEYS = ("roll", "pitch", "yaw", "action", "pos", "alt")
POSITION_REWARD_HYPER_KEYS = ("pos", "alt")
DEFAULT_REWARD_HYPERS = {key: 0.0 for key in REWARD_HYPER_KEYS}
STABLE_BAND = 0.001


def normalize_reward_hypers(reward_hypers: dict[str, float] | None) -> dict[str, float]:
    """Merge user-provided reward hypers with defaults (all 0.0 → coefficient 5^0 = 1)."""
    merged = dict(DEFAULT_REWARD_HYPERS)
    if reward_hypers:
        merged.update(reward_hypers)
    return merged


class HeliStabilization(Heli):
    """Keep roll, pitch, and yaw near zero."""

    def __init__(
        self,
        heli_name: str = "aw109",
        render_enabled: bool = False,
        max_time: float | None = None,
        instability_limit: float = 4,
        instability_penalty: float = -10.0,
        survival_bonus: float = 20.0,
        landing_penalty: float = 0,
        trim_cond: dict | None = None,
        action_limit: float = 1,
        init_attitude_range: float = 0.0,
        reward_hypers: dict[str, float] | None = None,
    ):
        super().__init__(heli_name=heli_name, render_enabled=render_enabled)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (N_SINGLE_FRAME_OBS,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            -action_limit,
            action_limit,
            (len(AGENT_ACTION_INDICES),),
            dtype=np.float32,
        )
        self._full_action_space = spaces.Box(
            -action_limit,
            action_limit,
            (self.heli_dyn.n_act,),
            dtype=np.float32,
        )
        self.init_attitude_range = init_attitude_range
        self.instability_limit = instability_limit
        self.instability_penalty = instability_penalty
        self.survival_bonus = survival_bonus
        self.landing_penalty = landing_penalty
        self._reward_hypers = normalize_reward_hypers(reward_hypers)
        self._prev_agent_action = np.zeros(len(AGENT_ACTION_INDICES), dtype=np.float32)
        self._current_agent_action = np.zeros(len(AGENT_ACTION_INDICES), dtype=np.float32)
        self._prev_npos = 0.0
        self._prev_epos = 0.0
        self.set_max_time(max_time)
        self.set_trim_cond(trim_cond or DEFAULT_TRIM_COND)

    def _get_obs(self) -> np.ndarray:
        base = self.heli_dyn.observation[STAB_OBS_INDICES].astype(np.float32)
        last = self.last_action[AGENT_ACTION_INDICES].astype(np.float32)
        return np.concatenate([base, last])

    def _expand_action(self, agent_action: np.ndarray) -> np.ndarray:
        full_action = np.zeros(self.heli_dyn.n_act, dtype=np.float32)
        full_action[AGENT_ACTION_INDICES] = agent_action
        return full_action

    def reset(self, seed=None, options=None):
        self._prev_agent_action.fill(0.0)
        self._current_agent_action.fill(0.0)
        super().reset(seed=seed, options=options)
        if self.init_attitude_range > 0.0:
            attitude = self.np_random.uniform(
                -self.init_attitude_range,
                self.init_attitude_range,
                size=3,
            )
            self.heli_dyn.state["euler"][:] = attitude
            self.heli_dyn.state["pqr"][:] = 0.0
            self.heli_dyn.dynamics(self.heli_dyn.state, set_observation=True)
        self._prev_npos = float(self.heli_dyn.observation[13])
        self._prev_epos = float(self.heli_dyn.observation[14])
        return self._get_obs(), self._get_info()

    def step(self, actions):
        agent_action = np.asarray(actions, dtype=np.float32).reshape(-1)
        self._current_agent_action = agent_action.copy()
        full_action = self._expand_action(agent_action)
        saved_action_space = self.action_space
        self.action_space = self._full_action_space
        try:
            _, reward, done, truncated, info = super().step(full_action)
        finally:
            self.action_space = saved_action_space
        self._prev_agent_action = agent_action.copy()
        return self._get_obs(), reward, done, truncated, info

    def _is_unstable(self) -> bool:
        roll, pitch, yaw = self.heli_dyn.state["euler"]
        if not np.all(np.isfinite((roll, pitch, yaw))):
            return True
        return (
            abs(roll) > self.instability_limit
            or abs(pitch) > self.instability_limit
            or abs(yaw) > self.instability_limit
        )

    def _is_failed(self):
        if self._is_unstable():
            return True
        return super()._is_failed()

    def _calculate_reward(self):
        roll, pitch, yaw = self.heli_dyn.state["euler"]
        npos, epos = self.heli_dyn.observation[13:15]
        alt = float(self.heli_dyn.observation[15])
        n_dev = float(npos) - self._prev_npos
        e_dev = float(epos) - self._prev_epos
        alt_dev = float(alt) - self.trim_cond["gr_alt"]
        self._prev_npos = float(npos)
        self._prev_epos = float(epos)

        rh = self._reward_hypers
        reward = -(
            5 ** rh["roll"] * roll ** 2
            + 5 ** rh["pitch"] * pitch ** 2
            + 5 ** rh["yaw"] * yaw ** 2
            + 5 ** rh["action"] * float(np.sum(self._current_agent_action ** 2)) / 1000
            + 5 ** rh["pos"] * (n_dev ** 2 + e_dev ** 2) / 1000
            + 5 ** rh["alt"] * alt_dev ** 2 / 1000
        )
        stable = (
            abs(roll) < STABLE_BAND
            and abs(pitch) < STABLE_BAND
            and abs(yaw) < STABLE_BAND
        )
        if self._is_failed():
            reward += self.instability_penalty
        elif self._is_time_up():
            reward += self.survival_bonus
        return reward, stable


class HeliPosition(HeliStabilization):
    """Hold position near origin with absolute position-deviation reward."""

    def __init__(
        self,
        heli_name: str = "aw109",
        render_enabled: bool = False,
        max_time: float | None = None,
        instability_limit: float = 4,
        instability_penalty: float = -10.0,
        survival_bonus: float = 20.0,
        landing_penalty: float = 0,
        trim_cond: dict | None = None,
        action_limit: float = 1,
        init_attitude_range: float = 0.0,
        init_pos_range: float = 50.0,
        pos_fail_limit: float = 200.0,
        reward_hypers: dict[str, float] | None = None,
    ):
        super().__init__(
            heli_name=heli_name,
            render_enabled=render_enabled,
            max_time=max_time,
            instability_limit=instability_limit,
            instability_penalty=instability_penalty,
            survival_bonus=survival_bonus,
            landing_penalty=landing_penalty,
            trim_cond=trim_cond,
            action_limit=action_limit,
            init_attitude_range=init_attitude_range,
            reward_hypers=reward_hypers,
        )
        self.init_pos_range = init_pos_range
        self.pos_fail_limit = pos_fail_limit

    def _is_failed(self):
        if super()._is_failed():
            return True
        npos = float(self.heli_dyn.observation[13])
        epos = float(self.heli_dyn.observation[14])
        return (npos ** 2 + epos ** 2) > self.pos_fail_limit ** 2

    def _get_obs(self) -> np.ndarray:
        """Use AGL (alt_gr) instead of MSL altitude in the observation."""
        base = self.heli_dyn.observation[POSITION_OBS_INDICES].astype(np.float32)
        last = self.last_action[AGENT_ACTION_INDICES].astype(np.float32)
        return np.concatenate([base, last])

    def _alt_agl(self) -> float:
        return float(self.heli_dyn.observation[ALT_AGL_INDEX])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.init_pos_range > 0.0:
            n_offset = float(self.np_random.uniform(-self.init_pos_range, self.init_pos_range))
            e_offset = float(self.np_random.uniform(-self.init_pos_range, self.init_pos_range))
            self.heli_dyn.state["xyz"][0] += n_offset
            self.heli_dyn.state["xyz"][1] += e_offset
            self.heli_dyn.dynamics(self.heli_dyn.state, set_observation=True)
            self._prev_npos = float(self.heli_dyn.observation[13])
            self._prev_epos = float(self.heli_dyn.observation[14])
            obs = self._get_obs()
        return obs, info

    def _calculate_reward(self):
        npos = float(self.heli_dyn.observation[13])
        epos = float(self.heli_dyn.observation[14])
        alt_dev = self._alt_agl() - self.trim_cond["gr_alt"]

        rh = self._reward_hypers
        reward = -(
            5 ** (-1.1) * (npos ** 2 + epos ** 2) / 1000.0
            + 5 ** (0.33) * alt_dev ** 2 / 1000.0
        )
        if self._is_failed():
            reward += self.instability_penalty
        elif self._is_time_up():
            reward += self.survival_bonus
        return reward, False
