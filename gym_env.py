"""Gymnasium wrapper for FCEnvEnvironment (neural PPO and integration tests)."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment import FCEnvEnvironment
from models import Action


def _encode_clue(clue: str) -> float:
    if clue == "HIDDEN":
        return 0.0
    h = 0.0
    for c in str(clue):
        h = (h * 0.13 + (ord(c) / 256.0)) % 1.0
    return float(h)


class FCOpenEnvGym(gym.Env):
    """16-dim Box observation, Discrete(4) action; reward and done from the environment."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 8) -> None:
        super().__init__()
        self._env = FCEnvEnvironment(max_steps=max_steps)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._max_steps = max_steps

    def _vec(self, obs) -> np.ndarray:
        v: list[float] = [
            obs.tokens / 100.0,
            min(1.0, obs.step_number / float(self._max_steps)),
            obs.low_remaining / 3.0,
            obs.high_remaining / 3.0,
            1.0 if obs.done else 0.0,
        ]
        for c in obs.revealed_clues:
            v.append(_encode_clue(c))
        while len(v) < 16:
            v.append(0.0)
        return np.asarray(v[:16], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random

            random.seed(int(seed))
            np.random.seed(int(seed))
        o = self._env.reset()
        return self._vec(o), {}

    def step(self, action):
        o = self._env.step(Action(action=int(action)))
        v = self._vec(o)
        term = bool(o.done)
        return v, float(o.reward), term, False, dict(getattr(o, "info", {}))
