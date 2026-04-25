"""PPO + VecNormalize inference for UI suggestions (load once, thread-safe for single process)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

# Lazy singleton (loaded on first use; avoids import time if only Play is opened)
_lock = threading.Lock()
_model = None
_venv = None


def _root() -> Path:
    return Path(__file__).resolve().parent


def get_predictor() -> tuple:
    """Load PPO + VecNormalize once. Raises FileNotFoundError if files missing."""
    global _model, _venv
    with _lock:
        if _model is not None:
            return _model, _venv

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from gym_env import FCOpenEnvGym

        mpath = _root() / "models" / "final_model.zip"
        vpath = _root() / "models" / "vecnormalize.pkl"
        if not mpath.is_file() or not vpath.is_file():
            raise FileNotFoundError(
                f"Missing {mpath} or {vpath}; run training to generate."
            )

        def _make() -> FCOpenEnvGym:
            return FCOpenEnvGym()

        venv = DummyVecEnv([_make])
        venv = VecNormalize.load(str(vpath), venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(str(mpath), env=venv)
        _model, _venv = model, venv
        return model, venv


def get_model_action(obs: Observation) -> int:
    """Map current Observation to training-sized vector, apply VecNorm, PPO predict."""
    from gym_env import FCOpenEnvGym

    model, venv = get_predictor()
    gym = FCOpenEnvGym()  # only for _vec(); inner env is unused for encoding
    raw = gym._vec(obs).astype(np.float32)
    # (1, 16) for vectorized model
    raw_b = np.expand_dims(raw, axis=0)
    obs_n = venv.normalize_obs(raw_b)
    action, _st = model.predict(obs_n, deterministic=True)
    a = np.asarray(action).reshape(-1)
    return int(a[0])
