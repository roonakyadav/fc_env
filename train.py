import csv
import json
import os
import random
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from environment import FCEnvEnvironment
from models import Action

ARTIFACT_DIR = "artifacts"
METRICS_CSV = os.path.join(ARTIFACT_DIR, "training_metrics.csv")
REWARD_PLOT = os.path.join(ARTIFACT_DIR, "reward_curve.png")
WINRATE_PLOT = os.path.join(ARTIFACT_DIR, "win_rate_vs_random.png")
QTABLE_PATH = os.path.join(ARTIFACT_DIR, "q_table.json")
EVAL_PATH = os.path.join(ARTIFACT_DIR, "evaluation.json")
# Override for fair multi-run comparisons, e.g. FC_EVAL_OUTPUT=artifacts/eval_300k.json
FC_EVAL_OUTPUT = os.getenv("FC_EVAL_OUTPUT", "").strip()
EVAL_OUTPUT_PATH = FC_EVAL_OUTPUT if FC_EVAL_OUTPUT else EVAL_PATH
SB3_MODEL_PATH = os.path.join("models", "final_model.zip")
BEST_MODEL_DIR = os.path.join("models", "best")
VEC_NORMALIZE_PATH = os.path.join("models", "vecnormalize.pkl")
SEED = 42

EPISODES = 900
EVAL_EPISODES = 300
PPO_TIMESTEPS = int(os.getenv("FC_PPO_TIMESTEPS", "800000"))
PPO_N_EVAL = 20


def run_ppo_and_save_sb3_model(timesteps: int = PPO_TIMESTEPS) -> dict[str, Any]:
    """
    PPO (Stable-Baselines3) with VecNormalize + EvalCallback.
    Saves: models/final_model.zip, models/vecnormalize.pkl, and best in models/best/
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from gym_env import FCOpenEnvGym

    set_random_seed(SEED, using_cuda=False)

    def _make() -> FCOpenEnvGym:
        return FCOpenEnvGym()

    train_env = DummyVecEnv([_make])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )
    # Eval env: same normalizer object references (updated during training).
    eval_env = DummyVecEnv([_make])
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path="./tensorboard_logs/eval",
        eval_freq=10000,
        n_eval_episodes=PPO_N_EVAL,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device="cpu",
        seed=SEED,
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name="PPO")

    os.makedirs("models", exist_ok=True)
    base = os.path.join("models", "final_model")
    model.save(base)
    train_env.save(VEC_NORMALIZE_PATH)

    return {
        "sb3_model_path": base + ".zip",
        "vecnorm_path": VEC_NORMALIZE_PATH,
        "best_dir": BEST_MODEL_DIR,
    }


def state_key(obs) -> tuple[int, int, int, int]:
    visible = sum(1 for clue in obs.revealed_clues if clue != "HIDDEN")
    tokens_bucket = min(10, obs.tokens // 10)
    return (visible, tokens_bucket, obs.low_remaining, obs.high_remaining)


def random_policy_action() -> int:
    return random.randint(0, 3)


def q_policy_action(obs, q_table: dict[tuple, float]) -> int:
    key = state_key(obs)
    return max(range(4), key=lambda a: q_table.get((key, a), 0.0))


def run_policy_episode(
    env: FCEnvEnvironment, policy_fn
) -> tuple[float, bool, int, int]:
    obs = env.reset()
    total_reward = 0.0
    n_steps = 0
    done = False
    while not done:
        action = policy_fn(obs)
        obs = env.step(Action(action=action))
        total_reward += obs.reward
        n_steps += 1
        done = obs.done
    tokens_used = max(0, 100 - int(obs.tokens))
    return total_reward, total_reward > 0.0, tokens_used, n_steps


def evaluate_policy(
    env: FCEnvEnvironment, policy_fn, episodes: int
) -> tuple[float, float, float, float]:
    rewards: list[float] = []
    tokens_used: list[int] = []
    step_counts: list[int] = []
    wins = 0
    for _ in range(episodes):
        reward, won, tok, n_st = run_policy_episode(env, policy_fn)
        rewards.append(reward)
        tokens_used.append(tok)
        step_counts.append(n_st)
        wins += int(won)
    n = float(episodes)
    return (
        sum(rewards) / n,
        wins / n,
        (sum(tokens_used) / n),
        (sum(step_counts) / n),
    )


def _save_q_table(q: dict[tuple, float]) -> None:
    rows = [
        {"state": list(state), "action": action, "value": value}
        for (state, action), value in q.items()
    ]
    with open(QTABLE_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def load_q_table() -> dict[tuple, float]:
    with open(QTABLE_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
    q: dict = {}
    for row in rows:
        q[(tuple(row["state"]), int(row["action"]))] = float(row["value"])
    return q


def evaluate_ppo_vec(
    model_path: str,
    vec_path: str,
    episodes: int = EVAL_EPISODES,
    *,
    seed: int = SEED,
    deterministic: bool = True,
) -> dict[str, float]:
    """
    Evaluate saved PPO + VecNormalize on raw env returns (norm_reward off for reported totals).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from gym_env import FCOpenEnvGym

    set_random_seed(seed, using_cuda=False)

    def _make() -> FCOpenEnvGym:
        return FCOpenEnvGym()

    venv = DummyVecEnv([_make])
    venv = VecNormalize.load(vec_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(model_path, env=venv)
    r_all: list[float] = []
    wins: list[int] = []
    tok: list[int] = []
    steps: list[int] = []
    gym0 = venv.venv.envs[0]

    for ep in range(episodes):
        # Same per-episode seeds across runs so win_rate / reward comparisons are not eval-luck.
        venv.seed(seed + ep)
        r_out = venv.reset()
        if isinstance(r_out, (tuple, list)) and r_out is not None:
            obs0 = r_out[0]
        else:
            obs0 = r_out
        obs = np.asarray(obs0, dtype=np.float32)
        total = 0.0
        stc = 0
        while True:
            a, _ = model.predict(obs, deterministic=deterministic)
            a = np.asarray(a).reshape(-1)
            step = venv.step(a)
            if len(step) == 5:
                o, rews, don, _, _ = step
            else:
                o, rews, don, _ = step
            o = o if not isinstance(o, tuple) else o[0]
            rews = np.array(rews, dtype=np.float32).reshape(-1)
            don = np.array(don, dtype=bool).reshape(-1)
            total += float(rews[0])
            stc += 1
            obs = o
            if don[0]:
                break
        toks = max(0, 100 - int(gym0._env.tokens))
        r_all.append(total)
        wins.append(1 if total > 0.0 else 0)
        tok.append(toks)
        steps.append(stc)

    n = float(episodes)
    return {
        "win_rate": sum(wins) / n,
        "avg_reward": sum(r_all) / n,
        "avg_steps": sum(steps) / n,
    }


def run_training_pipeline(seed: int = SEED) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    env = FCEnvEnvironment()

    best = None
    for attempt in range(3):
        q = defaultdict(float)
        alpha = 0.18
        gamma = 0.98
        epsilon = 0.32
        epsilon_min = 0.03
        epsilon_decay = 0.996
        rewards: list[float] = []
        wins: list[int] = []

        (
            baseline_reward,
            baseline_win_rate,
            baseline_tokens_used,
            baseline_steps,
        ) = evaluate_policy(env, lambda _: random_policy_action(), EVAL_EPISODES)

        with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "episode_reward", "rolling_reward", "rolling_win_rate"])

            for episode in range(1, EPISODES + 1):
                obs = env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    key = state_key(obs)
                    if random.random() < epsilon:
                        action = random_policy_action()
                    else:
                        action = max(range(4), key=lambda a: q[(key, a)])

                    next_obs = env.step(Action(action=action))
                    reward = next_obs.reward
                    next_key = state_key(next_obs)

                    q[(key, action)] = q[(key, action)] + alpha * (
                        reward + gamma * max(q[(next_key, a)] for a in range(4)) - q[(key, action)]
                    )
                    obs = next_obs
                    done = obs.done
                    total_reward += reward

                rewards.append(total_reward)
                wins.append(1 if total_reward > 0.0 else 0)
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                rw = rewards[-50:]
                ww = wins[-50:]
                writer.writerow([episode, total_reward, sum(rw) / len(rw), sum(ww) / len(ww)])

        (
            trained_reward,
            trained_win_rate,
            trained_tokens_used,
            trained_steps,
        ) = evaluate_policy(
            env, lambda o: q_policy_action(o, q), EVAL_EPISODES
        )
        reward_delta = trained_reward - baseline_reward
        win_delta = trained_win_rate - baseline_win_rate
        candidate: dict = {
            "q": dict(q),
            "baseline_reward": baseline_reward,
            "baseline_win_rate": baseline_win_rate,
            "baseline_tokens_used": baseline_tokens_used,
            "baseline_steps": baseline_steps,
            "trained_reward": trained_reward,
            "trained_win_rate": trained_win_rate,
            "trained_tokens_used": trained_tokens_used,
            "trained_steps": trained_steps,
            "reward_delta": reward_delta,
            "win_delta": win_delta,
            "rewards": rewards,
            "wins": wins,
            "attempt": attempt + 1,
        }
        if best is None or candidate["win_delta"] > best["win_delta"]:
            best = candidate
        if reward_delta > 0.5 and win_delta > 0.20:
            best = candidate
            break

    assert best is not None
    _save_q_table(best["q"])

    plt.figure(figsize=(10, 4))
    rolling = [
        sum(best["rewards"][max(0, i - 49) : i + 1]) / len(best["rewards"][max(0, i - 49) : i + 1])
        for i in range(len(best["rewards"]))
    ]
    plt.plot(rolling, label="Rolling reward (50 ep)")
    plt.axhline(
        y=best["baseline_reward"],
        color="r",
        linestyle="--",
        label="Random baseline reward",
    )
    plt.title("Reward vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REWARD_PLOT)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Random baseline", "Trained agent"],
        [best["baseline_win_rate"], best["trained_win_rate"]],
        color=["#888888", "#2E8B57"],
    )
    plt.ylim(0, 1)
    plt.ylabel("Win-like rate")
    plt.title("Before vs After Training")
    plt.tight_layout()
    plt.savefig(WINRATE_PLOT)
    plt.close()

    ppo = run_ppo_and_save_sb3_model(PPO_TIMESTEPS)
    ppo_path = ppo["sb3_model_path"]
    vec_path = ppo["vecnorm_path"]
    ppo_stats = evaluate_ppo_vec(
        ppo_path,
        vec_path,
        episodes=EVAL_EPISODES,
        seed=seed,
        deterministic=True,
    )

    result: dict[str, Any] = {
        "seed": seed,
        "attempt_used": best["attempt"],
        "random_eval": {
            "win_rate": round(best["baseline_win_rate"], 4),
            "avg_reward": round(best["baseline_reward"], 4),
            "avg_steps": round(best["baseline_steps"], 2),
        },
        "q_tabular_eval": {
            "win_rate": round(best["trained_win_rate"], 4),
            "avg_reward": round(best["trained_reward"], 4),
            "avg_steps": round(best["trained_steps"], 2),
        },
        "ppo_eval": {
            "win_rate": round(float(ppo_stats["win_rate"]), 4),
            "avg_reward": round(float(ppo_stats["avg_reward"]), 4),
            "avg_steps": round(float(ppo_stats["avg_steps"]), 2),
        },
        "q_reward_delta": round(best["reward_delta"], 4),
        "q_win_rate_delta": round(best["win_delta"], 4),
        "sb3_model_path": ppo_path,
        "vecnorm_path": vec_path,
        "sb3_eval_best_dir": ppo.get("best_dir", BEST_MODEL_DIR),
        "ppo_timesteps": PPO_TIMESTEPS,
    }

    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    result = run_training_pipeline()
    print("Training complete (tabular Q + PPO).")
    re = result["random_eval"]
    print(
        f"Random: reward={re['avg_reward']:.3f} win={re['win_rate']:.3f} "
        f"steps={re['avg_steps']:.2f}"
    )
    qe = result["q_tabular_eval"]
    print(
        f"Q-tab: reward={qe['avg_reward']:.3f} win={qe['win_rate']:.3f} "
        f"steps={qe['avg_steps']:.2f}"
    )
    pe = result["ppo_eval"]
    print(
        f"PPO:  reward={pe['avg_reward']:.3f} win={pe['win_rate']:.3f} "
        f"steps={pe['avg_steps']:.2f}"
    )
    print(f"Saved metrics: {METRICS_CSV}, {EVAL_OUTPUT_PATH}")
    print(f"Saved plots: {REWARD_PLOT}, {WINRATE_PLOT}")
    print(f"SB3 PPO model: {result['sb3_model_path']}")
    print(f"VecNormalize: {result['vecnorm_path']}")


if __name__ == "__main__":
    main()
