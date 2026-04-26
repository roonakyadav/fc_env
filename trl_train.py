"""TRL PPO training for FCOpenEnvGym using distilgpt2; rewards only from env.step()."""

import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from gym_env import FCOpenEnvGym

# Matches FCOpenEnvGym default; obs[1] = step_number / max_steps (see gym_env._vec).
GYM_MAX_STEPS = 8


def obs_to_prompt(obs_array, step_num, tokens_left):
    return (
        "You are a decision agent with a token budget.\n"
        f"Step: {step_num} | Tokens left: {tokens_left:.0f}\n"
        "Actions:\n"
        "0 = reveal cheap clue (costs few tokens)\n"
        "1 = reveal expensive clue (costs more tokens)\n"
        "2 = commit (accept the deal)\n"
        "3 = exit (walk away from deal)\n"
        "Strategy: spend just enough tokens to decide, then commit or exit.\n"
        "Respond with ONLY one digit: 0, 1, 2, or 3\n"
        "Action:"
    )


def parse_action(text, fallback=0):
    for char in text.strip():
        if char in "0123":
            return int(char)
    return fallback


def _step_and_tokens_from_obs(obs):
    """Map FCOpenEnvGym 16-dim observation to step index and token count."""
    tokens_left = float(obs[0]) * 100.0 if len(obs) > 0 else 50.0
    step_num = 0
    if len(obs) > 1:
        step_num = int(round(float(obs[1]) * GYM_MAX_STEPS))
        step_num = max(0, min(GYM_MAX_STEPS, step_num))
    return step_num, tokens_left


config = PPOConfig(
    model_name="distilgpt2",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=2,
    seed=42,
    log_with=None,
)


def run_trl_training(episodes=200, smoke_test=False):
    if smoke_test:
        episodes = 5

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLMWithValueHead.from_pretrained("distilgpt2")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("distilgpt2")
    ref_model = ref_model.to(device)
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

    env = FCOpenEnvGym()
    episode_rewards = []
    parse_successes = 0
    parse_total = 0
    query_tensors = []
    response_tensors = []
    rewards_buffer = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            step_num, tokens_left = _step_and_tokens_from_obs(obs)

            prompt = obs_to_prompt(obs, step_num, tokens_left)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            query_tensor = inputs["input_ids"][0].contiguous()

            with torch.no_grad():
                response_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=3,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_tensor = response_ids[0][inputs["input_ids"].shape[1] :].contiguous()
            response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)

            action = parse_action(response_text)
            parse_total += 1
            if response_text.strip() and response_text.strip()[0] in "0123":
                parse_successes += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)

            query_tensors.append(query_tensor)
            response_tensors.append(response_tensor)
            rewards_buffer.append(torch.tensor(float(reward), dtype=torch.float32, device=device))

            while len(query_tensors) >= config.batch_size:
                b = config.batch_size
                ppo_trainer.step(
                    query_tensors[:b],
                    response_tensors[:b],
                    rewards_buffer[:b],
                )
                query_tensors = query_tensors[b:]
                response_tensors = response_tensors[b:]
                rewards_buffer = rewards_buffer[b:]

        episode_rewards.append(episode_reward)
        if not smoke_test and episode % 20 == 0:
            recent = (
                mean(episode_rewards[-20:])
                if len(episode_rewards) >= 20
                else mean(episode_rewards)
            )
            print(f"Episode {episode}/{episodes} | Recent avg reward: {recent:.3f}")

    # Partial tail is dropped; PPOTrainer.step requires a full batch of size config.batch_size.

    parse_rate = parse_successes / max(parse_total, 1)
    return episode_rewards, parse_rate, model, tokenizer


def evaluate_trl(model, tokenizer, episodes=50):
    device = next(model.parameters()).device
    env = FCOpenEnvGym()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            step_num, tokens_left = _step_and_tokens_from_obs(obs)
            prompt = obs_to_prompt(obs, step_num, tokens_left)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            action = parse_action(response)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    return mean(rewards)


def evaluate_random(episodes=50):
    env = FCOpenEnvGym()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = random.randint(0, 3)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    return mean(rewards)


def save_artifacts(episode_rewards, trl_mean, baseline_mean, parse_rate, smoke_test=False):
    Path("artifacts").mkdir(exist_ok=True)

    if not smoke_test:
        window = 20
        rolling = [
            mean(episode_rewards[max(0, i - window) : i + 1])
            for i in range(len(episode_rewards))
        ]
        plt.figure(figsize=(9, 5))
        plt.plot(episode_rewards, alpha=0.3, label="Raw rewards")
        plt.plot(rolling, linewidth=2.4, label=f"Rolling avg (window={window})")
        plt.axhline(
            y=baseline_mean,
            linestyle="--",
            color="red",
            linewidth=2,
            label=f"Random baseline ({baseline_mean:.3f})",
        )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("TRL PPO Training: Reward vs Episodes")
        plt.legend()
        plt.tight_layout()
        plt.savefig("artifacts/trl_reward_curve.png", dpi=130)
        plt.close()

    summary = {
        "model": "distilgpt2",
        "training_episodes": len(episode_rewards),
        "eval_episodes": 50,
        "random_baseline_mean": round(baseline_mean, 4),
        "trl_policy_mean": round(trl_mean, 4),
        "improvement": round(trl_mean - baseline_mean, 4),
        "parse_success_rate": round(parse_rate, 4),
    }
    Path("artifacts/trl_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=== TRL PPO Training ===")
    eff_ep = 5 if args.smoke_test else args.episodes
    print(f"Model: distilgpt2 | Episodes: {eff_ep} | Smoke test: {args.smoke_test}")

    episode_rewards, parse_rate, model, tokenizer = run_trl_training(
        episodes=args.episodes,
        smoke_test=args.smoke_test,
    )

    print(f"Training complete. Parse success rate: {parse_rate * 100:.1f}%")

    if not args.smoke_test:
        print("Evaluating trained policy...")
        trl_mean = evaluate_trl(model, tokenizer, episodes=50)
        baseline_mean = evaluate_random(episodes=50)
    else:
        trl_mean = mean(episode_rewards) if episode_rewards else 0.0
        baseline_mean = evaluate_random(episodes=5)

    summary = save_artifacts(
        episode_rewards, trl_mean, baseline_mean, parse_rate, args.smoke_test
    )

    print()
    print("=== RESULTS ===")
    print(f"Random baseline mean: {baseline_mean:.3f}")
    print(f"TRL policy mean:      {trl_mean:.3f}")
    print(f"Improvement:          {trl_mean - baseline_mean:.3f}")
    print(f"Parse success rate:   {parse_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
