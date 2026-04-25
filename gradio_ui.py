"""Gradio UI for the HF Space: API-style manual play + trained vs random comparison."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import gradio as gr

from environment import FCEnvEnvironment
from models import Action


def _get_train() -> Any:
    """Lazy import so the Space app starts without loading SB3/torch until needed."""
    import train  # noqa: WPS433 — runtime import for optional comparison tab

    return train


def ensure_trained() -> dict:
    train = _get_train()
    if os.path.exists(train.EVAL_PATH):
        with open(train.EVAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return train.run_training_pipeline()


def rollout_trace(policy_name: str, max_steps: int = 8) -> tuple[str, float]:
    train = _get_train()
    env = FCEnvEnvironment(max_steps=max_steps)
    obs = env.reset()
    done = False
    total_reward = 0.0
    trace = []

    if policy_name == "Trained":
        q_table = train.load_q_table()

    while not done:
        if policy_name == "Trained":
            action = train.q_policy_action(obs, q_table)
        else:
            import random

            action = random.randint(0, 3)
        next_obs = env.step(Action(action=action))
        trace.append(
            f"step={next_obs.step_number} action={action} reward={next_obs.reward:.3f} "
            f"tokens={next_obs.tokens} done={next_obs.done}"
        )
        total_reward += next_obs.reward
        obs = next_obs
        done = next_obs.done

    return "\n".join(trace), total_reward


def run_demo() -> tuple[str, str, str, str, str]:
    train = _get_train()
    result = ensure_trained()
    trained_trace, trained_total = rollout_trace("Trained")
    random_trace, random_total = rollout_trace("Random")

    summary = (
        "## Before training vs after training\n"
        f"- Random baseline reward: **{result['baseline_reward']:.3f}**\n"
        f"- Trained reward: **{result['trained_reward']:.3f}**\n"
        f"- Reward delta: **{result['reward_delta']:.3f}**\n"
        f"- Random win rate: **{result['baseline_win_rate']:.3f}**\n"
        f"- Trained win rate: **{result['trained_win_rate']:.3f}**\n"
        f"- Win rate delta: **{result['win_rate_delta']:.3f}**\n\n"
        f"Single-run demo reward (trained): **{trained_total:.3f}**\n\n"
        f"Single-run demo reward (random): **{random_total:.3f}**"
    )
    return (
        summary,
        trained_trace,
        random_trace,
        train.REWARD_PLOT,
        train.WINRATE_PLOT,
    )


def retrain_and_demo() -> tuple[str, str, str, str, str]:
    _get_train().run_training_pipeline()
    return run_demo()


def _obs_to_dict(o) -> dict[str, Any]:
    return {
        "revealed_clues": list(o.revealed_clues),
        "tokens": o.tokens,
        "step_number": o.step_number,
        "low_remaining": o.low_remaining,
        "high_remaining": o.high_remaining,
        "done": o.done,
        "reward": o.reward,
    }


def _state_to_dict(s) -> dict[str, Any]:
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "tokens": s.tokens,
        "done": s.done,
        "low_revealed": s.low_revealed,
        "high_revealed": s.high_revealed,
    }


def manual_reset() -> tuple[FCEnvEnvironment, dict, dict, str]:
    env = FCEnvEnvironment()
    o = env.reset()
    s = env.state()
    return env, _obs_to_dict(o), _state_to_dict(s), "New episode. Choose an action (0–3) and click **Take step**."


def manual_step(
    env: Optional[FCEnvEnvironment], action: int | str | float
) -> tuple[FCEnvEnvironment, dict, dict, str]:
    if env is None:
        env = FCEnvEnvironment()
        o = env.reset()
        s = env.state()
        return (
            env,
            _obs_to_dict(o),
            _state_to_dict(s),
            "No episode yet; started a new one. Use **Reset** anytime to clear.",
        )

    act = int(action)
    o = env.step(Action(action=act))
    s = env.state()
    note = "Episode done — use **Reset** to start over." if o.done else "Step applied."
    return env, _obs_to_dict(o), _state_to_dict(s), note


def build_blocks() -> gr.Blocks:
    with gr.Blocks(title="FC OpenEnv") as demo:
        gr.Markdown(
            "# FC OpenEnv\n"
            "Use the **Interactive** tab to drive the same environment as the HTTP API, or **Comparison** "
            "for trained vs random rollouts and plots (loads training code on first use)."
        )

        with gr.Tabs():
            with gr.Tab("Interactive (action → observation, reward, state)"):
                gr.Markdown(
                    "Actions: **0** = low clue, **1** = high clue, **2** = stop (commit), **3** = skip. "
                    "`observation.reward` is the return from the last `step` (0 after a fresh `reset`)."
                )
                env_state = gr.State()  # type: ignore[var-annotated]  # FCEnvEnvironment
                with gr.Row():
                    action_dd = gr.Dropdown(
                        choices=[0, 1, 2, 3],
                        value=0,
                        label="Next action (integer)",
                    )
                with gr.Row():
                    reset_btn = gr.Button("Reset (new episode)", variant="primary")
                    step_btn = gr.Button("Take step", variant="secondary")
                with gr.Row():
                    obs_out = gr.JSON(label="Last observation (JSON)", value={})
                    state_out = gr.JSON(label="State metadata (JSON)", value={})
                log_out = gr.Textbox(label="Log", lines=2)

                reset_btn.click(
                    fn=manual_reset,
                    inputs=None,
                    outputs=[env_state, obs_out, state_out, log_out],
                )
                step_btn.click(
                    fn=manual_step,
                    inputs=[env_state, action_dd],
                    outputs=[env_state, obs_out, state_out, log_out],
                )

            with gr.Tab("Comparison (trained vs random + plots)"):
                with gr.Row():
                    run_btn = gr.Button("Run comparison", variant="primary")
                    retrain_btn = gr.Button("Retrain + run comparison")

                summary_box = gr.Markdown()
                with gr.Row():
                    trained_box = gr.Textbox(
                        label="Trained agent episode (action, reward, tokens, done)",
                        lines=12,
                    )
                    random_box = gr.Textbox(
                        label="Random policy episode (action, reward, tokens, done)",
                        lines=12,
                    )
                with gr.Row():
                    reward_img = gr.Image(label="Reward vs episodes", type="filepath")
                    winrate_img = gr.Image(label="Win-like rate: baseline vs trained", type="filepath")

                run_btn.click(
                    run_demo, outputs=[summary_box, trained_box, random_box, reward_img, winrate_img]
                )
                retrain_btn.click(
                    retrain_and_demo, outputs=[summary_box, trained_box, random_box, reward_img, winrate_img]
                )

        # Prime interactive tab so JSON is not empty on first load.
        demo.load(
            fn=manual_reset,
            inputs=None,
            outputs=[env_state, obs_out, state_out, log_out],
        )
    return demo
