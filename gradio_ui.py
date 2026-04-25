"""Gradio UI for the HF Space (mounted at /ui)."""

from __future__ import annotations

import gradio as gr

from environment import FCEnvEnvironment
from models import Action


def build_blocks() -> gr.Blocks:
    print("Gradio UI initialized", flush=True)
    with gr.Blocks(title="FC OpenEnv") as demo:
        gr.Markdown("# FC OpenEnv UI")
        gr.Markdown(
            "Open **Interactive** to step the same environment as `POST /step` on the API; "
            "**Status** is a quick connectivity check."
        )

        with gr.Tabs():
            with gr.Tab("Status"):
                btn = gr.Button("Test Button")
                output = gr.Textbox(label="Result")

                def click_fn() -> str:
                    return "UI is working"

                btn.click(click_fn, inputs=None, outputs=output)

            with gr.Tab("Interactive"):
                def _o(o) -> dict:
                    return {
                        "revealed_clues": list(o.revealed_clues),
                        "tokens": o.tokens,
                        "step_number": o.step_number,
                        "low_remaining": o.low_remaining,
                        "high_remaining": o.high_remaining,
                        "done": o.done,
                        "reward": o.reward,
                    }

                def _s(e: FCEnvEnvironment) -> dict:
                    st = e.state()
                    return {
                        "episode_id": st.episode_id,
                        "step_count": st.step_count,
                        "tokens": st.tokens,
                        "done": st.done,
                        "low_revealed": st.low_revealed,
                        "high_revealed": st.high_revealed,
                    }

                def on_reset() -> tuple[FCEnvEnvironment, dict, dict, str]:
                    env = FCEnvEnvironment()
                    o = env.reset()
                    return env, _o(o), _s(env), "New episode. Pick action 0–3, then **Take step**."

                def on_step(
                    env: FCEnvEnvironment | None, action: int | str | float
                ) -> tuple[FCEnvEnvironment, dict, dict, str]:
                    if env is None:
                        e = FCEnvEnvironment()
                        o = e.reset()
                        return e, _o(o), _s(e), "Started new episode."
                    o = env.step(Action(action=int(action)))
                    msg = "Episode done — use **Reset**." if o.done else "OK."
                    return env, _o(o), _s(env), msg

                env_st = gr.State()  # type: ignore[var-annotated]
                with gr.Row():
                    act = gr.Dropdown(choices=[0, 1, 2, 3], value=0, label="Action")
                with gr.Row():
                    b_reset = gr.Button("Reset (new episode)", variant="primary")
                    b_step = gr.Button("Take step")
                obs_json = gr.JSON(label="Last observation", value={})
                st_json = gr.JSON(label="State", value={})
                log = gr.Textbox(label="Log", lines=2, value="Click **Reset** to start.")

                b_reset.click(
                    on_reset, outputs=[env_st, obs_json, st_json, log]
                )
                b_step.click(
                    on_step, inputs=[env_st, act], outputs=[env_st, obs_json, st_json, log]
                )
    return demo
