import gradio as gr
from stable_baselines3 import PPO

from gym_wrapper import FCGymEnv

_env = FCGymEnv()
model = PPO.load("models/final_model.zip", device="cpu", env=_env)


def run_episode() -> str:
    obs, _ = _env.reset()
    log: list[str] = []
    done = False
    last_r = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = _env.step(int(action))
        last_r = float(reward)
        log.append(f"action {int(action)}  reward {last_r:.2f}  done={bool(done)}")

    body = "FC Env PPO (deterministic)\n\n" + "\n".join(log) + f"\n\nEpisode return: {last_r:.2f}"
    return body


demo = gr.Interface(
    fn=run_episode,
    inputs=[],
    outputs=gr.Textbox(label="Run log"),
    title="FC Env PPO",
    description="Runs one full episode with the loaded policy.",
)

if __name__ == "__main__":
    demo.launch()