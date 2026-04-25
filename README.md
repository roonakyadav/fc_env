---
title: FC Env PPO
emoji: 🎮
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
tags:
  - reinforcement-learning
  - ppo
  - gymnasium
---
## Model

The trained PPO policy is in `models/final_model.zip` (use Git LFS in this repo for large files).

## Run locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (e.g. `http://127.0.0.1:7860`).

## Hugging Face Space

The Space runs `app.py` as a **Gradio** app: it loads the policy, wraps the FC environment in Gymnasium, and the UI runs one full episode with the PPO policy.
