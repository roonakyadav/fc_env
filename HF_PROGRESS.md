# Hackathon compliance progress

All items below are verified against this repository (code + a full `python train.py` run, `uvicorn` smoke test, and `docker build`).

## HF_RULES — Phase 1

* [x] **OpenEnv-style structure** — `models.py`, `environment.py`, `server/`, `create_fastapi_app` in `core/env_server.py`, documented in README
* [x] **Training scripts** — `train.py` (Q-learning + SB3 PPO; TRL path documented for LLM + HTTP client; optional `pip install -e ".[trl]"`)
* [x] **Real training evidence** — `artifacts/reward_curve.png`, `artifacts/training_metrics.csv`, `artifacts/evaluation.json`, `artifacts/win_rate_vs_random.png`
* [x] **Deployable on HF Space** — `app.py` ASGI, Gradio UI at `/ui`, command in `openenv.yaml`
* [x] **README** — problem, environment, reward, training, results, local + HF + Docker
* [x] **Blog post** — `BLOG_POST.md`
* [x] **Typed dataclasses** — `Action`, `Observation`, `State` in `models.py`
* [x] **Gym-style API** — `reset` / `step` / `state` on env + `gym_env.FCOpenEnvGym`
* [x] **Client/server** — `client.FCEvOpenEnvClient`; training can target live URL
* [x] **FastAPI** — `create_fastapi_app` + `server/app.py` + `app:app` entry
* [x] **Reward only from environment** — all learning uses `env.step` rewards
* [x] **Shaped / multi-part reward** — see `environment.py` and README
* [x] **Baseline comparison** — random vs trained in `train.py` and Gradio
* [x] **MCP (subset)** — `GET /tools/list`, `POST /tools/call`
* [x] **Kernel sandbox (optional stub)** — `kernel_sandbox.py`, `benchmark.py`
* [x] **Docker (optional, provided)** — `Dockerfile` builds; **not** required for RULES2 deployment text

## RULES2 — Phase 2 (HF Space architecture)

* [x] **Server (mandatory)** — FastAPI, URL-callable
* [x] **Required endpoints** — `POST /reset`, `POST /step`, `GET /state`, `GET /health`, `GET /docs` (OpenAPI)
* [x] **Repository installable** — `pyproject.toml` + `pip install -e .` / `requirements.txt`
* [x] **Client/server separation** — `client` does not import `server` internals
* [x] **Local run** — `uvicorn app:app`
* [x] **Training connects to live env** — `FCEvOpenEnvClient` + documented remote training
* [x] **Registry/Docker** — not required for hackathon per RULES2; Dockerfile included anyway

## RULES3 — Phase 3 (Setup & deployment)

* [x] **Reproducible venv** — `requirements.txt` + README (`uv` / `pip` with CPU index)
* [x] **Project structure** — `models.py`, `environment.py`, `client.py`, `server/app.py`, `pyproject.toml`, `openenv.yaml`, `README.md`
* [x] **HF Space outputs** — Gradio: action input, observation text, state traces, plots, buttons for run/retrain
* [x] **Local & remote** — `uv run` equivalent: `uvicorn` + optional Docker
* [x] **Training** — reward from environment; `train.py` end-to-end
* [x] **Ecosystem** — named env + README like other OpenEnv-style projects

**Status:** all sections complete (`[x]`).
