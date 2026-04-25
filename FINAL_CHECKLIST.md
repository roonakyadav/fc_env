# Final submission checklist

## HF_RULES compliance

* [x] OpenEnv-style layout (`models`, `environment`, `server`, `core` factory) and gym wrapper
* [x] Training pipeline with real plots, CSV, evaluation JSON, baseline vs trained
* [x] HF Space–ready `app:app` (FastAPI + Gradio)
* [x] README: problem, environment, reward, training, results, deploy
* [x] Blog post (`BLOG_POST.md`)
* [x] Shaped, multi-component reward inside `environment.py`
* [x] MCP-style routes (`/tools/list`, `/tools/call`) and optional kernel `benchmark` stub
* [x] SB3 PPO + `models/final_model.zip`; TRL optional extra documented

## RULES2 compliance

* [x] `/reset`, `/step`, `/state`, `/health`, `/docs` on running server
* [x] Pip-installable package (`pyproject.toml`, clean imports)
* [x] `client.py` for remote `reset`/`step`/`state`
* [x] No Docker requirement for Space; Docker provided and builds

## RULES3 compliance

* [x] `pyproject.toml` (mandatory for `pip install`)
* [x] `openenv.yaml` with entrypoint
* [x] `server/app.py` FastAPI + `app.py` entry
* [x] Interactive Gradio: actions, observations, state, logs, artifacts
* [x] `train.py` uses only environment rewards
* [x] `Dockerfile` present and tested

## Deployment readiness

* [x] Runs locally (`uvicorn app:app`, venv + CPU torch index)
* [x] Configured for HF Spaces (port 7860, `app:app`, `/ui` Gradio)
* [x] API working (`/health`, `/reset`, `/step` JSON, `/state`)
* [x] Docker builds successfully

## Training readiness

* [x] `python train.py` completes without error
* [x] `artifacts/training_metrics.csv`, `artifacts/reward_curve.png`, `artifacts/evaluation.json`, `artifacts/win_rate_vs_random.png`
* [x] `models/final_model.zip` produced by PPO; `artifacts/q_table.json` from Q-learning
* [x] Reward and win-rate baselines included in `evaluation.json`

## Proof commands (repro)

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .
python train.py
uvicorn app:app --host 0.0.0.0 --port 7860
docker build -t fc-openenv:final .
```
