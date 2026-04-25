# OpenEnv — Environment Setup, Deployment & Training Rules

---

## Table of Contents

1. [Virtual Environment Setup](#1-virtual-environment-setup)
2. [Project Initialization](#2-project-initialization)
3. [Project Structure Requirements](#3-project-structure-requirements)
4. [Deployment Flow](#4-deployment-flow)
5. [What Successful Deployment Means](#5-what-successful-deployment-means)
6. [HF Space Output Requirements](#6-hf-space-output-requirements)
7. [Local & Remote Execution Modes](#7-local--remote-execution-modes)
8. [Training Integration Rules](#8-training-integration-rules)
9. [Critical Insight — Your Environment is a Reward Engine](#9-critical-insight--your-environment-is-a-reward-engine)
10. [Training Execution Options](#10-training-execution-options)
11. [Ecosystem Context](#11-ecosystem-context)
12. [Final System Pipeline](#12-final-system-pipeline)
13. [Status Checklist](#13-status-checklist)

---

## 1. Virtual Environment Setup

**MANDATORY** — project must run in an isolated, reproducible environment.

### Option A — Preferred

```bash
uv venv
uv pip install openenv-core
```

### Option B

```bash
conda create -n openenv python=3.12
conda activate openenv
```

> **Rule: Dependencies must be reproducible across machines.**

---

## 2. Project Initialization

```bash
openenv init my_env
```

This generates the following scaffold:

| File / Folder | Purpose |
|---------------|---------|
| `models.py` | Typed dataclasses |
| `environment.py` | Core environment logic |
| `server/` | FastAPI server |
| `pyproject.toml` | Packaging config |
| `Dockerfile` | Container definition |

> **Rule: Your project structure must follow the OpenEnv template exactly.**

---

## 3. Project Structure Requirements

**Required files:**

```
server/
models.py         → Action, Observation, State dataclasses
environment.py    → core environment logic
client.py         → client interface
server/app.py     → FastAPI entry point
pyproject.toml    → packaging
openenv.yaml      → config
README.md
```

> **Rules:**
> - Clean modular separation — **NOT everything in one file**
> - Must be **pip-installable** with clean import paths

---

## 4. Deployment Flow

```bash
openenv push
```

This performs the following steps automatically:

1. Validates environment
2. Prepares files
3. Uploads to HF Space
4. Builds automatically
5. Deploys

> **Rules:**
> - Your project **must pass validation**
> - Must deploy **without manual hacks or workarounds**

---

## 5. What Successful Deployment Means

After a successful push, all of the following must be true:

- [ ] Space URL is live
- [ ] API is working
- [ ] UI is visible
- [ ] Logs are clean — no crashes

> **Rule: "Runs locally" is NOT enough. Must run on HF infrastructure.**

---

## 6. HF Space Output Requirements

Your Space must display:

- Action input
- Observation output
- State tracking
- Logs

> **Rule: Must be interactive — even a basic interaction loop is required.**

---

## 7. Local & Remote Execution Modes

**Mode 1 — Run locally via server:**

```bash
uv run server
```

**Mode 2 — Run via Docker (optional):**

```bash
docker run -p 8000:8000 <image>
```

**Mode 3 — Clone Space directly:**

```bash
git clone <hf-space>
```

> **Rule: Environment must be reproducible outside of HF infrastructure.**

---

## 8. Training Integration Rules

### TRL (GRPO)

Core pattern:

```python
env.step(completion)
reward = result.reward
```

> **Rule: Reward must come FROM the environment — not manually computed inside the training script.**

---

### Unsloth Integration

Pattern:
1. Load model
2. Apply LoRA
3. Use environment as the reward function

> **Rule: The environment acts as the evaluation function — not the training script.**

---

## 9. Critical Insight — Your Environment is a Reward Engine

> **This wins or loses you the hackathon.**

Your environment is **not** just a simulation. It is **not** just an API.

It is a **reward engine**:

```
Model        → generates output
Environment  → scores it
Training     → optimizes on that score
```

The environment is the source of truth for what "good" means. If the reward is weak or externally computed, your RL loop is not real.

---

## 10. Training Execution Options

| Option | Cost | Speed | Notes |
|--------|------|-------|-------|
| **Local** (CPU / laptop) | ✅ Free | Slower | Fine for hackathon |
| **HF Jobs** (T4, A100, etc.) | ⚠️ Paid | Faster | Pay-as-you-go |

> **Rule: Stick to local training unless you have a specific reason to use HF Jobs.**

**GPU selection (only if using HF Jobs):**
- Small models → T4 is sufficient
- Do not over-allocate compute

```bash
hf jobs hardware   # shows available compute options
```

---

## 11. Ecosystem Context

Your environment should feel like a legitimate peer to existing environments, not a random script.

Examples of existing environments in the ecosystem:

- BrowserGym
- Chess
- CodeEnv
- Connect4
- Trading
- *(and others)*

> **Rule: Your environment should be structured and presented like one of these — a real, named, reusable environment.**

---

## 12. Final System Pipeline

Your full pipeline must connect end-to-end:

```
Environment (HF Space)
        ↓
  Exposes API
        ↓
Training script connects
        ↓
Model generates actions
        ↓
Environment returns reward
        ↓
Training updates model
```

Every link in this chain must work. A break at any point means your RL loop is incomplete.

---

## 13. Status Checklist

You are close to complete. Confirm the following before submission:

| Item | Status |
|------|--------|
| Training uses environment rewards (not manual) | ✅ Confirm |
| Space runs on HF infra without errors | ✅ Confirm |
| README clearly explains the full pipeline | ✅ Confirm |