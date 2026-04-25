# OpenEnv — HF Space Architecture & Deployment Rules

---

## Table of Contents

1. [Core Concept — What Your HF Space Must Provide](#1-core-concept--what-your-hf-space-must-provide)
2. [Important Architecture Rules](#2-important-architecture-rules)
3. [Development Workflow](#3-development-workflow)
4. [How Training Connects to Your Space](#4-how-training-connects-to-your-space)
5. [Critical Insight — What Your Space Actually Is](#5-critical-insight--what-your-space-actually-is)
6. [Local & Docker — Important Clarification](#6-local--docker--important-clarification)
7. [CLI Deployment](#7-cli-deployment)
8. [What Your Project Must Enable](#8-what-your-project-must-enable)
9. [Final System Summary](#9-final-system-summary)

---

## 1. Core Concept — What Your HF Space Must Provide

Every HF Space = **3 components**

---

### Component 1 — Server `(MANDATORY)`

Provides a running environment endpoint, accessible via URL:

```
https://<username>-<space>.hf.space
```

**Required endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `/reset` | Start a new episode |
| `/step` | Take an action |
| `/state` | Return metadata |
| `/health` | Must return healthy |
| `/docs` | API docs (FastAPI auto-generated) |

> **Rule: Your environment MUST be callable remotely.**

---

### Component 2 — Repository `(MANDATORY)`

Must be installable as a package:

```bash
pip install git+https://huggingface.co/spaces/<user>/<space>
```

That means:
- Proper Python package structure
- Importable modules
- Clean, explicit dependencies

> **Rule: Your Space is not just a UI — it is a package.**

---

### Component 3 — Registry `(OPTIONAL / ADVANCED)`

Docker image support:

```bash
docker pull registry.hf.space/<space>:latest
```

> **IMPORTANT:**
> - This is **optional**
> - You **do NOT need this** for the hackathon
> - Only relevant for scaling / infra work

---

## 2. Important Architecture Rules

### Rule 1 — Client-Server Separation

| Side | Responsibility |
|------|---------------|
| **Client** | Calls the environment remotely |
| **Server** | Runs the environment logic |

> **Rule: No tight coupling. Clients must NOT import server internals.**

---

### Rule 2 — Standard API Contract `(NON-NEGOTIABLE)`

Your environment **must** implement:

```python
reset()
step(action)
state()
```

> This is already enforced by OpenEnv — it must be followed exactly.

---

### Rule 3 — Async + Sync Support `(Recommended)`

- **Async** — preferred
- **Sync wrapper** — optional

> Not mandatory, but considered good practice.

---

## 3. Development Workflow

Judges expect your project to be locally runnable. Standard local dev flow:

```bash
git clone <space>
cd <space>
uv sync
uv run server
```

Or alternatively:

```bash
uvicorn app:app
```

> **Rule: Must run locally without friction.**

---

## 4. How Training Connects to Your Space

Typical training integration:

```python
client = EchoEnv(base_url="https://your-space.hf.space")
client.reset()
client.step(action)
```

> **CRITICAL RULE:**
> Your training script must:
> - Interact with a **live, running environment**
> - **NOT** operate on a static dataset

---

## 5. Critical Insight — What Your Space Actually Is

> **Most people miss this.**

Your Space serves **three roles simultaneously**:

| Role | Purpose |
|------|---------|
| **API** | RL training endpoint |
| **Package** | Reusable, importable environment |
| **Demo** | Judge-facing UI |

> **If any one of these is weak → you lose points.**

---

## 6. Local & Docker — Important Clarification

You may have seen references to Docker build, Docker run, and registry usage in the docs.

**The reality for this hackathon:**

| Feature | Required? |
|---------|-----------|
| `docker build` | ❌ NO |
| Registry pull | ❌ NO |
| CLI deployment | ❌ Optional |
| Simple Space push | ✅ YES |

> Docker and registry tooling are **not required** for submission. Do not spend time on them.

---

## 7. CLI Deployment

An optional deployment path via the OpenEnv CLI:

```bash
openenv init my_env
openenv push
```

> Not using this is **fine**. Simple HF Space push is sufficient.

---

## 8. What Your Project Must Enable

### ✅ Must Have

1. Running environment accessible via URL
2. Proper Python package structure
3. Training script that interacts with the live environment
4. `reset()` / `step()` / `state()` fully implemented
5. Space runs without errors

### ⚠️ Should Have (High Value)

- FastAPI endpoints fully working
- `/health` endpoint returning healthy
- Clean, working imports
- Reproducible local setup

### ❌ Ignore (For Now)

- Docker registry
- Advanced scaling
- Infrastructure optimizations

---

## 9. Final System Summary

Your entire system should behave as one connected pipeline:

```
HF Space
  └── runs environment
        └── exposes API
              └── training script connects
                    └── agent learns
                          └── results shown in README / UI
```