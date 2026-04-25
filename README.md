# FC OpenEnv — Due diligence under a probe budget (OpenEnv + HF Space)

## 🧠 Intuition (read this first)

Think **hiring** (or fraud review—it is the same shape):

- You can only ask **a few** questions, and each one **costs** time and money.
- You can **stop the process early** if the candidate looks bad.
- Or you **commit** and make an offer.

**The catch:**

- **Too many** questions waste the budget and still don’t remove uncertainty.
- **Hiring a bad** candidate is expensive.
- **Rejecting a good** one is expensive too.

**This environment trains agents to make that three-way trade-off**—not to maximize a score in isolation, but to be **right *and* efficient** under a probe limit. The card metaphor is just a compact way to run the same loop: **probe → decide → commit or exit**.

## 🎯 The point (in 5 seconds)

The agent must learn:

- ❌ **Do not** blindly gather all information.  
- ❌ **Do not** guess with almost no evidence.  
- ✅ **Do** spend **just enough** to decide.  
- ✅ **Do** walk away when the deal is not worth it.  

**That is the whole game.**

> This is not a “maximize reward” problem.  
> This is a **“know when to stop thinking”** problem.

---

**What this is (one line):** A reinforcement-learning environment where the agent must **decide which costly information to buy** before **committing or walking away**—not “pick an action in a grid.”

---

## ⚡ What to look for (the “aha” in the demo)

**The moment to notice:** the trained agent often **does less work than random** (fewer wasteful probes, more targeted stops or exits)—**and still gets a better score.** That is the “oh wow” frame: *efficiency with higher return*, not “try harder.”

**Do not** only look at the numbers—watch **behavior** in the **Trained** trace (and compare to **Random** on `/ui`):

- **Bad / trap-leaning cases:** the trained policy tends to **stop probing earlier** and **exit** when the signal is wrong—**saving tokens** instead of buying every expensive clue.
- **Uncertain cases:** it **samples** low-cost clues first, then **adds** high-cost clues only when it still does not know enough to stop or skip.
- **Confident good cases:** it **commits (stop)** when the evidence lines up, instead of burning the budget on redundant reveals.

That pattern **emerges** from the reward: it is not hand-scripted. If you only see “trained line is higher on the plot,” you have proof of learning; if you see the **token and action pattern** above, you have proof of **meaningful** learning.

---

## 30-Second demo (for judges)

Do this in order so the story lands in under half a minute.

1. Open your **Hugging Face Space** (see URL at the end of this file).
2. Open the **UI** path on the same Space, usually **`/ui`** (for example, `https://YOUR_USERNAME-YOUR_SPACENAME.hf.space/ui` once deployed).
3. Click **Run Comparison**. Read the **summary** (random vs trained metrics).
4. Scroll: **Trained** trace vs **Random** trace—same rules, different decisions (tokens, stop/skip, reward).
5. Open the two **images**: reward curve and win-rate bar chart—**visible learning vs baseline**.

If you only have the API: `POST /reset` → a few `POST /step` with `{"action":0}`–`3` → see `reward` and `tokens` in JSON; then read `artifacts/evaluation.json` after a local `python train.py`.

This is the “demo story” judges expect: **interaction** + **before/after** + **a plot that proves learning**.

---

## The real-world problem (why this is not a generic RL env)

**Swap test:** If you could drop in CartPole and nothing would change, you would lose. Here the **physics of the task** are specific:

- You are **purchasing information** (low vs high “probes”) from a **limited budget** (tokens).
- The **true value** of the item is hidden; **clues can be noisy or decoys** (trap episodes mimic misleading diligence).
- **Stopping** means you accept exposure (good or bad). **Skipping** is only optimal when the hidden state is a trap you never fully pay to resolve.

This maps to domains where **each query has a real cost** and **walking away** is a first-class action:

- **Fraud / risk review:** every extra signal (bureau, document, phone tree node) costs time and money; sometimes the right move is to **decline the customer** before overspending.
- **Hiring & procurement due diligence:** cheap screens vs expensive references; you **stop** and make an offer, or **walk** if cheap signals look wrong.
- **Model evaluation for deployment:** a fixed “evaluation token” budget to probe a black-box system before a go/no-go—same **probe → act or exit** pattern.

The sports-card surface is a **familiar, compact narrative**; the **structure** is the innovation: *budgeted, costly observation with trap-aware exit*.

---

## How this environment is built (code map)

| Piece | Role |
|--------|------|
| `models.py` | `Action`, `Observation`, `State` (typed; no ad hoc JSON) |
| `environment.py` | `FCEnvEnvironment` — all **reward** and episode logic here |
| `gym_env.py` | `Gymnasium` wrapper (16-dim obs) for PPO |
| `train.py` | Q-learning + random baseline + PPO; **only** uses rewards from `step()` |
| `core/env_server.py` + `server/app.py` | FastAPI `create_fastapi_app` + Gradio on `/ui` |
| `client.py` | Remote `reset` / `step` / `state` for a deployed Space |
| `gradio_ui.py` | Interactive comparison + plots |

**Actions (integer):** `0` = cheap probe, `1` = expensive probe, `2` = **commit** (stop), `3` = **exit** (skip).

**Observation:** revealed clues (or `HIDDEN`), `tokens`, step index, `done`, and **per-step** `reward` (shaped).

---

## Reward design (core innovation)

Judges are not only checking that “reward exists”—they are checking that **the objective teaches something non-obvious**. Ours is **deliberately multi-objective and anti-gaming** (sparse win-only would collapse to memorization).

We split reward into **dense step terms** and a **terminal term** (see `environment.py`).

### Per-step (dense) signals

| Mechanism | Role |
|-----------|------|
| **Low clue reveal** | +`0.08` when a new fact is shown; **−0.10** if the agent hammers an empty row; each reveal spends tokens. |
| **High clue reveal** | **Tiered cost** (15 / 20 / 25 tokens by remaining count); return `0.15 - cost/200` so better intel can still be “worth it” or **too expensive**. |
| **Stop (commit)** | Small **−0.05** to prefer informed commits over impulsive early stop. |
| **Skip (exit)** | **+0.10** on **trap** (correct walk-away), **−0.20** on good cards (opportunity cost of skipping a real asset). |
| **Budget hit zero** | Episode ends; extra **−0.10**—burning the full budget is rarely free. |

### Terminal (outcome) shaping

| Outcome | Effect |
|--------|--------|
| **Too few clues** (fewer than 2 total reveals) | **−1.0** — you cannot “guess”; forced diligence. |
| **Skip on a trap** | **+1.0** (walk away from a bad deal). |
| **Skip on a good card** | **−1.0** (you left value on the table). |
| **Commit on a trap** | Heavily negative mix of overspend + decoy (see `_terminal_reward` in code) — the policy must **not** dump tokens then buy. |
| **Commit on a real card** | `0.6 ×` estimated quality from hidden stats + **token bank bonus** + high-clue use bonus + mild over-reveal penalty. |

**Why this matters for learning (not just correctness):**

- **Trivial “always skip”** fails: skipping good items is punished; skipping traps is the rare high terminal reward.
- **Trivial “reveal everything”** fails: tiered costs and terminal penalties for over-reveal; reward is not monotonic in “more clues.”
- The agent must learn **when** to buy signal, **when** to hold tokens, and **which exit** is right—i.e. **long-horizon credit** without a single sparse 0/1 at the end of every episode in the same way.

All training uses **only** `observation.reward` from `step()`.

---

## Training and proof artifacts

1. **Q-learning** vs a **random** baseline: `artifacts/evaluation.json`, `artifacts/training_metrics.csv`, `artifacts/reward_curve.png`, `artifacts/win_rate_vs_random.png`, `artifacts/q_table.json`.
2. **PPO (Stable-Baselines3)** on the Gym wrapper → `models/final_model.zip` (control length with `FC_PPO_TIMESTEPS`, default `14000`).

**TRL / Unsloth (LLM policies):** this task is discrete; for a **text policy**, call the Space with `client.FCEvOpenEnvClient` and use **returned reward only**—optional: `pip install -e ".[trl]"`.

---

## Interactive UI (a first-class feature)

The Gradio app is not an afterthought—it is the **judge-facing demo**.

At **`/ui`** you get:

- **Run Comparison** and **Retrain + Run Comparison** to refresh metrics.
- **Side-by-side episode traces** (step, action, reward, tokens, done) for **trained** vs **random**—visible **state transition** without reading code.
- Embedded **reward curve** and **win-rate** plots from `artifacts/`.

**API (same process):** `GET /`, `GET /health`, `POST /reset`, `POST /step` with `{"action": 0-3 }`, `GET /state`, `GET /docs`; `GET /tools/list`, `POST /tools/call` (MCP-style surface).

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
# or: python app.py
```

- **UI:** `http://localhost:7860/ui`
- **Client example:**

```python
from client import FCEvOpenEnvClient
with FCEvOpenEnvClient("http://127.0.0.1:7860") as c:
    c.health()
    c.reset()
    c.step(0)
```

---

## Why this matters (bridge to the real world)

| Use the env for… | It stress-tests… |
|-------------------|-----------------|
| **RL under investigation budgets** | Stopping when marginal information isn’t worth its cost. |
| **“Decline the bad” policies** | Trap episodes where **exit** is optimal. |
| **Benchmarking LMs** as policies | Same HTTP `step` reward; swap tabular / PPO / **TRL** without rewriting the world. |
| **Teaching “reward = specification”** | The behavior you want is **in** the environment, not hand-coded in `train.py`. |

---

## ⚠️ Failure modes (when reward design breaks)

These are the failure stories **good** submissions acknowledge—**any** cost–information environment can fail this way if the objective is wrong:

| If… | The policy often… | Why it matters here |
|-----|-------------------|---------------------|
| Probes are **too cheap** relative to terminal mistakes | **Over-explores** (burns the whole budget) | You never learn “when to stop buying information.” |
| **Exit / skip** is over-penalized | **Exits too early** or **never** skips bad traps | You cannot model “walk away from a bad deal” as a win. |
| Rewards are **too sparse** (e.g. one bit at the end) | **Learning collapses** or **plateaus**; signal is too late | Dense + terminal terms exist so credit is not only at episode end. |

Our multi-part reward is there to **avoid** these collapse modes; if something still looks flat, the first place to look is **whether the environment is actually giving gradable signal**, not the learning algorithm alone.

---

## Differentiation (how this competes on “innovation” when everyone is compliant)

Most RL environments (implicitly) reward *do more, get more.* **This one punishes that**—extra probes have real cost, and **more actions are not a free win.** Here, the sharp contrast is: **more steps ≠ better outcomes** unless they are the *right* steps.

Many submissions will pass **OpenEnv + training + Space**. This project’s angle on top of that:

1. **Environment idea:** **Costly, typed probes** + **trap distribution** + **legitimate “do nothing by leaving”** (skip)—not a replacement grid world.
2. **Reward design:** **Multi-stage** dense + **trap-conditioned skip** + **commit-vs-exit** asymmetry; explicitly **anti-constant** action policies.
3. **Demo:** **Trained vs random** traces + **curves** in one place (`/ui`), not a buried notebook.

**Space URL (replace with yours):** `https://huggingface.co/spaces/<your-username>/<your-space-name>`

**Deploy:** Prefer **Docker** (this repo’s `Dockerfile`, port `7860`) or a custom command; see `openenv.yaml` for the suggested `uvicorn` line. Plain **Gradio SDK** is awkward because the ASGI `app` is **FastAPI** with Gradio mounted at **`/ui`**.

---

## Install and reproduce

**From a Space (Git LFS):**

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "git+https://huggingface.co/spaces/<user>/<space>.git@main"
```

**Local dev:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pip install -e .
python train.py
```

**Docker:** `docker build -t fc-openenv .` then `docker run -p 7860:7860 fc-openenv`

---

## Longer read

- **`BLOG_POST.md`:** Story, results, and what we learned (submission narrative).
- **`kernel_sandbox.py` / `benchmark.py`:** Optional advanced “code → benchmark” pattern (not used in the main loop).

## Project layout

```
models.py  environment.py  gym_env.py  train.py  app.py  client.py
server/app.py  core/env_server.py  gradio_ui.py
artifacts/  models/final_model.zip
pyproject.toml  openenv.yaml  Dockerfile  requirements.txt  BLOG_POST.md
```

## 🧠 Final takeaway

The goal is not just to be **right**.

The goal is to be right **at the lowest cost**—in probes, time, and tokens. This environment makes that trade-off **unavoidable**: the reward is not a generic “higher is better for more actions” game; it is a **stopping problem** with teeth.
