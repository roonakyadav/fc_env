# The environment is the spec: due diligence on a budget

> **What this teaches:** The difference between a policy that is **right on average** and one that is **right without wasting probes**—under the same API and the same training loop.

> **TL;DR** We built a reinforcement-learning environment that is *not* interchangeable with a generic grid or card game. The player pays **token cost** for **noisy, tiered information**, then must **commit** or **walk**—and some deals are **traps** where the best action is to leave. Learning is shown with **curves, baselines, and side-by-side episode traces** on a live Space.

---

## 1. The problem in plain language

Imagine you can only run **a few expensive checks** before you decide to **approve a loan**, **close a hire**, or **block a high-risk user**. Every extra check costs time and money. Sometimes the data **lies** or is **incomplete on purpose** (a “trap” case). The wrong policy is not “play it safe and always gather more data”—it is to **waste the whole budget** and still commit, or to **pass on a good deal** because you never looked.

**FC OpenEnv** makes that tension explicit: a **token budget**, **low vs high “probe”** actions, a **commit** action, and a **skip/exit** action. Episodes are drawn from a distribution that includes **traps**—low true value with **misleading** clues—so the agent must learn **when not to play**.

This is the **clever bit** for judges: the world is not “maximize a score in a known MDP toy”; it is **allocate attention under uncertainty** with **a real option to leave**.

---

## 2. Our approach (not “we used RL”)

1. **Environment as the single source of truth** — all rewards and outcomes come from `step()`; `train.py` does not relabel “success.”
2. **Two training signals** — **tabular Q-learning** for fast, interpretable baselines, and **PPO** (Stable-Baselines3) on a **Gymnasium** wrapper for a neural policy saved as `models/final_model.zip`.
3. **Same API for any policy** — FastAPI endpoints `reset` / `step` / `state` plus a **Gradio** UI that shows **trained vs random** so judges see behavior without cloning the repo.
4. **Optional LLM path** — `client.FCEvOpenEnvClient` can target a remote Space; **TRL / Unsloth** can wrap the same reward (optional `pip install -e ".[trl]"`).

---

## 3. Reward design (this is the idea)

A single **+1 if you win** would make the task **too sparse** and would invite brittle policies. We use a **multi-component** reward (implemented in `environment.py`):

**Dense (per step):** small gains for **useful** reveals, penalties for **redundant** purchases, tiered **costs** for “high” clues, and a **skip** term that flips sign depending on **trap** vs good card (walking away is sometimes *correct*).

**Terminal (episode end):** a hard penalty if the agent did **not** gather a minimum of information; a **asymmetric** outcome for “skip on trap” vs “skip on value”; a **down-weighted** outcome if the agent **committed to a trap** after overspending; and a **quality + efficiency** mix if the card was real and the agent **stopped**.

**Creativity, not just correctness:** the best constant policy (always reveal, always skip, always stop) **does not** maximize this mixture— the agent has to **sequence** actions and **condition on partial clues**.

---

## 4. Training and what we show as evidence

After `python train.py`, the repo writes:

- **`artifacts/evaluation.json`** — baseline vs trained **mean reward** and **win-like rate** (and path to the SB3 zip).
- **`artifacts/reward_curve.png`** — rolling return vs episode with a **random** horizontal reference.
- **`artifacts/win_rate_vs_random.png`** — **before/after** bar chart.
- **`artifacts/training_metrics.csv`** — episode-level log for replots.

**What a judge should look at first:** the **bar chart** and **curve** in `artifacts/` and the same figures **embedded in the Gradio** page at `/ui`. Numbers change with random seed; the **separation** between random and trained is the story.

*Optional after you publish the Space: add 1–2 screenshots of the Gradio comparison (summary + two traces) into your submission or deck.*

---

## 5. The demo (what to do in 30 seconds on the Space)

1. Open **`/ui`**.  
2. Click **Run Comparison**.  
3. Read the **summary** (deltas).  
4. Read **two episode traces** side by side: random vs trained.  
5. Glance at the two **images** (reward and win rate).

If you are technical: `POST /reset` and a few `POST /step` calls; watch `tokens` and `reward` in JSON.

That is the **narrative arc**: *same API, worse vs better policy, visual proof*.

---

## 6. Why this is not “just another compliant OpenEnv”

- **Idea:** **query budgeting** and **opt-out** as first-class actions—not only navigation in a grid.  
- **Reward:** explicitly **anti-collapsed**; skip is **not** always right; **more data** is **not** always better.  
- **Demo:** **interaction + plots + traces** in one URL.

---

## 7. Lessons and limits

- **Stochasticity** in clue order and trap sampling means policies should be evaluated with **enough** eval episodes (we use hundreds in `train.py`).  
- For **very large** language models, the bottleneck is **calls to the environment**, not the tabular or PPO baselines—**TRL/GRPO** is the next step on the *same* reward.  
- **Assumption:** the metaphor is “card reveals”; the **abstraction** is for any **costly probe, commit, or exit** workflow.

**Space URL (set when you ship):** `https://huggingface.co/spaces/<username>/<space>`

---

## 🧠 Takeaway

**Good agents are not only accurate—they are efficient.** This environment bakes *efficiency* into the same reward as *correctness*: waste the budget, skip a good deal, or commit to a trap, and the score says so. That is the lesson we want a judge to remember in one breath.

---

*This post is the companion to `README.md`—read the code for exact coefficients and edge cases; read this for the story judges remember.*
