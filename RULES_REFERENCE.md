# Rules Reference

## Part 1 — HF Rules



# OpenEnv Hackathon — Complete Rules & Strategy Guide

---

## Table of Contents

1. [Non-Negotiable Requirements](#1-non-negotiable-requirements)
2. [What You Are Actually Building](#2-what-you-are-actually-building)
3. [Judging Breakdown](#3-judging-breakdown)
4. [What Winning Projects Look Like](#4-what-winning-projects-look-like)
5. [RL Training — Critical Rules](#5-rl-training--critical-rules)
6. [Reward Design](#6-reward-design)
7. [Environment Design Rules](#7-environment-design-rules)
8. [Training Proof Requirements](#8-training-proof-requirements)
9. [What Makes You Stand Out](#9-what-makes-you-stand-out)
10. [Themes](#10-themes)
11. [Current Project Audit — Connect4](#11-current-project-audit--connect4)
12. [Fix Strategy — Upgrade Path](#12-fix-strategy--upgrade-path)
13. [Technical & System Rules](#13-technical--system-rules)
14. [Final Structured Checklist](#14-final-structured-checklist)

---

## 1. Non-Negotiable Requirements

> Miss even ONE of these → you are cooked.

- [ ] Must use **OpenEnv (latest)**
- [ ] Must have a **training script** (TRL or Unsloth)
- [ ] Must show **real training evidence** (reward/loss plots)
- [ ] Must **deploy environment on HF Spaces**
- [ ] Must have a **README** containing:
  - Problem description
  - Environment description
  - Results
- [ ] Must include either a **blog post** OR a **video under 2 minutes**
- [ ] Judges evaluate from your **Space URL only**
- [ ] **No changes allowed after submission deadline**

---

## 2. What You Are Actually Building

You are **not** building a model. You are **not** building a UI.

You are building:

```
Environment → Agent → Reward → Learning Loop
```

**Core idea:**

> Model acts → environment evaluates → reward → model improves

---

## 3. Judging Breakdown

### 🥇 Environment Innovation — 40%

- Is it **new**?
- Is it **challenging**?
- Does it test **real behavior**?

> **If your idea is basic → you already lost.**

---

### 🥈 Storytelling — 30%

- Can a **non-technical person** understand it?
- Is it **engaging**?

> **If the README is boring → you lose here.**

---

### 🥉 Training Improvement — 20%

Must show:
- Reward curves
- Before vs. after comparison
- Baseline comparison

> **No proof of learning = instant downgrade.**

---

### ⚙️ Reward + Pipeline — 10%

- Is reward **logical**?
- Does training **actually work**?

> **Bad reward = fake learning.**

---

## 4. What Winning Projects Look Like

Judges explicitly said:

> **Messy + ambitious + real training > polished + boring**

| ✅ Good | ❌ Bad |
|--------|--------|
| Hard problem | Clean UI with no real learning |
| Visible learning | Toy problem |
| Imperfect UI is fine | No baseline comparison |

---

## 5. RL Training — Critical Rules

### Minimum Correct Pipeline

```
Prompt → Action → Environment → Reward → Update
```

> If you don't have this loop → you didn't do RL.

### Key Rules

- RL only works if **success probability > 0**
- Use **GRPO / RLVR** (verifiable rewards)
- Prefer **programmatic reward** over LLM-judge-only reward

### Common Failures

| Failure | Consequence |
|---------|-------------|
| Task too hard | Reward always 0 |
| Weak reward | Model cheats |
| No baseline | No comparison possible |
| No plots | No proof of learning |

---

## 6. Reward Design

> **Your reward = your task definition. Bad reward → model hacks system.**

### ✅ Good Reward Characteristics

- Multi-component
- Hard to game
- Provides meaningful signal

**Example of a good reward:**

```
+ correctness
+ format
+ speed
- cheating
- timeout
```

### ❌ Bad Reward Characteristics

- Only 0/1 at end (sparse)
- Easy to exploit
- Vague signal

### Reward Hacking — This Is Real

The model **WILL**:
- Exploit bugs
- Bypass logic
- Fake success

You **MUST**:
- Add constraints
- Add checks
- Monitor outputs

---

## 7. Environment Design Rules

### Required Methods

```python
reset()    # Initialize environment, return initial observation
step()     # Execute action, return observation + reward + done
state()    # Return metadata about episode (optional but expected)
```

### Required Elements

- State / Observation representation
- Reward computation
- Follow **Gym-style API**
- Follow **client/server separation**
- Follow **OpenEnv structure**

### Pitfalls to Avoid

- Unrealistic environment
- No failure cases
- Static (non-interactive) tasks

---

## 8. Training Proof Requirements

> **This is the most important part for winning.**

You **MUST** show:

- 📈 Reward curve (reward vs. episodes)
- 📊 Baseline vs. trained agent comparison
- 🎥 Actual before/after outputs

**What judges want to see:**

| Stage | Output |
|-------|--------|
| Before training | Bad / random output |
| After training | Visibly better output |

> If you cannot show this clearly → you lose.

---

## 9. What Makes You Stand Out

1. **Original problem** — NOT chess, NOT tic-tac-toe
2. **Strong reward** — teaches something real
3. **Real learning** — visible improvement over time
4. **Clear story** — problem → solution → results

---

## 10. Themes

Pick **ONE** direction (these are directions, not hard limits):

- Multi-agent interaction
- Long-horizon planning
- World modeling
- Self-improvement
- Wildcard

---

## 11. Current Project Audit — Connect4

### What Was Built

**Environment:**
- Connect4 RL environment
- Implements: `reset()`, `step(action)`, `_check_win()`
- Standard 6×7 board
- Reward: +1 on win

**RL Loop (Conceptually Correct):**

```python
while not done:
    observation = env.observe()
    action = agent.choose(observation)
    result = env.step(action)
    reward = result.reward
    agent.learn(reward)
```

**Architecture:**

Follows OpenEnv structure with typed dataclasses (`Action`, `Observation`, `State`) and API endpoints (`reset`, `step`, `state`).

---

### Honest Assessment

#### ❌ Problem #1: Idea is Weak

Connect4 = same category as chess, snake, tic-tac-toe, grid-world clones.

Judges have explicitly said they've seen all of these. **This kills your 40% innovation score immediately.**

#### ❌ Problem #2: Reward is Too Simple

Current reward:
```
win  → +1
else → 0
```

This is **sparse**, **not informative**, and **easy to plateau on**. Kills learning quality and reward design score.

#### ❌ Problem #3: No Proof of Learning

Missing:
- Reward curves
- Baseline comparison
- Trained vs. untrained outputs

**Without this: your project = demo, not RL.**

#### ❌ Problem #4: No "Why This Matters"

Saying "we trained an agent to play Connect4" is not enough. Judges want:
- A capability gap being addressed
- Real-world relevance
- Research potential

---

### Current Score Projection (If Submitted As-Is)

| Category | Score |
|----------|-------|
| Innovation | ❌ LOW |
| Storytelling | ⚠️ MEDIUM |
| Training Proof | ❌ LOW |
| Reward Design | ❌ LOW |

> **This is not a winning submission.**

---

## 12. Fix Strategy — Upgrade Path

> You don't need to throw everything away. Upgrade the idea, not restart.

### Step 1 — Reframe the Narrative (Critical)

**Don't say:**
> "Connect4 agent"

**Say:**
> "Strategic multi-agent reasoning environment with adversarial planning"

This repositions your project under the **Multi-Agent Interaction** theme.

---

### Step 2 — Add Real Complexity

Current state: deterministic, perfect info, trivial strategy.

Add **any one** of these:

| Option | What to Add |
|--------|-------------|
| **A: Partial Observability** | Hide opponent moves, add limited memory |
| **B: Multi-step Reasoning** | Delayed rewards, planning required |
| **C: Tool Use** | Agent simulates moves internally before playing |

---

### Step 3 — Fix the Reward (Most Important)

**Replace:**
```
+1  win
0   otherwise
```

**With:**
```
+1.0   win
-1.0   loss
+0.2   good move (heuristic)
-0.2   bad move
+0.1   blocking opponent's threat
+0.1   creating 3-in-a-row
```

This gives: dense reward ✅, learnable signal ✅

---

### Step 4 — Show Learning

You **must** produce:

- **Plot 1:** Reward vs. episodes
- **Plot 2:** Win rate vs. baseline (random agent)
- **Demo:** Random agent vs. trained agent side-by-side

---

### Step 5 — Add One Killer Twist

Pick **one**:

| Option | Description |
|--------|-------------|
| 🔥 **Multi-agent negotiation** (Best) | Agents communicate before moving, attempt deception |
| 🔥 **Self-play curriculum** | Difficulty increases automatically as agent improves |
| 🔥 **Explainable reasoning** | Agent outputs natural language reasoning for each move |

---

## 13. Technical & System Rules

### Environment Design Pattern (Strict Contract)

#### Core Loop

```python
reset()
# → initializes environment
# → returns initial observation

step(action)
# → executes action
# → returns: observation, reward, done flag

state()
# → returns metadata about episode (optional but expected)
```

#### RL Interaction Model

```
Agent:       observes state → takes action → receives reward → learns
Environment: executes action → updates state → computes reward
```

---

### Type System Requirement

OpenEnv enforces **typed interfaces using dataclasses**. No loose dicts or random JSON.

```python
@dataclass
class Action:
    ...

@dataclass
class Observation:
    state: ...
    reward: float
    done: bool

@dataclass
class State:
    episode_id: str
    step_count: int
```

---

### Environment Implementation Pattern

```python
class Environment:

    def reset(self):
        ...

    def step(self, action):
        ...
        return observation

    def state(self):
        ...
```

Requirements:
- Deterministic or stochastic logic
- Full episode lifecycle
- Reward calculation **inside** the environment

---

### Server Requirement (OpenEnv Standard)

Environment **must** be exposed via a FastAPI server:

```python
from core.env_server import create_fastapi_app

env = YourEnvironment()
app = create_fastapi_app(env)
```

> This is how judges interact with your environment. It must be runnable.

---

### Docker Support

- **Optional**, but part of the system
- Standard pattern: `Dockerfile` + containerized environment
- Not mandatory for submission; only needed for advanced setups

---

### MCP (Model Context Protocol) — Tool System

**Problem:** Agents need external tools (APIs, file system, database, etc.)

**Solution:** Use MCP interface

**Core endpoints:**
```
tools/list   → discover available tools
tools/call   → execute a tool
```

This enables: tool-augmented agents and real-world interaction environments.

---

### Advanced Environment Type — Kernel Sandbox

Agent submits **code**, not actions.

**Flow:**
```
reset()
  → detect GPU
  → run baseline (cuBLAS)
  → store baseline performance

step()
  → receive kernel code
  → compile it
  → run benchmark

benchmark()
  → measure: TFLOPS, bandwidth, occupancy, registers
  → compute: speedup vs baseline
```

Output must include: performance metrics and improvement score.

---

### Deployment Requirements

**CLI Workflow:**
```bash
openenv init my_env
cd my_env
openenv push
# OR
openenv push --repo-id username/my-env
```

**HF Spaces requirement:**
- Environment must be **runnable** and **accessible via URL**
- Judges will pull and run your environment

---

### Accepted Problem Types

From the official "Ideas" slide:

- Any environment agents can learn from
- Realistic simulations
- Interactive systems

**Examples hinted at:** vending systems, customer simulations, marketplaces, real-world interaction loops

> **Not just games. Not static problems. Not trivial environments.**

---

### Training Integration

Must be compatible with either:
- **Unsloth**, OR
- **HuggingFace TRL**

Environment must be:
- Compatible with RL training loop
- Usable programmatically

---

### System Architecture

Expected project structure:

```
models.py        → dataclasses (Action, Observation, State)
environment.py   → core environment logic
server/app.py    → FastAPI wrapper
```

---

### Key System Idea

> **OpenEnv = "Write environment once → train anywhere"**

- No custom glue code
- Standardized interface
- Reusable environments

---

### Universal Design Pattern

```
Observe → Act → Reward → Learn → Repeat
```

---

### Implied Constraints (Very Important)

Your environment **must**:
- Be **interactive** (not a static dataset)
- Support **iterative learning**
- Produce **measurable outputs**
- Allow **training loop integration**
- Expose a **clear reward signal**
- Follow the **API contract exactly**

---

## 14. Final Structured Checklist

### MUST HAVE

- [ ] `reset()` / `step()` / `state()` implemented
- [ ] Dataclass-based types (`Action`, `Observation`, `State`)
- [ ] FastAPI server wrapping the environment
- [ ] Working environment loop (runs end-to-end)
- [ ] Reward logic inside the environment
- [ ] HF Spaces deployment (accessible via URL)
- [ ] Training script (TRL or Unsloth)
- [ ] Reward curves / training plots
- [ ] README with problem, environment, results
- [ ] Blog post OR video (< 2 min)

### SHOULD HAVE

- [ ] Training compatibility verified (TRL / Unsloth)
- [ ] Measurable metrics (win rate, reward improvement)
- [ ] Meaningful state transitions
- [ ] Baseline comparison (e.g., random agent)
- [ ] Before/after output demos

### OPTIONAL / ADVANCED

- [ ] MCP tools integration
- [ ] Docker containerization
- [ ] Kernel sandbox-style task (code submission → benchmark)
- [ ] Multi-agent interaction
- [ ] Self-play curriculum
- [ ] Explainable agent reasoning

### EXPECTED FINAL OUTPUT

An environment that:
1. **Runs** — fully functional, accessible via HF Spaces URL
2. **Trains** — integrates with TRL or Unsloth RL loop
3. **Proves improvement** — visible reward curves and before/after comparison

## Part 2 — Space Architecture Rules

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

## Part 3 — Setup & Deployment Rules

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