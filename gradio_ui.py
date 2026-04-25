"""Gradio UI for the HF Space (mounted at /ui)."""

from __future__ import annotations

import ast
import gradio as gr

from environment import FCEnvEnvironment
from models import Action

# Match reset() in FCEnvEnvironment (for token bar; backend unchanged)
MAX_TOKENS = 100
HIDDEN = "HIDDEN"

# High-contrast dark shell + layout (forced; ignores system light/dark)
CSS_STRING = """
/* Lock browser + Gradio to dark, avoid washed-out system rendering */
html {
  color-scheme: dark !important;
}

body {
  background-color: #0f1115 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}

.gradio-container, .gradio-container.fillable {
  background-color: #0f1115 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}

footer, .footer, [class*="footer"] {
  display: none !important;
}

/* Headings */
h1, h2, h3, h4 {
  color: #ffffff !important;
}

/* All buttons: solid, readable; Gradio uses .gr-button */
button, .gr-button {
  background-color: #1f2937 !important;
  color: #ffffff !important;
  border: 1px solid #2e3440 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #ffffff !important;
}
button span, .gr-button span {
  color: #ffffff !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #ffffff !important;
}
.gr-button-primary, button.gr-button-primary {
  background-color: #1f2937 !important;
  color: #ffffff !important;
  border-color: #2e3440 !important;
}
.gr-button-primary:hover, button:hover:enabled {
  background-color: #2b3444 !important;
  border-color: #00c853 !important;
  color: #ffffff !important;
}

/* Disabled: distinct via color, no opacity < 1 (keeps text readable) */
button:disabled, .gr-button:disabled, button.gr-button:disabled {
  opacity: 1 !important;
  color: #9ca3af !important;
  -webkit-text-fill-color: #9ca3af !important;
  background-color: #0f1319 !important;
  border-color: #1a202c !important;
  cursor: not-allowed;
}

/* Cards & blocks */
.card, .gr-box, .gr-panel, .gr-html {
  background-color: #1a1d24 !important;
  color: #ffffff !important;
  border-radius: 10px !important;
  opacity: 1 !important;
}
.gr-form {
  background: #0f1115 !important;
  color: #ffffff !important;
  opacity: 1 !important;
  border: none !important;
}
.gr-html, .gr-html * {
  color: #ffffff;
}
.gr-html .fc-muted, .prose .fc-muted { color: #cccccc !important; }
main, main.contain, .form, .form > div {
  background: #0f1115 !important;
  color: #ffffff !important;
}

/* Inputs (if any) */
textarea, input, select {
  background-color: #111318 !important;
  color: #ffffff !important;
  border-color: #2e3440 !important;
  opacity: 1 !important;
}

/* Labels */
label, .label-wrap, [data-testid] label {
  color: #cccccc !important;
  opacity: 1 !important;
}

/* Markdown: strong contrast, no wash-out */
.markdown, .prose {
  color: #ffffff !important;
  opacity: 1 !important;
}
.markdown p, .prose p {
  color: #e5e7eb !important;
  line-height: 1.6 !important;
  opacity: 1 !important;
}
.markdown li, .prose li { color: #e5e7eb !important; opacity: 1 !important; }
.markdown strong, .prose strong {
  color: #ffffff !important;
  opacity: 1 !important;
  font-weight: 600;
}
.markdown code, .prose code {
  background-color: #1f2937 !important;
  color: #ffffff !important;
  padding: 2px 6px;
  border-radius: 6px;
  border: 1px solid #2a2f3a !important;
  opacity: 1 !important;
}
.markdown h1, .markdown h2, .markdown h3, .prose h1, .prose h2, .prose h3 {
  color: #ffffff !important;
  font-weight: 600 !important;
  opacity: 1 !important;
}
[class*="markdown"] {
  color: #ffffff !important;
  opacity: 1 !important;
}
.tabitem, button[role="tab"] {
  color: #ffffff !important;
  opacity: 1 !important;
}
[class*="tab-nav"] {
  color: #ffffff !important;
  opacity: 1 !important;
}

/* App text helper */
p, .gradio-container p, li, span:not(.fc-reward-pos):not(.fc-reward-neg) {
  opacity: 1 !important;
}

/* Compare / About: card shell */
.content-card, .gr-group.content-card, div.content-card {
  background-color: #161a22 !important;
  border: 1px solid #2a2f3a !important;
  border-radius: 12px !important;
  padding: 16px !important;
  margin-top: 12px !important;
  opacity: 1 !important;
  box-shadow: 0 1px 0 0 #0a0c10 inset, 0 4px 20px -8px #000a;
}
.content-card .prose, .content-card .markdown, .content-card [class*="markdown"] {
  background: transparent !important;
  color: #ffffff !important;
  opacity: 1 !important;
}
.content-card .gr-markdown { min-height: 0; }

/* Counters above actions */
.fc-counter { font-size: 0.95rem; line-height: 1.4; }

.section-title {
  font-size: 18px;
  font-weight: 600;
  color: #ffffff !important;
  margin-bottom: 8px;
  line-height: 1.3;
  opacity: 1 !important;
  letter-spacing: 0.01em;
  border-bottom: 1px solid #2a2f3a;
  padding-bottom: 10px;
  margin-top: 0;
}

/* FC: panels & secondary (no faint grey) */
.fc-muted { color: #cccccc !important; opacity: 1 !important; }
.fc-panel {
  background-color: #1a1d24 !important;
  color: #ffffff !important;
  border: 1px solid #2e3440 !important;
  border-radius: 10px;
  padding: 14px 16px;
  opacity: 1 !important;
}
.fc-h2 { color: #ffffff !important; font-weight: 600; font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; margin: 0 0 8px; }

/* Clue cards */
.clue-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 800px) { .clue-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
@media (max-width: 520px) { .clue-grid { grid-template-columns: 1fr; } }
.clue-card {
  background-color: #1a1d24 !important;
  border: 1px solid #2e3440 !important;
  color: #ffffff !important;
  border-radius: 8px;
  padding: 10px 12px;
  min-height: 64px;
  transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  opacity: 1 !important;
}
.clue-card--new {
  background-color: #222831 !important;
  border-color: #00c853 !important;
  box-shadow: 0 0 0 1px #00c853;
  animation: fc-pulse 0.8s ease 1;
}
@keyframes fc-pulse {
  from { box-shadow: 0 0 0 2px #00c853; }
  to { box-shadow: 0 0 0 0 #00c853; }
}
.clue-card-k {
  color: #cccccc !important;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 4px;
  opacity: 1 !important;
}
.clue-card-t { color: #cccccc !important; font-size: 0.65rem; opacity: 1 !important; }
.clue-card-v { color: #ffffff !important; font-size: 0.9rem; line-height: 1.4; white-space: pre-wrap; }
.clue-hidden { color: #aaaaaa !important; font-size: 1.1rem; }

/* Token bar */
.fc-token-bar { margin-top: 6px; }
.fc-token-track { height: 6px; background: #0f1115; border-radius: 3px; overflow: hidden; border: 1px solid #2e3440; }
.fc-token-fill { height: 100%; background: linear-gradient(90deg, #0d8044, #00c853); border-radius: 3px; }

/* Rewards (intentional accent) */
.fc-reward-pos { color: #00c853 !important; font-weight: 600; -webkit-text-fill-color: #00c853; }
.fc-reward-neg { color: #ff5252 !important; font-weight: 600; -webkit-text-fill-color: #ff5252; }
.fc-reward-neu { color: #cccccc !important; font-weight: 600; -webkit-text-fill-color: #cccccc; }
"""


def _format_clue_line(raw: str) -> str:
    if raw == HIDDEN:
        return "???"
    try:
        t = ast.literal_eval(raw)
        if isinstance(t, (list, tuple)) and len(t) == 2:
            k, v = t
            label = str(k).replace("_", " ").strip().title()
            return f"{label}: {v}"
    except (ValueError, SyntaxError, TypeError):
        pass
    return raw


def _tier_name(idx: int) -> str:
    return "Low-cost" if idx < 3 else "High-cost"


def _clue_cards_html(
    revealed_clues: tuple[str, ...] | list[str], highlight_index: int | None
) -> str:
    n = 6
    if len(revealed_clues) < n:
        return "<div class='fc-muted'>Start a new episode to see clues.</div>"

    parts: list[str] = [
        "<p class='fc-h2' style='margin:0 0 10px'>Clues</p>",
        "<div class='clue-grid'>",
    ]
    for i in range(n):
        raw = revealed_clues[i]
        is_hidden = raw == HIDDEN
        body = '<span class="clue-hidden">???</span>' if is_hidden else _format_clue_line(raw)
        hcls = "clue-card clue-card--new" if highlight_index == i else "clue-card"
        tier = _tier_name(i)
        parts.append(
            f'<div class="{hcls}">'
            f'<div class="clue-card-k">Clue {i + 1} <span class="clue-card-t">({tier})</span></div>'
            f'<div class="clue-card-v">{body}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def _newly_revealed_index(
    pre: list[str] | tuple[str, ...] | None, post: tuple[str, ...]
) -> int | None:
    if pre is None or len(pre) < 6 or len(post) < 6:
        return None
    for i in range(6):
        if pre[i] == HIDDEN and post[i] != HIDDEN:
            return i
    return None


def _outcome_text(
    user_action: int,
    pre: dict,
    o,
    new_idx: int | None,
) -> str:
    if pre is None:
        return "—"

    pre_tokens = int(pre.get("tokens", 0))
    if user_action in (0, 1) and pre_tokens <= 0 and o.done:
        return "Out of tokens — decision committed (forced)."

    if new_idx is not None and user_action == 0:
        return "Low-cost clue revealed"
    if new_idx is not None and user_action == 1:
        return "High-cost clue revealed"

    if user_action == 0 and int(pre.get("low_remaining", 0)) == 0 and pre_tokens > 0:
        return "No low-cost clues remaining. Tokens were spent; nothing left to reveal."
    if user_action == 1 and int(pre.get("high_remaining", 0)) == 0 and pre_tokens > 0:
        return "No high-cost clues remaining. Tokens were spent; nothing left to reveal."

    if user_action == 0 and new_idx is None and pre_tokens > 0:
        return "No low-cost clues left — spent tokens, no new clue"
    if user_action == 1 and new_idx is None and pre_tokens > 0:
        return "No high-cost clues left — spent tokens, no new clue"

    if user_action == 2:
        if o.reward < -0.1:
            return "Bad decision penalty" if o.done else "Cost applied"
        return "You committed" if o.done else "—"

    if user_action == 3:
        if o.reward > 0.05:
            return "Smart skip saved cost"
        if o.reward < -0.01:
            return "Costly skip on a strong pick"
        return "You skipped the candidate"

    if o.done:
        return "Episode over"
    return "—"


def _snapshot(o) -> dict:
    return {
        "revealed_clues": o.revealed_clues,
        "tokens": o.tokens,
        "step_number": o.step_number,
        "low_remaining": o.low_remaining,
        "high_remaining": o.high_remaining,
        "done": o.done,
        "reward": o.reward,
    }


def _last_step_html(o, outcome: str) -> str:
    r = float(o.reward)
    if r > 1e-6:
        rc, cls = f"+{r:.3f}", "fc-reward-pos"
    elif r < -1e-6:
        rc, cls = f"{r:.3f}", "fc-reward-neg"
    else:
        rc, cls = f"{r:.3f}", "fc-reward-neu"
    return (
        f"<div class='fc-panel'>"
        f"<p class='fc-h2' style='margin:0 0 10px'>Last step result</p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Step reward</span> "
        f"<span class='{cls}'>{rc}</span></p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Tokens left</span> {o.tokens} / {MAX_TOKENS}</p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Step</span> {o.step_number}</p>"
        f"<p style='margin:8px 0 0' class='fc-muted'>{outcome}</p>"
        f"</div>"
    )


def _token_bar_html(o) -> str:
    pct = min(100.0, max(0.0, 100.0 * o.tokens / max(1, MAX_TOKENS)))
    return (
        f"<div class='fc-panel fc-token-bar'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;gap:12px'>"
        f"<span class='fc-h2' style='margin:0'>Tokens</span>"
        f"<span class='fc-muted'>{o.tokens} / {MAX_TOKENS}</span></div>"
        f"<div class='fc-token-track'><div class='fc-token-fill' style='width:{pct}%;'></div></div></div>"
    )


def _flow_badge(
    o,
    last_action: int | None,
) -> str:
    if o.done and last_action == 2:
        return "Episode finished · <span class='fc-reward-neg' style='font-weight:600'>You committed</span>"
    if o.done and last_action == 3:
        return "Episode finished · <span class='fc-reward-pos' style='font-weight:600'>You skipped</span>"
    if o.done:
        return "Episode finished"
    return "Episode running — reveal clues, skip, or commit when ready."


def _button_state_from_obs(o) -> tuple:
    d = o.done
    toks = o.tokens
    can_low = (not d) and toks > 0 and o.low_remaining > 0
    can_high = (not d) and toks > 0 and o.high_remaining > 0
    can_choose = not d
    return (
        gr.update(interactive=can_low, visible=True),
        gr.update(interactive=can_high, visible=True),
        gr.update(interactive=can_choose, visible=True),
        gr.update(interactive=can_choose, visible=True),
    )


COUNTER_INACTIVE = (
    '<div class="fc-counter" style="margin:0">'
    '<span style="color:#9ca3af">Low clues left: —</span></div>',
    '<div class="fc-counter" style="margin:0">'
    '<span style="color:#9ca3af">High clues left: —</span></div>',
)


def _counter_markdown_pair(o) -> tuple[str, str]:
    c_low = "#ff5252" if o.low_remaining == 0 else "#00c853"
    c_high = "#ff5252" if o.high_remaining == 0 else "#00c853"
    low = (
        f'<div class="fc-counter" style="margin:0">'
        f'<span style="color:{c_low};font-weight:600">Low clues left: {o.low_remaining}</span></div>'
    )
    high = (
        f'<div class="fc-counter" style="margin:0">'
        f'<span style="color:{c_high};font-weight:600">High clues left: {o.high_remaining}</span></div>'
    )
    return low, high


def build_blocks() -> gr.Blocks:
    with gr.Blocks(
        title="FC Decision Lab",
        theme=gr.themes.Base(),
        css=CSS_STRING,
    ) as demo:
        gr.HTML(
            "<div class='fc-panel' style='border:none'>"
            "<h1 style='color:#f0f1f3;font-size:1.5rem;font-weight:700;margin:0 0 6px'>"
            "FC decision lab</h1>"
            "<p class='fc-muted' style='margin:0;font-size:0.95rem'>"
            "Reveal low- or high-cost clues, then commit or skip before you run out of steps.</p></div>"
        )

        with gr.Tabs():
            with gr.Tab("▶️ Play"):
                st = gr.State()  # type: ignore[var-annotated]  # { "env", "pre_obs" }

                flow = gr.HTML(
                    "<p class='fc-muted' style='margin:0'>Start a new episode to begin.</p>"
                )

                with gr.Row():
                    low_counter = gr.Markdown(
                        COUNTER_INACTIVE[0],
                        elem_classes=["fc-counter-wrap"],
                        sanitize_html=False,
                    )
                    high_counter = gr.Markdown(
                        COUNTER_INACTIVE[1],
                        elem_classes=["fc-counter-wrap"],
                        sanitize_html=False,
                    )

                with gr.Row():
                    b_low = gr.Button("🔍 Reveal Low-cost Clue", interactive=False)
                    b_high = gr.Button("💎 Reveal High-cost Clue", interactive=False)
                with gr.Row():
                    b_skip = gr.Button("❌ Skip Candidate", interactive=False, variant="stop")
                    b_commit = gr.Button("✅ Commit Decision", interactive=False)

                b_reset = gr.Button("Start new episode", variant="primary")

                _welcome = (
                    "<p class='fc-muted' style='margin:0'>"
                    "Press <strong>Start new episode</strong> to play.</p>"
                )
                clues = gr.HTML(_welcome)
                token_block = gr.HTML("<p class='fc-muted' style='margin:0'>—</p>")
                last_block = gr.HTML(
                    "<div class='fc-panel'><p class='fc-muted' style='margin:0'>"
                    "No step yet</p></div>"
                )

                def on_start() -> tuple:
                    e = FCEnvEnvironment()
                    o = e.reset()
                    snap = _snapshot(o)
                    s0: dict = {"env": e, "pre_obs": snap}
                    lo, hi = _counter_markdown_pair(o)
                    bup = _button_state_from_obs(o)
                    return (
                        s0,
                        _clue_cards_html(o.revealed_clues, None),
                        _token_bar_html(o),
                        _last_step_html(o, "Episode started — your move."),
                        _flow_badge(o, None),
                        lo,
                        hi,
                    ) + bup

                def on_step(
                    s: dict | None,
                    user_action: int,
                ) -> tuple:
                    if not s or s.get("env") is None:
                        idle = (gr.update(),) * 10
                        return (s,) + idle

                    e: FCEnvEnvironment = s["env"]  # type: ignore[assignment]
                    pre_dict = s.get("pre_obs")
                    if not isinstance(pre_dict, dict):
                        return on_start()
                    pre_clues: tuple = tuple(pre_dict.get("revealed_clues", ()))
                    o = e.step(Action(action=user_action))
                    if len(pre_clues) < 6:
                        new_idx = None
                    else:
                        new_idx = _newly_revealed_index(pre_clues, o.revealed_clues)
                    out = _snapshot(o)
                    next_s: dict = {"env": e, "pre_obs": out}
                    bup = _button_state_from_obs(o)
                    lo, hi = _counter_markdown_pair(o)
                    return (
                        next_s,
                        _clue_cards_html(o.revealed_clues, new_idx),
                        _token_bar_html(o),
                        _last_step_html(
                            o,
                            _outcome_text(user_action, pre_dict, o, new_idx),
                        ),
                        _flow_badge(o, user_action),
                        lo,
                        hi,
                    ) + bup

                b_reset.click(
                    on_start,
                    inputs=None,
                    outputs=[
                        st,
                        clues,
                        token_block,
                        last_block,
                        flow,
                        low_counter,
                        high_counter,
                        b_low,
                        b_high,
                        b_skip,
                        b_commit,
                    ],
                )
                b_low.click(
                    lambda s: on_step(s, 0),
                    inputs=[st],
                    outputs=[
                        st,
                        clues,
                        token_block,
                        last_block,
                        flow,
                        low_counter,
                        high_counter,
                        b_low,
                        b_high,
                        b_skip,
                        b_commit,
                    ],
                )
                b_high.click(
                    lambda s: on_step(s, 1),
                    inputs=[st],
                    outputs=[
                        st,
                        clues,
                        token_block,
                        last_block,
                        flow,
                        low_counter,
                        high_counter,
                        b_low,
                        b_high,
                        b_skip,
                        b_commit,
                    ],
                )
                b_commit.click(
                    lambda s: on_step(s, 2),
                    inputs=[st],
                    outputs=[
                        st,
                        clues,
                        token_block,
                        last_block,
                        flow,
                        low_counter,
                        high_counter,
                        b_low,
                        b_high,
                        b_skip,
                        b_commit,
                    ],
                )
                b_skip.click(
                    lambda s: on_step(s, 3),
                    inputs=[st],
                    outputs=[
                        st,
                        clues,
                        token_block,
                        last_block,
                        flow,
                        low_counter,
                        high_counter,
                        b_low,
                        b_high,
                        b_skip,
                        b_commit,
                    ],
                )

            with gr.Tab("📊 Compare"):
                with gr.Group(elem_classes="content-card"):
                    gr.Markdown(
                        '<div class="section-title">Trained vs random</div>\n\n'
                        "The environment supports **policy evaluation**: a **random** policy "
                        "picks a legal action at random. A **trained** policy (e.g. tabular Q-learning "
                        "or PPO in `train.py`) optimizes for cumulative reward and win rate against that "
                        "baseline.\n\n"
                        "Training writes metrics under `artifacts/`. Run `python train.py` locally to "
                        "reproduce curves. The **API** here stays the same; only the **policy** changes.",
                        sanitize_html=False,
                    )

            with gr.Tab("ℹ️ About"):
                with gr.Group(elem_classes="content-card"):
                    gr.Markdown(
                        '<div class="section-title">About this lab</div>\n\n'
                        "You manage a **token budget** and choose **low-cost** versus **high-cost** clues. "
                        "Reveals fill the six **Clues** cards; hidden slots show **???** until you pay to "
                        "reveal them. When you **commit** or **skip**, the episode ends; rewards reflect "
                        "how well you read the situation. This interface uses the same **step** logic as "
                        "the Space **API** — the front end is only here for a clear, decision-first "
                        "experience.",
                        sanitize_html=False,
                    )

    return demo
