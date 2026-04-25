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
  background-color: #0b0f14 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}

.gradio-container, .gradio-container.fillable {
  background-color: #0b0f14 !important;
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
  background: #0b0f14 !important;
  color: #ffffff !important;
  opacity: 1 !important;
  border: none !important;
}
.gr-html, .gr-html * {
  color: #ffffff;
}
.gr-html .fc-muted, .prose .fc-muted { color: #cccccc !important; }
main, main.contain, .form, .form > div {
  background: #0b0f14 !important;
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

/* Header (game) */
.header-card, .gr-group.header-card, div.header-card {
  background: linear-gradient(135deg, #111827, #1f2937) !important;
  padding: 20px 22px !important;
  border-radius: 14px !important;
  border: 1px solid #2a2f3a !important;
  margin-bottom: 12px !important;
  box-shadow: 0 8px 32px -12px rgba(0, 0, 0, 0.55) !important;
}
.header-card .prose, .header-card h1, .header-card p { color: #ffffff !important; }

/* Clue cell (6 separate markdowns; inner HTML) */
.clue-v2-inner {
  display: block;
  background: #141922;
  border: 1px solid #2a2f3a;
  border-radius: 12px;
  padding: 20px 14px;
  min-height: 100px;
  text-align: center;
  position: relative;
  overflow: hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.clue-v2-inner::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(60, 120, 255, 0.04), transparent 50%);
  pointer-events: none;
  opacity: 0.5;
}
.clue-v2-inner.revealed {
  background: #1f2937;
  border-color: #4b5563;
}
.clue-v2-inner.just-revealed {
  border-color: #00a651;
  box-shadow: 0 0 0 1px #00a651, 0 6px 20px -6px rgba(0, 166, 81, 0.35);
  animation: clue-pop 0.5s ease 1;
}
@keyframes clue-pop {
  from { transform: scale(0.98); opacity: 0.9; }
  to { transform: scale(1); opacity: 1; }
}
.clue-v2-t { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 8px; position: relative; z-index: 1; }
.clue-v2-b { font-size: 1.05rem; color: #f3f4f6; font-weight: 600; line-height: 1.3; position: relative; z-index: 1; }
.clue-v2-b.hid { color: #4b5563; font-size: 1.25rem; letter-spacing: 0.15em; }

.clue-md { margin: 0 !important; }
.clue-md .prose { min-height: 0 !important; }
button.gr-button.btn-primary, .btn-primary, .gr-button.btn-reveal.btn-primary {
  min-height: 48px;
  background: #374151 !important;
  border-color: #4b5563 !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border-radius: 10px !important;
}
button.gr-button.btn-secondary, .gr-button.btn-reveal.btn-secondary {
  min-height: 48px;
  background: #1f2937 !important;
  border-color: #3d4a5c !important;
  color: #f9fafb !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border-radius: 10px !important;
}
button.gr-button.btn-danger, .btn-danger {
  min-height: 48px;
  background: #7f1d1d !important;
  border-color: #b91c1c !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border-radius: 10px !important;
}
button.gr-button.btn-success, .btn-success {
  min-height: 48px;
  background: #065f46 !important;
  border-color: #059669 !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border-radius: 10px !important;
}
button.gr-button.btn-cta {
  min-height: 50px;
  background: #1e3a5f !important;
  border-color: #2d5a8a !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  border-radius: 12px !important;
  width: 100% !important;
  max-width: 100% !important;
}
.counter-bar, .gr-markdown.counter-bar, .gr-markdown.counter-bar .prose { font-size: 1rem; margin: 0 !important; line-height: 1.5; }
.counter-bar p { margin: 0; }

/* Read-only token slider: looks like a bar */
.form .token-slider input[type=range] {
  height: 8px; border-radius: 4px;
  accent-color: #059669;
}
.form .token-slider, .form .token-slider label { color: #e5e7eb !important; }

/* Episode log (scrollable) */
.history-panel, .history-box, .history-box .prose, .history-panel .prose {
  max-height: 280px !important;
  overflow-y: auto !important;
  padding: 12px 14px !important;
  margin-top: 0 !important;
  background: #0f172a !important;
  color: #e5e7eb !important;
  border: 1px solid #2a2f3a !important;
  border-radius: 12px !important;
  font-size: 0.88rem;
  line-height: 1.5;
  opacity: 1 !important;
}
.history-box .prose p, .history-box .prose li, .history-panel .prose p { color: #e5e7eb !important; }
.history-box h2, .history-panel h2, .history-panel h2 { color: #ffffff !important; }
.last-step-card { border: 1px solid #2a2f3a; border-radius: 12px; }
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


def _clue_cell_value(
    index: int,
    raw: str,
    highlight_index: int | None,
) -> str:
    tier = _tier_name(index)
    hidden = raw == HIDDEN
    body = _format_clue_line(raw) if not hidden else "???"
    rcls = "clue-v2-inner"
    if not hidden:
        rcls += " revealed"
    if highlight_index == index and not hidden:
        rcls += " just-revealed"
    body_cls = "clue-v2-b hid" if hidden else "clue-v2-b"
    return (
        f'<div class="{rcls}">'
        f'<div class="clue-v2-t">Clue {index + 1} · {tier}</div>'
        f'<div class="{body_cls}">{body}</div>'
        f"</div>"
    )


def _six_clue_values(
    revealed: tuple[str, ...] | list[str],
    highlight: int | None,
) -> tuple[str, str, str, str, str, str]:
    if not revealed or len(revealed) < 6:
        ph = _clue_cell_value(0, HIDDEN, None)
        return (ph, ph, ph, ph, ph, ph)
    return tuple(
        _clue_cell_value(i, revealed[i], highlight)  # type: ignore[index]
        for i in range(6)
    )


COUNTER_BAR_IDLE = (
    '<div class="counter-g">'
    '<p class="counter-line" style="margin:0">'
    "Low clues: <span class=\"c-mute\" style=\"color:#6b7280\">—</span> · High clues: "
    '<span class="c-mute" style="color:#6b7280">—</span></p>'
    '<p style="margin:6px 0 0;font-size:0.8rem;color:#6b7280">Start a new episode to play.</p>'
    "</div>"
)


def _counter_bar_value(o) -> str:
    lo, hi = o.low_remaining, o.high_remaining
    cl = "#b91c1c" if lo == 0 else "#00a651"
    ch = "#b91c1c" if hi == 0 else "#00a651"
    if lo == 0 and hi == 0:
        hint = '<p style="margin:8px 0 0;font-size:0.85rem;color:#fca5a5">No reveals left. Use Commit or Skip.</p>'
    elif lo == 0:
        hint = (
            '<p style="margin:8px 0 0;font-size:0.85rem;color:#fca5a5">'
            "No low-cost clues remaining.</p>"
        )
    elif hi == 0:
        hint = (
            '<p style="margin:8px 0 0;font-size:0.85rem;color:#fca5a5">'
            "No high-cost clues remaining.</p>"
        )
    else:
        hint = ""
    return (
        '<div class="counter-g">'
        f'<p class="counter-line" style="margin:0">'
        f'Low clues: <strong style="color:{cl}">{lo}</strong> · '
        f'High clues: <strong style="color:{ch}">{hi}</strong></p>{hint}</div>'
    )


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
        f"<div class='fc-panel last-step-card'>"
        f"<p class='fc-h2' style='margin:0 0 10px'>Last step result</p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Step reward</span> "
        f"<span class='{cls}'>{rc}</span></p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Tokens left</span> {o.tokens} / {MAX_TOKENS}</p>"
        f"<p style='margin:4px 0'><span class='fc-muted'>Step</span> {o.step_number}</p>"
        f"<p style='margin:8px 0 0' class='fc-muted'>{outcome}</p>"
        f"</div>"
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


# Episode log (0–3 = env Action values; matches buttons in UI)
_ACTION_NAMES: dict[int, str] = {
    0: "Reveal low",
    1: "Reveal high",
    2: "Commit",
    3: "Skip",
}

HISTORY_STATE_INIT: dict = {"lines": [], "cum": 0.0, "stale": True}


def _action_name(action: int) -> str:
    return _ACTION_NAMES.get(action, f"Action {action}")


def _cumulative_assessment(cum: float) -> str:
    if cum > 0.05:
        return "Cumulative reward is **positive** on this run (sum of all step rewards)."
    if cum < -0.05:
        return "Cumulative reward is **negative** on this run (sum of all step rewards)."
    return "Cumulative reward is about **neutral** (sum of all step rewards)."


def _log_after_step(
    h: dict | None,
    user_action: int,
    o,
) -> dict:
    prev: dict = {**HISTORY_STATE_INIT, **(h or {})}
    lines: list[str] = list(prev.get("lines", []))
    prior_cum = float(prev.get("cum", 0.0))
    cum = prior_cum + float(o.reward)
    an = _action_name(user_action)
    step_entry = f"""**Step {o.step_number}**
- **Action:** {an}
- **Step reward:** {o.reward:+.4f}
- **Tokens left:** {o.tokens}
- **Low left:** {o.low_remaining}
- **High left:** {o.high_remaining}
- **Done (after this step):** {o.done}"""
    lines.append(step_entry)
    if o.done:
        if user_action == 2:
            end = "Commit (terminal action)"
        elif user_action == 3:
            end = "Skip (terminal action)"
        else:
            end = f"{an} or system limit (tokens, clues, max steps, or all revealed)"
        final = f"""## Final result
- **Total steps (env step number):** {o.step_number}
- **Final tokens:** {o.tokens}
- **Last step reward:** {o.reward:+.4f}
- **Cumulative reward (episode sum):** {cum:+.4f}
- **End driver:** {end}
- **Outcome (cumulative return):** {_cumulative_assessment(cum)}"""
        lines.append(final)
    return {"lines": lines, "cum": cum, "stale": False}


def _log_to_markdown(h: dict | None) -> str:
    h = h or HISTORY_STATE_INIT
    if h.get("stale", True) and not h.get("lines"):
        return "### Episode history\n\n_No actions yet. Start a new episode to play._\n"
    if not h.get("lines"):
        return "### Episode history\n\n_New episode — steps will be logged as you act._\n"
    return "### Episode history\n\n" + "\n\n---\n\n".join(h["lines"]) + "\n"


def build_blocks() -> gr.Blocks:
    with gr.Blocks(
        title="FC Decision Lab",
        theme=gr.themes.Base(),
        css=CSS_STRING,
    ) as demo:
        with gr.Tabs():
            with gr.Tab("Play"):
                st = gr.State()  # type: ignore[var-annotated]  # { "env", "pre_obs" }
                history_state = gr.State(HISTORY_STATE_INIT)  # type: ignore[var-annotated]

                with gr.Group(elem_classes="header-card"):
                    gr.Markdown("# FC Decision Lab")
                    gr.Markdown(
                        "Reveal clues strategically. **Commit** or **Skip** before your token budget runs out."
                    )

                b_reset = gr.Button("Start new episode", elem_classes=["btn-cta"])

                _ph = _clue_cell_value(0, HIDDEN, None)
                with gr.Row():
                    clue0 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)
                    clue1 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)
                    clue2 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)
                with gr.Row():
                    clue3 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)
                    clue4 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)
                    clue5 = gr.Markdown(_ph, elem_classes=["clue-md"], sanitize_html=False)

                with gr.Row():
                    b_low = gr.Button(
                        "Reveal low",
                        interactive=False,
                        elem_classes=["btn-reveal", "btn-primary"],
                    )
                    b_high = gr.Button(
                        "Reveal high",
                        interactive=False,
                        elem_classes=["btn-reveal", "btn-secondary"],
                    )
                with gr.Row():
                    b_skip = gr.Button(
                        "Skip",
                        interactive=False,
                        elem_classes=["btn-danger"],
                    )
                    b_commit = gr.Button(
                        "Commit",
                        interactive=False,
                        elem_classes=["btn-success"],
                    )

                counter_bar = gr.Markdown(
                    COUNTER_BAR_IDLE, elem_classes=["counter-bar"], sanitize_html=False
                )
                token_slider = gr.Slider(
                    0,
                    MAX_TOKENS,
                    value=MAX_TOKENS,
                    label="Token budget (remaining)",
                    interactive=False,
                    show_label=True,
                    elem_classes=["token-slider"],
                )
                last_block = gr.HTML(
                    "<div class='fc-panel last-step-card'>"
                    "<p class='fc-h2' style='margin:0 0 8px'>Last step</p>"
                    "<p class='fc-muted' style='margin:0'>No action yet. Start a new episode.</p></div>"
                )
                flow = gr.HTML(
                    "<p class='fc-muted' style='margin:0;font-size:0.9rem'>"
                    "Press **Start** to begin.</p>"
                )
                history_display = gr.Markdown(
                    _log_to_markdown(HISTORY_STATE_INIT),
                    elem_classes=["history-box", "history-panel"],
                )

                _out_play = [
                    st,
                    history_state,
                    clue0,
                    clue1,
                    clue2,
                    clue3,
                    clue4,
                    clue5,
                    last_block,
                    flow,
                    counter_bar,
                    token_slider,
                    history_display,
                    b_low,
                    b_high,
                    b_skip,
                    b_commit,
                ]

                def on_start() -> tuple:
                    e = FCEnvEnvironment()
                    o = e.reset()
                    snap = _snapshot(o)
                    s0: dict = {"env": e, "pre_obs": snap}
                    h0: dict = {"lines": [], "cum": 0.0, "stale": False}
                    h_md = _log_to_markdown(h0)
                    cells = _six_clue_values(o.revealed_clues, None)
                    bup = _button_state_from_obs(o)
                    return (
                        s0,
                        h0,
                        *cells,
                        _last_step_html(o, "Episode started — your move."),
                        _flow_badge(o, None),
                        _counter_bar_value(o),
                        gr.update(value=o.tokens),
                        h_md,
                    ) + bup

                def on_step(
                    s: dict | None,
                    user_action: int,
                    hist: dict | None,
                ) -> tuple:
                    if not s or s.get("env") is None:
                        return (s,) + (gr.update(),) * 16

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
                    next_s: dict = {"env": e, "pre_obs": _snapshot(o)}
                    bup = _button_state_from_obs(o)
                    cells = _six_clue_values(o.revealed_clues, new_idx)
                    h_new = _log_after_step(hist, user_action, o)
                    h_md = _log_to_markdown(h_new)
                    return (
                        next_s,
                        h_new,
                        *cells,
                        _last_step_html(
                            o,
                            _outcome_text(user_action, pre_dict, o, new_idx),
                        ),
                        _flow_badge(o, user_action),
                        _counter_bar_value(o),
                        gr.update(value=o.tokens),
                        h_md,
                    ) + bup

                b_reset.click(
                    on_start,
                    inputs=None,
                    outputs=_out_play,
                )
                b_low.click(
                    lambda s, h: on_step(s, 0, h),
                    inputs=[st, history_state],
                    outputs=_out_play,
                )
                b_high.click(
                    lambda s, h: on_step(s, 1, h),
                    inputs=[st, history_state],
                    outputs=_out_play,
                )
                b_commit.click(
                    lambda s, h: on_step(s, 2, h),
                    inputs=[st, history_state],
                    outputs=_out_play,
                )
                b_skip.click(
                    lambda s, h: on_step(s, 3, h),
                    inputs=[st, history_state],
                    outputs=_out_play,
                )

            with gr.Tab("Compare"):
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

            with gr.Tab("About"):
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
