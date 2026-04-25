"""Gradio UI for the HF Space (mounted at /ui)."""

from __future__ import annotations

import ast
import gradio as gr

from environment import FCEnvEnvironment
from models import Action

# Match reset() in FCEnvEnvironment (for token bar; backend unchanged)
MAX_TOKENS = 100
HIDDEN = "HIDDEN"

# Game shell: dark + neon; forced dark for consistent "premium" look on all systems
CSS_STRING = r"""
/* --- FC Decision Lab: game skin --- */
html { color-scheme: dark !important; }

body, .gradio-container, .gradio-container.fillable, main, main.contain, .form, .form > div {
  background-color: #0b0f14 !important;
  color: #ffffff !important;
  opacity: 1 !important;
}

footer, .footer, [class*="footer"] { display: none !important; }

h1, h2, h3, h4 { color: #ffffff !important; }

/* Tabs */
button[role="tab"], [class*="tab-nav"], .tabitem {
  color: #e5e7eb !important;
  border-color: #2a3440 !important;
}

/* Markdown / prose */
.markdown, .prose, [class*="markdown"] {
  color: #e5e7eb !important;
  opacity: 1 !important;
}
.markdown p, .prose p, .prose li { color: #e5e7eb !important; line-height: 1.55 !important; }
.markdown code, .prose code {
  background: #1a222c !important;
  border: 1px solid #2a3440 !important;
  color: #e5e7eb !important;
  border-radius: 6px;
  padding: 2px 6px;
}
.label-wrap, label, [data-testid] label { color: #9ca3af !important; }

/* Primary layout blocks */
.content-card, .gr-group.content-card, div.content-card {
  background: linear-gradient(180deg, #111820 0%, #0b0f14 100%) !important;
  border: 1px solid #1f2a35 !important;
  border-radius: 14px !important;
  padding: 20px 22px !important;
  margin-top: 12px !important;
  box-shadow:
    0 0 0 1px rgba(0, 212, 255, 0.06) inset,
    0 12px 40px -20px #000a;
}
.content-card .prose, .content-card [class*="markdown"] { background: transparent !important; }
.section-title {
  font-size: 1.05rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #00d4ff !important;
  margin: 0 0 12px 0;
  border-bottom: 1px solid rgba(0, 212, 255, 0.2);
  padding-bottom: 10px;
}

/* Gradio form rows */
.gr-form { background: #0b0f14 !important; border: none !important; }

/* === Buttons: game CTA (override Gradio) === */
button, .gr-button, button.gr-button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  transition: transform 0.12s ease, box-shadow 0.2s ease, background 0.2s ease, border-color 0.2s !important;
  opacity: 1 !important;
}
button:active:not(:disabled), .gr-button:active:not(:disabled) {
  transform: scale(0.97) !important;
}
button:disabled, .gr-button:disabled, button.gr-button:disabled {
  cursor: not-allowed !important;
  opacity: 0.5 !important;
  filter: grayscale(0.25) !important;
  box-shadow: none !important;
  transform: none !important;
}

/* Start episode */
button.gr-button.fc-btn--start, .gr-button.fc-btn--start {
  min-height: 50px;
  width: 100% !important;
  max-width: 100% !important;
  background: linear-gradient(180deg, #1e2a36, #1a222c) !important;
  border: 1px solid #00d4ff !important;
  color: #ffffff !important;
  box-shadow: 0 0 20px -6px #00d4ff, 0 4px 18px -8px #000;
}
button.gr-button.fc-btn--start:hover:enabled, .gr-button.fc-btn--start:hover:enabled {
  transform: scale(1.02) !important;
  box-shadow: 0 0 28px -4px #00d4ff, 0 8px 24px -10px #000a !important;
}

/* Reveal low — cyan */
button.gr-button.fc-btn--low, .gr-button.fc-btn--low {
  min-height: 50px;
  background: #141c24 !important;
  color: #e0f7ff !important;
  border: 1px solid #00d4ff !important;
  box-shadow: 0 0 18px -8px #00d4ff;
  -webkit-text-fill-color: #e0f7ff !important;
}
button.gr-button.fc-btn--low:hover:enabled, .gr-button.fc-btn--low:hover:enabled {
  transform: scale(1.03) !important;
  background: #16212c !important;
  box-shadow: 0 0 26px -4px #00d4ff !important;
  border-color: #33ddff !important;
}

/* Reveal high — orange (no purple; warm accent) */
button.gr-button.fc-btn--high, .gr-button.fc-btn--high {
  min-height: 50px;
  background: #1c1810 !important;
  color: #fff4e0 !important;
  border: 1px solid #d97706 !important;
  box-shadow: 0 0 18px -8px rgba(245, 197, 66, 0.65);
  -webkit-text-fill-color: #fff4e0 !important;
}
button.gr-button.fc-btn--high:hover:enabled, .gr-button.fc-btn--high:hover:enabled {
  transform: scale(1.03) !important;
  background: #221a0e !important;
  box-shadow: 0 0 26px -2px #f5c54255 !important;
  border-color: #f5c542 !important;
}

/* Refresh — neutral */
button.gr-button.fc-btn--refresh, .gr-button.fc-btn--refresh {
  min-height: 50px;
  background: #151820 !important;
  color: #d1d5db !important;
  border: 1px solid #3d4a5c !important;
  box-shadow: 0 0 12px -6px #2a3340;
  -webkit-text-fill-color: #d1d5db !important;
}
button.gr-button.fc-btn--refresh:hover:enabled, .gr-button.fc-btn--refresh:hover:enabled {
  transform: scale(1.03) !important;
  border-color: #5c6b80 !important;
  box-shadow: 0 0 18px -4px #4b5563cc !important;
}

/* Commit — green */
button.gr-button.fc-btn--commit, .gr-button.fc-btn--commit {
  min-height: 50px;
  background: #0a1f16 !important;
  color: #c8ffe8 !important;
  border: 1px solid #00ff88 !important;
  box-shadow: 0 0 20px -8px #00ff88a0;
  -webkit-text-fill-color: #c8ffe8 !important;
}
button.gr-button.fc-btn--commit:hover:enabled, .gr-button.fc-btn--commit:hover:enabled {
  transform: scale(1.03) !important;
  background: #0c261b !important;
  box-shadow: 0 0 28px -2px #00ff88bb !important;
  border-color: #33ff99 !important;
}

/* Spacing for 2x2 action grid */
.fc-actions-row { gap: 12px !important; margin: 0 !important; }
.fc-actions-row .gr-block { min-width: 0 !important; }

/* === HTML: header === */
.fc-hgame { margin-bottom: 4px; }
.fc-game-header h1 {
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: 0.02em;
  margin: 0 0 8px 0;
  color: #ffffff !important;
  text-shadow: 0 0 24px rgba(0, 212, 255, 0.15);
}
.fc-game-header .fc-sub {
  color: #9ca3af !important;
  font-size: 0.98rem;
  margin: 0;
  line-height: 1.45;
}

/* === 6 card grid: flip + glow === */
.fc-clue-arena { max-width: 900px; margin: 0 auto; padding: 4px 0 8px; }
.fc-clue-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}
@media (max-width: 700px) {
  .fc-clue-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }
}
@media (max-width: 420px) {
  .fc-clue-grid { grid-template-columns: 1fr; }
}

.fc-card-scene {
  perspective: 1000px;
  min-height: 148px;
  border-radius: 16px;
}
.fc-card-scene--small { min-height: 132px; }

.fc-card-inner {
  position: relative;
  width: 100%;
  min-height: 148px;
  transition: transform 0.5s ease, box-shadow 0.25s ease, filter 0.25s ease;
  transform-style: preserve-3d;
  border-radius: 16px;
  cursor: default;
}
.fc-card-scene--small .fc-card-inner { min-height: 132px; }

/* Hidden: not flipped */
.fc-card-scene.is-hidden .fc-card-inner { transform: rotateY(0deg); }
/* Revealed: show answer side */
.fc-card-scene.is-revealed .fc-card-inner { transform: rotateY(180deg); }
/* Newly revealed in this response: play flip in from 0 */
.fc-card-scene.is-revealed.is-new .fc-card-inner {
  transform: rotateY(0deg);
  animation: fc-reveal-flip 0.55s ease forwards;
}
@keyframes fc-reveal-flip {
  from { transform: rotateY(0deg); }
  to { transform: rotateY(180deg); }
}

/* Pulse when just revealed (after flip) */
.fc-card-scene.is-new .fc-card-face--front {
  box-shadow: 0 0 0 1px #00ff88, 0 0 32px -8px #00ff8866, inset 0 1px 0 rgba(255,255,255,0.06) !important;
  animation: fc-pulse 0.65s ease 1;
}
@keyframes fc-pulse {
  from { filter: brightness(1.2); }
  to { filter: brightness(1); }
}

.fc-card-scene.is-hidden:hover .fc-card-inner {
  transform: rotateY(0deg) scale(1.05);
  filter: brightness(1.08);
}
.fc-card-scene.is-hidden .fc-card-face--back {
  box-shadow: 0 0 0 1px #f5c54244, 0 8px 32px -12px #f5c54255, 0 0 40px -20px #f5c54233;
}

.fc-card-face {
  position: absolute;
  inset: 0;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 12px 14px;
  box-sizing: border-box;
  border: 1px solid #2a3440;
  overflow: hidden;
}

.fc-card-face--back {
  z-index: 2;
  background: linear-gradient(160deg, #2a2010 0%, #f5c542 30%, #c9a028 100%);
  color: #1a1206;
  border: 1px solid #f0d36a;
}
.fc-card-question {
  font-size: 2.25rem;
  font-weight: 900;
  letter-spacing: 0.1em;
  line-height: 1;
  text-shadow: 0 2px 0 #0002;
  color: #2b1f08;
}
.fc-card-badge {
  position: absolute;
  top: 8px; left: 8px;
  font-size: 0.62rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  padding: 4px 8px;
  border-radius: 8px;
  z-index: 3;
  border: 1px solid;
}
.fc-badge--low {
  color: #b6f0ff;
  background: rgba(0, 0, 0, 0.2);
  border-color: #00d4ff88;
  text-shadow: 0 0 8px #00d4ff88;
  box-shadow: 0 0 12px -4px #00d4ff;
}
.fc-badge--high {
  color: #ffe4c4;
  background: rgba(0,0,0,0.2);
  border-color: #f5c54288;
  text-shadow: 0 0 8px #d97706aa;
  box-shadow: 0 0 12px -4px #d97706;
}

.fc-card-face--front {
  transform: rotateY(180deg);
  background: linear-gradient(200deg, #1a222c, #0f1318) !important;
  border: 1px solid #2f3d4d;
  z-index: 1;
  box-shadow: inset 0 1px 0 #ffffff0a, 0 0 0 0 transparent;
  align-items: flex-start;
  text-align: left;
}
.fc-card-lbl {
  font-size: 0.64rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #9ca3af !important;
  margin-bottom: 6px;
  text-shadow: none;
  width: 100%;
  word-wrap: break-word;
}
.fc-card-val {
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.4;
  color: #e8fbff !important;
  width: 100%;
  text-shadow: 0 0 20px #00d4ff22;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* Footer strip: counters + token */
.fc-play-footer { margin-top: 8px; }
.fc-counter-row { margin-bottom: 10px; font-size: 0.95rem; color: #d1d5db !important; }
.fc-counter-row strong { font-weight: 700; }
.fc-mute { color: #9ca3af !important; }
.fc-hint-amber { color: #fca5a5 !important; font-size: 0.86rem; margin: 6px 0 0; }

.fc-token-outer { margin-top: 4px; }
.fc-token-label { font-size: 0.95rem; margin: 0 0 6px; color: #e5e7eb; font-weight: 600; }
.fc-token-track {
  position: relative;
  height: 10px;
  width: 100%;
  background: #0a0d10;
  border: 1px solid #1f2a32;
  border-radius: 999px;
  overflow: hidden;
  box-shadow: inset 0 1px 0 #0006;
}
.fc-token-fill {
  height: 100%;
  width: 0%;
  min-width: 0;
  max-width: 100%;
  border-radius: 999px;
  transition: width 0.45s ease, background 0.35s ease, box-shadow 0.3s;
  background: #00ff88;
  box-shadow: 0 0 20px -4px #00ff8866, inset 0 1px 0 #fff3;
}
.fc-token-fill--low {
  background: linear-gradient(90deg, #ff4d4d, #f5c542) !important;
  box-shadow: 0 0 18px -2px #ff4d4d99;
}
.fc-token-fill--mid {
  background: linear-gradient(90deg, #f5c542, #d4a017) !important;
  box-shadow: 0 0 16px -2px #f5c54288;
}
.fc-token-fill--hi {
  background: linear-gradient(90deg, #1a7a50, #00ff88) !important;
  box-shadow: 0 0 18px -2px #00ff8899;
}

/* Last action card + flow */
.fc-card-log {
  background: #121a22;
  border: 1px solid #243040;
  border-radius: 14px;
  padding: 16px 18px;
  margin: 0;
  box-shadow: 0 0 0 1px #00d4ff0d inset, 0 6px 24px -12px #0008;
}
.fc-card-log h3 {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #6b7c8a !important;
  margin: 0 0 8px; font-weight: 700;
}
.fc-flow-line { margin: 0; font-size: 0.88rem; color: #9ca3af; line-height: 1.4; }
.fc-reward-pos { color: #00ff88 !important; font-weight: 700; }
.fc-reward-neg { color: #ff4d4d !important; font-weight: 700; }
.fc-reward-neu { color: #d1d5db !important; font-weight: 600; }
.fc-oneline-log { font-size: 0.95rem; color: #e5e7eb; margin: 0 0 6px; line-height: 1.45; }

/* History cards */
.gr-accordion, details.gr-accordion { background: #0b0f14 !important; border: none; }
summary { color: #e5e7eb; font-weight: 700; letter-spacing: 0.02em; }
.fc-history-entries { max-height: 300px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding: 2px; }
.fc-hist-card {
  background: #0f141a;
  border: 1px solid #1f2a32;
  border-left: 3px solid #00d4ff;
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 0.88rem;
  color: #e5e7eb;
  line-height: 1.45;
}
.fc-hist-card--final { border-left-color: #f5c542; background: #141008; }
.fc-hist-st { color: #6b7c8a; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; display: block; margin-bottom: 4px; }
"""

STATIC_HEADER = """
<div class="fc-game-header">
  <h1>FC Decision Lab</h1>
  <p class="fc-sub">Reveal clues strategically. Spend tokens wisely.</p>
</div>
"""


def _clue_label_value(raw: str) -> tuple[str, str, bool]:
    """(label, value, hidden) for card face."""
    if raw == HIDDEN:
        return ("", "???", True)
    try:
        t = ast.literal_eval(raw)
        if isinstance(t, (list, tuple)) and len(t) == 2:
            k, v = t
            label = str(k).replace("_", " ").strip().upper()
            return (label, str(v).strip(), False)
    except (ValueError, SyntaxError, TypeError):
        pass
    return ("CLUE", str(raw), False)


def _tier_badge_class(idx: int) -> str:
    return "fc-badge--low" if idx < 3 else "fc-badge--high"


def _tier_label(idx: int) -> str:
    return "LOW" if idx < 3 else "HIGH"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_six_clues(
    revealed: tuple[str, ...] | list[str] | None,
    highlight: int | None,
) -> str:
    if not revealed or len(revealed) < 6:
        revealed = [HIDDEN] * 6
    out = ['<div class="fc-clue-arena"><div class="fc-clue-grid">']
    for i in range(6):
        raw = revealed[i]  # type: ignore[index]
        label, value, hidden = _clue_label_value(raw)
        bcls = _tier_badge_class(i)
        bt = _tier_label(i)
        if hidden:
            state = "is-hidden"
            newc = ""
            inner = (
                f'<div class="fc-card-face fc-card-face--back">'
                f'<span class="fc-card-badge {bcls}">{bt}</span>'
                f'<div class="fc-card-question">???</div></div>'
            )
        else:
            newc = " is-new" if (highlight is not None and i == highlight) else ""
            state = f"is-revealed{newc}"
            vhtml = _html_escape(value)
            lhtml = _html_escape(label)
            inner = (
                f'<div class="fc-card-face fc-card-face--back">'
                f'<span class="fc-card-badge {bcls}">{bt}</span>'
                f'<div class="fc-card-question">???</div></div>'
                f'<div class="fc-card-face fc-card-face--front">'
                f'<span class="fc-card-badge {bcls}">{bt}</span>'
                f'<div class="fc-card-lbl">{lhtml}</div>'
                f'<div class="fc-card-val">{vhtml}</div></div>'
            )
        out.append(
            f'<div class="fc-card-scene {state}">'
            f'<div class="fc-card-inner">{inner}</div></div>'
        )
    out.append("</div></div>")
    return "".join(out)


COUNTER_FOOTER_IDLE = (
    "<div class='fc-play-footer'>"
    "<div class='fc-counter-row'>"
    "Low: <span class='fc-mute' style='font-weight:600;'>—</span> &nbsp;·&nbsp; "
    "High: <span class='fc-mute' style='font-weight:600;'>—</span></div>"
    "<p class='fc-mute' style='margin:0 0 8px;font-size:0.86rem;'>"
    "Start a new episode to play.</p>"
    "<div class='fc-token-outer'>"
    "<p class='fc-token-label' style='color:#6b7280'>"
    f"Tokens: 0 / {MAX_TOKENS}</p><div class='fc-token-track'>"
    "<div class='fc-token-fill' style='width:0%'></div></div></div></div>"
)


def _token_bar_fill_class(pct: float) -> str:
    if pct >= 55.0:
        return "fc-token-fill fc-token-fill--hi"
    if pct >= 25.0:
        return "fc-token-fill fc-token-fill--mid"
    return "fc-token-fill fc-token-fill--low"


def _play_footer_html(o) -> str:
    lo, hi = o.low_remaining, o.high_remaining
    cl = "#ff4d4d" if lo == 0 else "#00d4ff"
    ch = "#ff4d4d" if hi == 0 else "#f5c542"
    if lo == 0 and hi == 0:
        hint = (
            "<p class='fc-hint-amber' style='margin:8px 0 0'>"
            "No reveals left. Use <strong>Commit</strong> or <strong>Refresh</strong>."
            "</p>"
        )
    elif lo == 0:
        hint = (
            "<p class='fc-hint-amber' style='margin:6px 0 0;'>"
            "No <strong>LOW</strong> cost clues left.</p>"
        )
    elif hi == 0:
        hint = (
            "<p class='fc-hint-amber' style='margin:6px 0 0;'>"
            "No <strong>HIGH</strong> cost clues left.</p>"
        )
    else:
        hint = ""
    toks = int(o.tokens)
    w = 100.0 * toks / MAX_TOKENS if MAX_TOKENS else 0.0
    fillcls = _token_bar_fill_class(w)
    return (
        "<div class='fc-play-footer'>"
        "<div class='fc-counter-row'>"
        f"Low: <strong style='color:{cl}'>{lo}</strong> &nbsp;·&nbsp; "
        f"High: <strong style='color:{ch}'>{hi}</strong>"
        f"</div>{hint}"
        "<div class='fc-token-outer'>"
        f"<p class='fc-token-label' style='margin:0 0 6px;'>"
        f"Tokens: {toks} / {MAX_TOKENS}</p>"
        "<div class='fc-token-track'>"
        f'<div class="{fillcls}" style="width: {w:.1f}%;">'
        "</div></div></div></div>"
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
            return "Smart refresh/skip saved cost"
        if o.reward < -0.01:
            return "Costly refresh on a strong pick"
        return "You refreshed"

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


def _oneline_last_action(
    o,
    last_action: int | None,
    outcome: str,
) -> str:
    if last_action is None:
        return _html_escape(
            (outcome or "Episode started — your move.").strip()
        )
    r = float(o.reward)
    if r > 1e-6:
        rs = f"+{r:.2f}"
    else:
        rs = f"{r:.2f}"
    an = _action_name(last_action)
    return f"[Step {o.step_number}] {an} → {rs} reward — {_html_escape(outcome)}"


def _last_step_html(
    o,
    outcome: str,
    last_action: int | None = None,
) -> str:
    r = float(o.reward)
    if r > 1e-6:
        rc, cls = f"+{r:.3f}", "fc-reward-pos"
    elif r < -1e-6:
        rc, cls = f"{r:.3f}", "fc-reward-neg"
    else:
        rc, cls = f"{r:.3f}", "fc-reward-neu"
    line = _oneline_last_action(o, last_action, outcome)
    detail = (
        ""
        if last_action is None
        else f"<p class='fc-mute' style='font-size:0.88rem;margin:8px 0 0'>{_html_escape(outcome)}</p>"
    )
    return (
        f"<div class='fc-card-log'>"
        f"<h3>Last action</h3>"
        f"<p class='fc-oneline-log'>{line}</p>"
        f"<p style='margin:0 0 0;'><span class='fc-mute' style='color:#6b7c8a'>"
        f"Step reward</span> <span class='{cls}' style='font-size:1.05em'>{rc}</span> · "
        f"<span class='fc-mute' style='color:#6b7c8a'>Tokens</span> "
        f"{o.tokens} / {MAX_TOKENS} · <span class='fc-mute' style='color:#6b7c8a'>Step</span> {o.step_number}</p>"
        f"{detail}"
        f"</div>"
    )


def _flow_badge(
    o,
    last_action: int | None,
) -> str:
    if o.done and last_action == 2:
        return (
            "<p class='fc-flow-line'>"
            "Episode finished · <span class='fc-reward-neg' style='font-weight:800'>"
            "Committed</span></p>"
        )
    if o.done and last_action == 3:
        return (
            "<p class='fc-flow-line'>"
            "Episode finished · <span class='fc-reward-pos' style='font-weight:800'>"
            "Refreshed</span></p>"
        )
    if o.done:
        return "<p class='fc-flow-line'>Episode <strong>finished</strong></p>"
    return (
        "<p class='fc-flow-line' style='color:#00d4ff'>"
        "In progress — reveal, refresh, or commit when ready."
        "</p>"
    )


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


# Episode log (0–3 = env Action values; same API as before)
_ACTION_NAMES: dict[int, str] = {
    0: "Reveal Low",
    1: "Reveal High",
    2: "Commit",
    3: "Refresh",
}

HISTORY_STATE_INIT: dict = {
    "lines": [],
    "cum": 0.0,
    "stale": True,
    "card_rows": [],
}


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
    card_rows: list[dict] = list(prev.get("card_rows", []))
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
    card_rows.append(
        {
            "step": int(o.step_number),
            "action": an,
            "reward": float(o.reward),
            "final": False,
        }
    )
    if o.done:
        if user_action == 2:
            end = "Commit (terminal action)"
        elif user_action == 3:
            end = "Refresh (terminal; env action SKIP)"
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
        card_rows.append(
            {
                "step": int(o.step_number),
                "action": "—",
                "reward": float(cum),
                "final": True,
                "cum": float(cum),
            }
        )
    return {
        "lines": lines,
        "cum": cum,
        "stale": False,
        "card_rows": card_rows,
    }


def _log_to_html(h: dict | None) -> str:
    h = h or HISTORY_STATE_INIT
    if h.get("stale", True) and not h.get("card_rows") and not h.get("lines"):
        return (
            "<div class='fc-history-entries' style='padding:12px'><p class='fc-mute' "
            "style='margin:0'>_No actions yet. Start a new episode._</p></div>"
        )
    if not h.get("card_rows") and not h.get("lines"):
        return (
            "<div class='fc-history-entries' style='padding:12px'>"
            "<p class='fc-mute' style='margin:0'>_New episode — steps will log as you act._</p></div>"
        )
    parts: list[str] = ['<div class="fc-history-entries">']
    for row in h.get("card_rows", []):
        if row.get("final"):
            c = float(row.get("cum", 0.0))
            sc = f"+{c:.2f}" if c > 0 else f"{c:.2f}"
            parts.append(
                f"<div class='fc-hist-card fc-hist-card--final'>"
                f"<span class='fc-hist-st'>Episode total</span>"
                f"Final cumulative reward: <strong class='fc-reward-pos'>{sc}</strong></div>"
            )
            continue
        stn = int(row.get("step", 0))
        an = str(row.get("action", "—"))
        r = float(row.get("reward", 0.0))
        rs = f"+{r:.2f}" if r > 0 else f"{r:.2f}"
        cls = "fc-reward-pos" if r > 1e-6 else "fc-reward-neg" if r < -1e-6 else "fc-reward-neu"
        an_esc = _html_escape(an)
        parts.append(
            f"<div class='fc-hist-card'>"
            f"<span class='fc-hist-st'>Step {stn}</span>"
            f"<span>{an_esc} → <span class='{cls}'>{rs}</span> reward</span></div>"
        )
    parts.append("</div>")
    return "".join(parts)


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

                gr.HTML(STATIC_HEADER, elem_classes=["fc-hgame"])

                b_reset = gr.Button(
                    "Start new episode",
                    elem_classes=["fc-btn--start", "gr-button", "gr-button-primary"],
                )

                card_block = gr.HTML(_render_six_clues((HIDDEN,) * 6, None))

                with gr.Row(elem_classes=["fc-actions-row"]):
                    b_low = gr.Button(
                        "Reveal Low",
                        interactive=False,
                        elem_classes=["gr-button", "fc-btn--low", "gr-button-primary"],
                    )
                    b_high = gr.Button(
                        "Reveal High",
                        interactive=False,
                        elem_classes=["gr-button", "fc-btn--high", "gr-button-primary"],
                    )
                with gr.Row(elem_classes=["fc-actions-row"]):
                    b_skip = gr.Button(
                        "Refresh",
                        interactive=False,
                        elem_classes=["gr-button", "fc-btn--refresh", "gr-button-primary"],
                    )
                    b_commit = gr.Button(
                        "Commit",
                        interactive=False,
                        elem_classes=["gr-button", "fc-btn--commit", "gr-button-primary"],
                    )

                footer_status = gr.HTML(
                    COUNTER_FOOTER_IDLE, elem_classes=["fc-footer-wrap"]
                )
                last_block = gr.HTML(
                    "<div class='fc-card-log'>"
                    "<h3>Last action</h3><p class='fc-mute' style='margin:0'>"
                    "No action yet. Start a new episode.</p></div>"
                )
                flow = gr.HTML(
                    "<p class='fc-flow-line' style='color:#6b7c8a; margin:0;'>"
                    "Press <strong>Start new episode</strong> to begin.</p>"
                )
                with gr.Accordion("Episode history", open=False, elem_id="fc-history-accordion"):
                    history_display = gr.HTML(
                        _log_to_html(HISTORY_STATE_INIT), elem_classes=["fc-history-box"]
                    )

                _out_play = [
                    st,
                    history_state,
                    card_block,
                    last_block,
                    flow,
                    footer_status,
                    history_display,
                    b_low,
                    b_high,
                    b_skip,
                    b_commit,
                ]
                n_out = 12
                n_skip_updates = n_out - 1

                def on_start() -> tuple:
                    e = FCEnvEnvironment()
                    o = e.reset()
                    snap = _snapshot(o)
                    s0: dict = {"env": e, "pre_obs": snap}
                    h0: dict = {
                        "lines": [],
                        "cum": 0.0,
                        "stale": False,
                        "card_rows": [],
                    }
                    h_html = _log_to_html(h0)
                    bup = _button_state_from_obs(o)
                    return (
                        s0,
                        h0,
                        _render_six_clues(o.revealed_clues, None),
                        _last_step_html(
                            o,
                            "Episode started — your move.",
                            None,
                        ),
                        _flow_badge(o, None),
                        _play_footer_html(o),
                        h_html,
                    ) + bup

                def on_step(
                    s: dict | None,
                    user_action: int,
                    hist: dict | None,
                ) -> tuple:
                    if not s or s.get("env") is None:
                        return (s,) + (gr.update(),) * n_skip_updates

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
                    h_new = _log_after_step(hist, user_action, o)
                    h_html = _log_to_html(h_new)
                    return (
                        next_s,
                        h_new,
                        _render_six_clues(o.revealed_clues, new_idx),
                        _last_step_html(
                            o,
                            _outcome_text(user_action, pre_dict, o, new_idx),
                            user_action,
                        ),
                        _flow_badge(o, user_action),
                        _play_footer_html(o),
                        h_html,
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
                        "Reveals fill the six **clue** cards; hidden slots show **???** until you pay to "
                        "reveal them. When you **commit** or **refresh** (the previous **skip**), the episode "
                        "ends; rewards reflect how well you read the situation. This interface uses the same "
                        "**step** logic as the Space **API** — the front end is only here for a clear, "
                        "decision-first experience.",
                        sanitize_html=False,
                    )

    return demo
