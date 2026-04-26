"""Gradio UI for the HF Space (mounted at /ui)."""

from __future__ import annotations

import ast
import json
import gradio as gr
from pathlib import Path

from environment import FCEnvEnvironment
from models import Action

# Match reset() in FCEnvEnvironment (for token bar; backend unchanged)
MAX_TOKENS = 100
HIDDEN = "HIDDEN"

# Live score panel (Play tab); reset on new episode
LIVE_STATS_DEFAULT: dict[str, int | float] = {
    "current_tokens": MAX_TOKENS,
    "total_reward": 0.0,
    "step_count": 0,
}

# Human-readable clue attribute badges (Play card grid HTML)
ATTR_LABELS: dict[str, str] = {
    "nationality": "NATION",
    "position": "POSITION",
    "club": "CLUB",
    "league": "LEAGUE",
    "decade": "DECADE",
    "caps": "CAPS",
    "age": "AGE",
    "goals": "GOALS",
}

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

/* Live stats (above action buttons) */
.fc-live-wrap { margin: 12px 0 16px; max-width: 900px; margin-left: auto; margin-right: auto; }
.fc-live-title { margin: 0 0 10px; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: #7dd3fc !important; text-align: center; }
.fc-live-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
@media (max-width: 640px) { .fc-live-grid { grid-template-columns: repeat(2, 1fr); } }
.fc-live-tile {
  text-align: center; background: linear-gradient(180deg, #0f172a, #0b1018); border: 1px solid #1e2a3a; border-radius: 12px;
  padding: 12px 10px 14px; box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.08) inset, 0 4px 20px -12px #0008;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.fc-live-tile:hover { border-color: #00d4ff33; box-shadow: 0 0 24px -10px #00d4ff22, 0 4px 20px -12px #0008; }
.fc-live-ico { font-size: 1.1rem; line-height: 1; display: block; margin-bottom: 4px; }
.fc-live-lbl { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8b9aad !important; font-weight: 700; margin: 0 0 4px; }
.fc-live-val { font-size: 1.2rem; font-weight: 800; color: #e8f0f8 !important; line-height: 1.2; }
.fc-live-val--pos { color: #00ff88 !important; text-shadow: 0 0 16px #00ff8822; }
.fc-live-val--neg { color: #ff4d4d !important; text-shadow: 0 0 16px #ff4d4d22; }
.fc-live-val--neu { color: #cbd5e1 !important; }

/* Confidence (below Live Stats) */
.fc-conf-outer { max-width: 900px; margin: 0 auto 14px; }
.fc-conf-card {
  text-align: center; background: linear-gradient(180deg, #0e1520, #0a0d12); border: 1px solid #1e2a3a; border-radius: 14px;
  padding: 16px 18px 18px; box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.06) inset, 0 4px 24px -12px #0006;
  transition: box-shadow 0.25s, border-color 0.2s;
}
.fc-conf-title { margin: 0 0 10px; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; color: #7c8b9c !important; }
.fc-conf-line { margin: 0; font-size: 1.45rem; font-weight: 800; line-height: 1.25; letter-spacing: 0.02em; }
.fc-conf-line span { text-shadow: 0 0 28px currentColor; }
.fc-conf--hi { color: #4ade80 !important; --glow: #00ff88; }
.fc-conf--med { color: #f5c542 !important; --glow: #f5c542; }
.fc-conf--lo { color: #f87171 !important; --glow: #ff4d4d; }
.fc-conf--unk { color: #8b9aab !important; font-size: 1.1rem; font-weight: 600; text-shadow: none; }
.fc-conf--hi, .fc-conf--med, .fc-conf--lo { text-shadow: 0 0 20px var(--glow)44; }
.fc-conf-wrap--hi { border-color: #14532d88; box-shadow: 0 0 0 1px #00ff8822 inset, 0 0 40px -20px #00ff8840; }
.fc-conf-wrap--med { border-color: #ca8a0444; box-shadow: 0 0 0 1px #f5c5421a inset, 0 0 36px -20px #f5c54233; }
.fc-conf-wrap--lo { border-color: #7f1d1d55; box-shadow: 0 0 0 1px #ff4d4d1a inset, 0 0 36px -20px #ff4d4d2a; }
.fc-conf-wrap--unk { border-color: #1e2a3a; box-shadow: 0 4px 20px -12px #0006; }

/* Episode trace (step-by-step, below Last action) */
.fc-trace-outer { margin-top: 10px; }
.fc-trace-panel {
  background: #101820; border: 1px solid #243142; border-radius: 12px; padding: 16px 18px;
  margin: 0; font-family: "JetBrains Mono", "SFMono-Regular", ui-monospace, Menlo, monospace;
  font-size: 0.88rem; line-height: 1.5; color: #e2e8f0; box-shadow: inset 0 1px 0 #ffffff0a;
}
.fc-trace-title { margin: 0 0 10px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8 !important; font-weight: 800; }
.fc-trace-empty { margin: 0; color: #6b7c8b !important; }
.fc-trace-lines { display: flex; flex-direction: column; gap: 6px; }
.fc-trace-step { margin-bottom: 10px; }
.fc-trace-step:last-of-type { margin-bottom: 0; }
.fc-trace-line { padding: 2px 0; }
.fc-trace-reason { font-size: 0.78rem; color: #7c8b9e !important; margin: 3px 0 0; padding: 0 0 0 0.9rem; line-height: 1.4; }
.fc-trace-pos { color: #00ff88 !important; font-weight: 700; }
.fc-trace-neg { color: #ff4d4d !important; font-weight: 700; }
.fc-trace-neu { color: #cbd5e1 !important; font-weight: 600; }
.fc-trace-final { margin-top: 12px; padding-top: 10px; border-top: 1px solid #1e2a3a; font-weight: 800; }

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

/* ========== Compare / About: game dashboard (tabs) ========== */
.fc-tab-dashboard, .gr-html .fc-tab-dashboard, .prose .fc-tab-dashboard { color: #e8ecf1 !important; }
.fc-tab-dashboard a { color: #00d4ff !important; }
.fc-tab-dashboard code {
  background: #131a24 !important; color: #f1f5f9 !important; padding: 2px 7px; border-radius: 6px;
  font-size: 0.88em; border: 1px solid #2a3848;
}
.fc-dashboard-title {
  font-size: 1.4rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 6px; color: #f8fafc !important;
  text-shadow: 0 0 32px rgba(0, 212, 255, 0.18);
  line-height: 1.2;
}
.fc-dashboard-sub {
  font-size: 0.95rem; color: #e2e8f0 !important; margin: 0 0 20px; line-height: 1.5; max-width: 56ch;
  font-weight: 500;
}
.fc-compare-arena { margin: 0 0 4px; }

.fc-compare-row {
  display: flex; flex-wrap: wrap; align-items: stretch; justify-content: center;
  gap: 14px 18px; margin: 0 0 20px;
}
.fc-compare-card {
  flex: 1 1 260px; max-width: 420px; min-width: 0;
  border-radius: 18px; padding: 22px 22px 20px; position: relative; overflow: hidden;
  min-height: 0;
  display: flex; flex-direction: column;
  background: linear-gradient(165deg, #121a25 0%, #0a0d12 100%);
  border: 1px solid #1e2835; transition: transform 0.22s ease, box-shadow 0.25s;
}
.fc-compare-card::before {
  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  background: linear-gradient(150deg, rgba(255,255,255,0.04) 0%, transparent 40%);
  opacity: 0.5;
}
.fc-compare-card:hover { transform: translateY(-3px); }
.fc-compare-card--random {
  border-color: #7c2d1244;
  box-shadow: 0 0 40px -18px #ff4d4d99, 0 12px 32px -18px #000, inset 0 1px 0 #ff4d4d0d;
}
.fc-compare-card--random:hover { box-shadow: 0 0 48px -10px #ff4d4d55, 0 16px 40px -16px #000, inset 0 1px 0 #ff4d4d14; }
.fc-compare-card--trained {
  border-color: #00d4ff4d;
  box-shadow:
    0 0 0 1px rgba(0, 255, 136, 0.1),
    0 0 50px -16px #00d4ff66,
    0 0 50px -12px #00ff8840,
    0 12px 36px -20px #000, inset 0 1px 0 #00d4ff1a;
}
.fc-compare-card--trained:hover {
  box-shadow: 0 0 0 1px rgba(0, 255, 170, 0.15), 0 0 60px -10px #00d4ff80, 0 16px 40px -14px #000, inset 0 1px 0 #00ff8822;
}
.fc-compare-h {
  display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 0 0 16px; position: relative; z-index: 1;
}
.fc-compare-h h3 {
  margin: 0; font-size: 1.1rem; font-weight: 800; letter-spacing: 0.03em; color: #f1f5f9 !important;
  text-shadow: 0 0 20px currentColor; line-height: 1.2;
}
.fc-compare-card--random .fc-compare-h h3 { color: #fb923c !important; text-shadow: 0 0 16px #ff4d4d35; }
.fc-compare-card--trained .fc-compare-h h3 { color: #4ade80 !important; text-shadow: 0 0 20px #00d4ff44, 0 0 14px #00ff8833; }
.fc-compare-icon { font-size: 1.75rem; line-height: 1; filter: drop-shadow(0 0 6px #fff3); }
.fc-compare-metrics { display: flex; flex-direction: column; gap: 16px; position: relative; z-index: 1; flex: 1; justify-content: space-evenly; }
.fc-metric {
  text-align: center; background: #0a0e14aa; border: 1px solid #1e2832; border-radius: 12px; padding: 12px 12px 14px;
  backdrop-filter: blur(6px);
}
.fc-compare-token-note {
  font-size: 0.8rem; color: #8c9baf !important; text-align: center; margin: 4px 0 18px; line-height: 1.4;
  letter-spacing: 0.02em; max-width: 44rem; margin-left: auto; margin-right: auto;
}
.fc-compare-card--random .fc-metric { border-color: #3f2a1f; }
.fc-compare-card--trained .fc-metric { border-color: #1a3a35; }
.fc-metric-label { display: block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #a8b0bc !important; margin-bottom: 4px; }
.fc-metric-value { display: block; font-size: 1.85rem; font-weight: 800; line-height: 1.1; color: #f8fafc !important; }
.fc-compare-card--random .fc-metric-value { color: #fecdd3 !important; }
.fc-compare-card--trained .fc-metric-value { color: #a7f3d0 !important; text-shadow: 0 0 24px #00ff8822; }

.fc-compare-vsplit {
  display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 48px; flex: 0 0 auto; align-self: center;
  gap: 6px;
}
.fc-vs {
  display: flex; align-items: center; justify-content: center;
  min-width: 48px; min-height: 48px; border-radius: 12px; font-size: 0.95rem; font-weight: 900; letter-spacing: 0.06em; color: #0b0f14 !important;
  background: linear-gradient(180deg, #f5c542, #c99a1e);
  border: 1px solid #fde68a; box-shadow: 0 0 24px -4px #f5c54288, 0 4px 0 #7c5a0a44;
  text-shadow: 0 1px 0 #fffc; position: relative; z-index: 1;
}
.fc-vs-connector { display: none; }
@media (min-width: 700px) {
  .fc-vs-connector { display: block; width: 2px; height: 28px; background: linear-gradient(180deg, #f5c542, #00d4ff44); border-radius: 1px; opacity: 0.5; }
}
@media (max-width: 699px) {
  .fc-compare-row { flex-direction: column; align-items: stretch; }
  .fc-compare-vsplit { flex-direction: row; min-width: 100%; max-width: 100%; padding: 8px 0; gap: 10px; }
  .fc-compare-vsplit .fc-vs-connector {
    display: block; flex: 1; height: 2px; min-width: 24px;
    background: linear-gradient(90deg, #ff4d4d44, #f5c542, #00d4ff55); opacity: 0.55; border-radius: 1px;
  }
  .fc-dashboard-title { font-size: 1.2rem; }
}

.fc-compare-summary {
  margin: 0 0 8px; padding: 16px 18px; text-align: center; font-size: 1.05rem; line-height: 1.4;
  font-weight: 800; color: #f0fdf4 !important;
  background: linear-gradient(90deg, #0a1510, #0a1418, #0a1418); border: 1px solid #14532d; border-radius: 14px;
  box-shadow: 0 0 40px -20px #00ff8833, inset 0 1px 0 #00ff8822; letter-spacing: 0.01em;
}
.fc-compare-summary .fc-snum { color: #4ade80 !important; }
.fc-compare-foot { font-size: 0.8rem; color: #7c8796 !important; margin: 12px 0 0; }
.fc-compare-na { color: #94a3b8; font-size: 0.95rem; }
.fc-training-depth {
  margin: 16px 0 10px;
  border: 1px solid #1f3348;
  border-radius: 14px;
  background: linear-gradient(180deg, #0b1626, #0b1118);
  box-shadow: 0 0 30px -18px #00d4ff55, inset 0 1px 0 #ffffff08;
  padding: 16px 16px 14px;
}
.fc-training-depth h3 {
  margin: 0 0 12px;
  font-size: 1rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  color: #dbeafe !important;
  text-transform: uppercase;
  text-align: center;
}
.fc-depth-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 700px) {
  .fc-depth-grid { grid-template-columns: 1fr; }
}
.fc-depth-card {
  border: 1px solid #22405a;
  border-radius: 12px;
  background: #0a121d;
  padding: 12px 12px 10px;
  box-shadow: 0 0 22px -16px #00d4ff66;
}
.fc-depth-step {
  margin: 0 0 6px;
  font-size: 0.83rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #93c5fd !important;
  font-weight: 700;
}
.fc-depth-line {
  margin: 2px 0;
  font-size: 0.93rem;
  color: #e2e8f0 !important;
}
.fc-depth-line strong { font-weight: 800; color: #f8fafc !important; }
.fc-depth-line .fc-pos { color: #4ade80 !important; font-weight: 800; }
.fc-depth-conv {
  margin: 12px 0 0;
  text-align: center;
  color: #bfdbfe !important;
  font-weight: 700;
}
.fc-depth-insight {
  margin: 10px auto 0;
  max-width: 48rem;
  text-align: center;
  font-style: italic;
  color: #bbf7d0 !important;
  background: linear-gradient(90deg, #052e2b66, #0a283366);
  border: 1px solid #14532d66;
  border-radius: 10px;
  padding: 10px 12px;
}
.fc-dashboard-wrap { max-width: 1000px; margin: 0 auto; }
.fc-dashboard-grad { padding: 22px 4px; border-radius: 18px; }
.fc-dashboard-grad--compare {
  background: linear-gradient(180deg, #0b1422 0%, #0b0f14 55%, #0a0c0f 100%);
  border: 1px solid #1a222c; box-shadow: inset 0 1px 0 #00d4ff0d, 0 8px 40px -20px #0006;
  margin: 0 0 8px;
}

/* About: glass info cards */
.fc-dashboard-grad--about {
  background: linear-gradient(180deg, #0a1018, #0b0f14, #0a0c0f);
  border: 1px solid #1a222c; padding: 8px 4px 4px; margin: 0 0 4px; border-radius: 18px; box-shadow: 0 8px 32px -18px #0005;
}
.fc-about-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 0; }
@media (max-width: 900px) { .fc-about-grid { grid-template-columns: 1fr; } }
.fc-glass {
  position: relative; border-radius: 16px; padding: 20px 18px 20px; overflow: hidden; min-height: 0;
  background: rgba(18, 24, 34, 0.45);
  border: 1px solid rgba(0, 212, 255, 0.14);
  box-shadow: 0 8px 32px -16px #000, inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter: blur(10px) saturate(120%);
  -webkit-backdrop-filter: blur(10px) saturate(120%);
  transition: transform 0.22s ease, border-color 0.2s, box-shadow 0.25s;
}
.fc-glass::before {
  content: ""; position: absolute; inset: 0; background: linear-gradient(135deg, #00d4ff08, transparent 55%); pointer-events: none;
}
.fc-glass:hover { transform: translateY(-3px) scale(1.01); border-color: #00d4ff33; box-shadow: 0 12px 40px -18px #000, 0 0 32px -12px #00d4ff2a, inset 0 1px 0 #fff0; }
.fc-glass-h {
  display: flex; align-items: center; gap: 10px; margin: 0 0 12px; position: relative; z-index: 1; border-bottom: 1px solid #00d4ff1a; padding-bottom: 10px;
}
.fc-glass-ico { font-size: 1.5rem; line-height: 1; }
.fc-glass-t { margin: 0; font-size: 1.05rem; font-weight: 800; color: #f1f5f9 !important; letter-spacing: 0.04em; text-shadow: 0 0 18px #00d4ff1a; }
.fc-glass ul { list-style: none; margin: 0; padding: 0; position: relative; z-index: 1; }
.fc-glass li { margin: 0; padding: 7px 0; padding-left: 0; color: #e2e8f0 !important; font-size: 0.95rem; line-height: 1.45; border-top: 1px solid #ffffff0a; }
.fc-glass li:first-of-type { border-top: 0; padding-top: 0; }
.fc-glass li::before { content: "· "; color: #f5c542; font-weight: 900; margin-right: 4px; }

/* Training Insights tab: chart cards (dashboard, not raw plots) */
.fc-insights-page { max-width: 920px; margin: 0 auto; padding: 8px 6px 24px; display: flex; flex-direction: column; gap: 28px; }
.fc-insight-card {
  background: linear-gradient(180deg, #0f172a 0%, #0b0f14 100%) !important;
  border: 1px solid rgba(0, 212, 255, 0.18) !important;
  border-radius: 18px !important;
  padding: 20px 22px 24px !important;
  box-shadow: 0 0 0 1px rgba(0, 255, 136, 0.05) inset, 0 0 48px -20px rgba(0, 212, 255, 0.18), 0 12px 40px -24px #000a;
  transition: box-shadow 0.28s ease, border-color 0.25s, transform 0.22s ease;
}
.fc-insight-card:hover {
  border-color: rgba(0, 255, 136, 0.32) !important;
  box-shadow: 0 0 0 1px rgba(0, 255, 136, 0.1) inset, 0 0 56px -12px rgba(0, 212, 255, 0.35), 0 0 40px -16px rgba(0, 255, 136, 0.12), 0 16px 48px -20px #000b;
  transform: translateY(-2px);
}
.fc-insight-head { margin: 0 0 16px; text-align: left; }
.fc-insight-title {
  margin: 0 0 6px; font-size: 1.2rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; color: #f8fafc !important;
  text-shadow: 0 0 24px rgba(0, 212, 255, 0.2);
  line-height: 1.25;
}
.fc-insight-subtitle { margin: 0 0 10px; font-size: 0.95rem; line-height: 1.5; color: #cbd5e1 !important; }
.fc-insight-badge {
  display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: #a5f3fc !important; background: rgba(0, 212, 255, 0.1); border: 1px solid rgba(0, 212, 255, 0.25);
  border-radius: 8px; padding: 4px 10px; line-height: 1.2;
  box-shadow: 0 0 16px -4px rgba(0, 212, 255, 0.25);
}
.fc-insight-miss, .fc-insight-miss-box, .fc-insight-miss-box p {
  text-align: center; color: #8b9aad !important; font-size: 0.95rem; margin: 0; padding: 32px 16px;
}
/* Center plot; cap width; keep aspect */
.fc-insight-card .gr-image, .fc-insight-card [class*="image"] { max-width: 720px; margin: 0 auto !important; }
.fc-insight-card .image-container, .fc-insight-card .image-container > div { max-width: 100% !important; justify-content: center !important; }
.fc-insight-card img, .fc-insight-card .image-container img {
  max-width: min(100%, 720px) !important; width: 100% !important; height: auto !important;
  display: block !important; margin: 0 auto !important; border-radius: 10px; box-shadow: 0 4px 24px -8px #0006;
}
"""

STATIC_HEADER = """
<div class="fc-play-page-header play-header">
  <h1 class="fc-play-h1">FC Decision Lab</h1>
  <p class="fc-play-tagline">Reveal clues strategically. Spend tokens wisely.</p>
  <div class="fc-play-header-line" aria-hidden="true"></div>
</div>
"""

# First line of Blocks (before Tabs): font preconnect + JetBrains Mono only.
GRADIO_APP_FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
"""

# Play tab: complete skin; every declaration uses !important (Gradio loads CSS after ours).
PLAY_TAB_HEAD_INJECT = r"""
<style id="fc-play-styles">
#fc-play-root.play-tab {
  max-width: 980px !important;
  margin: 0 auto !important;
  padding: 18px 20px 20px !important;
  box-sizing: border-box !important;
  background:
    radial-gradient(130% 80% at 10% -10%, rgba(22, 57, 106, 0.32), transparent 55%),
    radial-gradient(120% 80% at 90% 0%, rgba(17, 73, 86, 0.28), transparent 52%),
    linear-gradient(155deg, #0a0e1a 0%, #090d17 45%, #080b14 100%) !important;
  border-radius: 18px !important;
  border: 1px solid #1b2438 !important;
}

.play-tab,
.play-tab .wrap,
.play-tab [class*="block"],
.play-tab .gr-block {
  background: transparent !important;
  color: #dfe7f7 !important;
  border: none !important;
  box-shadow: none !important;
}

.play-tab .card {
  margin: 0 !important;
  padding: 0 !important;
}
.play-tab .card + .card { margin-top: 14px !important; }
.play-tab .section-title {
  margin: 0 0 8px !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.11em !important;
  text-transform: uppercase !important;
  color: #7f91b4 !important;
  border: none !important;
  padding: 0 !important;
}
.play-tab .panel-board .section-title,
.play-tab .panel-stats .section-title,
.play-tab .panel-confidence .section-title,
.play-tab .panel-actions .section-title { display: none !important; }
.play-tab .panel-confidence { display: none !important; }

.play-tab .play-header h1 {
  font-size: 1.05rem !important;
  letter-spacing: 0.12em !important;
  font-weight: 700 !important;
  color: #6e7da0 !important;
  margin: 0 !important;
  text-transform: uppercase !important;
}
.play-tab .play-header p,
.play-tab .play-header .fc-play-tagline {
  margin: 6px 0 10px !important;
  color: #8795b3 !important;
  font-size: 0.83rem !important;
}
.play-tab .fc-play-header-line { display: none !important; }

.play-tab .stats-row {
  display: flex !important;
  justify-content: flex-end !important;
  flex-wrap: wrap !important;
  gap: 10px !important;
  width: 100% !important;
  margin: 0 0 2px !important;
}
.play-tab .stats-row > div { flex: 0 0 auto !important; }
.play-tab .fc-stat-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 999px;
  padding: 7px 14px 7px 11px;
  border: 1px solid #2a3a5c;
  background: linear-gradient(180deg, rgba(19, 29, 47, 0.95), rgba(14, 22, 37, 0.95));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 8px 20px -16px #000;
  font-family: "JetBrains Mono", monospace;
}
.play-tab .fc-chip-ico { font-size: 0.95rem; opacity: 0.95; }
.play-tab .fc-chip-lbl {
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  color: #8ea1c7;
  text-transform: uppercase;
}
.play-tab .fc-chip-val { font-size: 0.93rem; font-weight: 700; color: #f5f9ff; }
.play-tab .fc-chip-val.pos { color: #62f4ad; }
.play-tab .fc-chip-val.neg { color: #ff7f7f; }

.play-tab .start-btn > div,
.play-tab .start-btn > div > div {
  display: flex !important;
  justify-content: center !important;
}
.play-tab .start-btn button,
.play-tab .start-btn .gr-button {
  width: min(320px, 100%) !important;
  min-height: 48px !important;
  border-radius: 999px !important;
  border: 1px solid #2ec7ff !important;
  background: linear-gradient(180deg, rgba(20, 36, 58, 0.95), rgba(14, 26, 45, 0.95)) !important;
  color: #ddf7ff !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  font-size: 0.72rem !important;
  font-weight: 800 !important;
  box-shadow: 0 0 0 1px rgba(46, 199, 255, 0.18) inset, 0 0 22px -8px rgba(16, 215, 255, 0.55) !important;
  animation: fcStartPulse 1.8s ease-in-out infinite !important;
}
@keyframes fcStartPulse {
  0% { box-shadow: 0 0 0 1px rgba(46, 199, 255, 0.18) inset, 0 0 16px -8px rgba(16, 215, 255, 0.4); }
  50% { box-shadow: 0 0 0 1px rgba(46, 199, 255, 0.34) inset, 0 0 30px -6px rgba(16, 215, 255, 0.8); }
  100% { box-shadow: 0 0 0 1px rgba(46, 199, 255, 0.18) inset, 0 0 16px -8px rgba(16, 215, 255, 0.4); }
}

.play-tab .fc-encore-board { margin: 10px auto 0 !important; max-width: 860px; }
.play-tab .fc-board-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 4px 0 10px;
}
.play-tab .fc-board-title {
  font-size: 0.95rem;
  font-weight: 800;
  color: #edf4ff;
  letter-spacing: 0.08em;
}
.play-tab .fc-board-count {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  padding: 3px 10px;
  border-radius: 999px;
  background: #15253f;
  border: 1px solid #34537f;
  color: #91bbff;
}
.play-tab .fc-clue-grid-host {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 780px) {
  .play-tab .fc-clue-grid-host { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.play-tab .fc-encore-card {
  position: relative;
  min-height: 136px;
  border-radius: 14px;
  border: 1px solid #2a3a5c;
  background: linear-gradient(180deg, #1a2035, #13192a);
  padding: 12px 12px 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: box-shadow .18s ease, transform .18s ease, border-color .18s ease;
}
.play-tab .fc-encore-card::after {
  content: "";
  position: absolute;
  inset: -40% 40% auto -40%;
  height: 160%;
  transform: rotate(17deg);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  opacity: 0;
  transition: opacity .2s ease;
}
.play-tab .fc-encore-card:hover { box-shadow: 0 0 22px -10px rgba(83, 136, 255, 0.7); border-color: #3d5688; transform: translateY(-1px); }
.play-tab .fc-encore-card:hover::after { opacity: 1; }
.play-tab .fc-encore-card.is-new { box-shadow: 0 0 0 1px #6effc8 inset, 0 0 26px -10px #4efbc0; }

.play-tab .fc-card-tier {
  position: absolute;
  left: 10px;
  top: 9px;
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.09em;
  border-radius: 999px;
  padding: 3px 10px;
  border: 1px solid;
}
.play-tab .fc-encore-card.is-low .fc-card-tier {
  color: #72f9ff;
  border-color: #2ad4dd;
  background: rgba(17, 94, 102, 0.28);
}
.play-tab .fc-encore-card.is-high .fc-card-tier {
  color: #ffd69b;
  border-color: #f5a64a;
  background: rgba(140, 76, 16, 0.3);
}

.play-tab .fc-card-hidden {
  text-align: center;
  color: #6b7895;
  line-height: 1.2;
}
.play-tab .fc-card-hidden .ico { font-size: 1.7rem; display: block; margin-bottom: 3px; opacity: 0.78; }
.play-tab .fc-card-hidden .q { letter-spacing: 0.34em; font-family: "JetBrains Mono", monospace; font-size: 0.95rem; }
.play-tab .fc-card-revealed { text-align: center; padding-top: 16px; max-width: 100%; }
.play-tab .fc-card-clue {
  color: #f7fbff;
  font-size: 1.02rem;
  font-weight: 700;
  line-height: 1.22;
  text-shadow: 0 0 18px rgba(87, 171, 255, 0.18);
  word-break: break-word;
}
.play-tab .fc-card-attr {
  margin-top: 7px;
  color: #8ea1c7;
  font-size: 0.66rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.play-tab .fc-play-footer {
  margin: 10px auto 0 !important;
  max-width: 860px;
  border-radius: 10px;
  border: 1px solid #1f2b46;
  background: #0f1627;
  padding: 8px 10px;
}
.play-tab .fc-counter-row,
.play-tab .fc-token-label { color: #9fb0d0 !important; font-family: "JetBrains Mono", monospace !important; font-size: 0.78rem !important; }
.play-tab .fc-token-track { height: 6px !important; background: #09101d !important; border-color: #1f2a3e !important; }

.play-tab .confidence-shell {
  margin: 6px auto 2px !important;
  max-width: 860px !important;
  border: 1px solid #243553 !important;
  border-radius: 9px !important;
  background: linear-gradient(180deg, #10192c, #0d1625) !important;
  padding: 10px 12px !important;
  display: grid !important;
  grid-template-columns: auto 1fr auto !important;
  align-items: center !important;
  gap: 10px !important;
}
.play-tab .confidence-lbl {
  color: #8ea1c7 !important;
  letter-spacing: 0.1em !important;
  font-size: 0.65rem !important;
  text-transform: uppercase !important;
  margin: 0 !important;
}
.play-tab .confidence-track {
  height: 7px;
  border-radius: 999px;
  overflow: hidden;
  background: #0a111e;
  border: 1px solid #22314d;
}
.play-tab .confidence-fill { height: 100%; border-radius: inherit; transition: width .2s ease; }
.play-tab .confidence-fill--hi { width: 88%; background: linear-gradient(90deg, #1fde9a, #00e676); }
.play-tab .confidence-fill--med { width: 58%; background: linear-gradient(90deg, #ffc46a, #ff9800); }
.play-tab .confidence-fill--lo { width: 28%; background: linear-gradient(90deg, #ff8a80, #ff5252); }
.play-tab .confidence-fill--unk { width: 8%; background: linear-gradient(90deg, #5f7297, #7f93ba); }
.play-tab .confidence-txt { margin: 0 !important; font-size: 0.8rem !important; font-weight: 700 !important; text-align: right !important; font-family: "JetBrains Mono", monospace !important; }
.play-tab .confidence-txt--hi { color: #4ef3b7 !important; }
.play-tab .confidence-txt--med { color: #ffc46a !important; }
.play-tab .confidence-txt--lo { color: #ff8383 !important; }
.play-tab .confidence-txt--unk { color: #a0b0ce !important; }

.play-tab .btn-grid {
  max-width: 520px !important;
  margin: 0 auto !important;
  gap: 10px !important;
}
.play-tab .btn-grid button, .play-tab .btn-grid .gr-button {
  width: 100% !important;
  min-height: 46px !important;
  border-radius: 8px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  font-size: 0.74rem !important;
  font-weight: 800 !important;
}
#fc-play-btn-low button, #fc-play-btn-low .gr-button {
  background: linear-gradient(180deg, #0bf6ff, #00d8de) !important;
  border: 1px solid #2ef8ff !important;
  color: #04222a !important;
}
#fc-play-btn-high button, #fc-play-btn-high .gr-button {
  background: linear-gradient(180deg, #ffb347, #ff940f) !important;
  border: 1px solid #ffc163 !important;
  color: #281603 !important;
}
#fc-play-btn-refresh button, #fc-play-btn-refresh .gr-button {
  background: transparent !important;
  border: 1px solid #4b628f !important;
  color: #dce8ff !important;
}
#fc-play-btn-commit button, #fc-play-btn-commit .gr-button {
  background: linear-gradient(180deg, #1dff9c, #00e676) !important;
  border: 1px solid #63ffb8 !important;
  color: #032113 !important;
  min-height: 50px !important;
}
.play-tab button:disabled, .play-tab .gr-button:disabled {
  opacity: 0.5 !important;
  filter: saturate(0.5) !important;
}

.play-tab .log-panel,
.play-tab .fc-card-log,
.play-tab .fc-trace-panel {
  max-width: 860px !important;
  margin: 0 auto !important;
  border-radius: 10px !important;
  border: 1px solid #1f2d47 !important;
  background: #0a111e !important;
  padding: 12px 13px !important;
}
.play-tab .fc-oneline-log,
.play-tab .fc-play-log-meta,
.play-tab .fc-trace-line,
.play-tab .fc-trace-reason,
.play-tab .fc-trace-empty {
  font-family: "JetBrains Mono", monospace !important;
  color: #c5d3ec !important;
  font-size: 0.8rem !important;
}
.play-tab .fc-reward-pos { color: #45f2af !important; }
.play-tab .fc-reward-neg { color: #ff8383 !important; }
.play-tab .fc-reward-neu { color: #b7c4df !important; }

.play-tab .status-hint,
.play-tab .fc-play-flow-live {
  margin: 6px auto 2px !important;
  max-width: 860px !important;
  text-align: center !important;
  color: #7686a8 !important;
  font-style: italic !important;
  font-size: 0.82rem !important;
  font-family: "JetBrains Mono", monospace !important;
  padding: 5px 0 !important;
}
.play-tab .status-hint strong,
.play-tab .fc-flow-line strong { color: #9cb7e6 !important; }
.play-tab .fc-history-scroll { max-height: 260px !important; overflow-y: auto !important; padding-right: 4px !important; }

@media (max-width: 640px) {
  #fc-play-root.play-tab { padding: 14px 12px 16px !important; }
  .play-tab .fc-chip-lbl { display: none; }
  .play-tab .confidence-shell { grid-template-columns: 1fr !important; }
  .play-tab .confidence-txt { text-align: left !important; }
}
</style>
"""


def _load_evaluation_json() -> dict:
    p = Path(__file__).resolve().parent / "artifacts" / "evaluation.json"
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _fmetric_win(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    return f"{float(x) * 100:.1f}%"


def _fmetric_reward(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    v = float(x)
    s = f"{v:+.3f}"
    return s


def _fmetric_steps(x: object | None) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "—"
    return f"{float(x):.2f}"


def _compare_panel_html() -> str:
    data = _load_evaluation_json()
    _r = data.get("random_eval")
    rdict: dict = _r if isinstance(_r, dict) else {}
    if isinstance(data.get("ppo_eval"), dict) and data["ppo_eval"]:
        tdict: dict = data["ppo_eval"]
        trained_name = "PPO"
    elif isinstance(data.get("q_tabular_eval"), dict) and data.get("q_tabular_eval"):
        tdict = data["q_tabular_eval"]
        trained_name = "Q-table"
    else:
        tdict = {}
        trained_name = "Trained"

    wr = rdict.get("win_rate")
    wtr = tdict.get("win_rate")
    if (
        wr is not None
        and wtr is not None
        and isinstance(wr, (int, float))
        and isinstance(wtr, (int, float))
    ):
        dpp = (float(wtr) - float(wr)) * 100.0
        if dpp >= 0:
            summary = (
                f"Trained agent outperforms random by "
                f'<span class="fc-snum">+{dpp:.1f}</span> percentage points in win rate.'
            )
        else:
            summary = (
                f"Win rate (trained &minus; random): <span class=\"fc-snum\">{dpp:+.1f}</span> points."
            )
    else:
        summary = (
            "Run <code>python train.py</code> to build <code>artifacts/evaluation.json</code> "
            "and populate this panel."
        )
        summary = f'<span class="fc-compare-na">{summary}</span>'

    foot = ""
    if data and rdict and tdict:
        foot = (
            f'<p class="fc-compare-foot">Source: <code>artifacts/evaluation.json</code> &middot; '
            f"Trained metrics: {trained_name}</p>"
        )

    depth_300 = {}
    depth_1200 = {}
    base = Path(__file__).resolve().parent / "artifacts"
    p300 = base / "eval_300k.json"
    p1200 = base / "eval_1200k.json"
    try:
        with p300.open(encoding="utf-8") as f:
            raw300 = json.load(f)
            if isinstance(raw300, dict):
                d = raw300.get("ppo_eval")
                depth_300 = d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        depth_300 = {}
    try:
        with p1200.open(encoding="utf-8") as f:
            raw1200 = json.load(f)
            if isinstance(raw1200, dict):
                d = raw1200.get("ppo_eval")
                depth_1200 = d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        depth_1200 = {}

    w300 = _fmetric_win(depth_300.get("win_rate"))
    r300 = _fmetric_reward(depth_300.get("avg_reward"))
    w1200 = _fmetric_win(depth_1200.get("win_rate"))
    r1200 = _fmetric_reward(depth_1200.get("avg_reward"))

    conv = "Model converged early (no further gains after 300k steps)"
    if (
        isinstance(depth_300.get("win_rate"), (int, float))
        and isinstance(depth_1200.get("win_rate"), (int, float))
        and isinstance(depth_300.get("avg_reward"), (int, float))
        and isinstance(depth_1200.get("avg_reward"), (int, float))
    ):
        dw = abs(float(depth_1200["win_rate"]) - float(depth_300["win_rate"]))
        dr = abs(float(depth_1200["avg_reward"]) - float(depth_300["avg_reward"]))
        if dw > 1e-6 or dr > 1e-6:
            conv = "Longer training shows measurable change after 300k steps"

    return f"""<div class="fc-tab-dashboard fc-dashboard-wrap">
<div class="fc-dashboard-grad fc-dashboard-grad--compare">
  <h2 class="fc-dashboard-title">Trained vs Random</h2>
  <p class="fc-dashboard-sub">Side-by-side results from the same environment. Bigger is better for win rate and average reward; steps depend on your strategy.</p>
  <div class="fc-compare-arena">
    <div class="fc-compare-row">
      <div class="fc-compare-card fc-compare-card--random" role="group" aria-label="Random policy">
        <div class="fc-compare-h">
          <h3>Random Policy</h3>
        </div>
        <div class="fc-compare-metrics">
          <div class="fc-metric">
            <span class="fc-metric-label">Win rate</span>
            <span class="fc-metric-value">{_fmetric_win(rdict.get("win_rate"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg reward</span>
            <span class="fc-metric-value">{_fmetric_reward(rdict.get("avg_reward"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg steps</span>
            <span class="fc-metric-value">{_fmetric_steps(rdict.get("avg_steps"))}</span>
          </div>
        </div>
      </div>
      <div class="fc-compare-vsplit" aria-hidden="true">
        <div class="fc-vs-connector"></div>
        <div class="fc-vs">VS</div>
        <div class="fc-vs-connector"></div>
      </div>
      <div class="fc-compare-card fc-compare-card--trained" role="group" aria-label="Trained policy">
        <div class="fc-compare-h">
          <h3>Trained Policy</h3>
        </div>
        <div class="fc-compare-metrics">
          <div class="fc-metric">
            <span class="fc-metric-label">Win rate</span>
            <span class="fc-metric-value">{_fmetric_win(tdict.get("win_rate"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg reward</span>
            <span class="fc-metric-value">{_fmetric_reward(tdict.get("avg_reward"))}</span>
          </div>
          <div class="fc-metric">
            <span class="fc-metric-label">Avg steps</span>
            <span class="fc-metric-value">{_fmetric_steps(tdict.get("avg_steps"))}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
  <p class="fc-compare-token-note">Token usage metric is under calibration, so we are focusing on reliable performance metrics.</p>
  <div class="fc-training-depth" role="group" aria-label="Training depth analysis">
    <h3>🧠 Training Depth Analysis</h3>
    <div class="fc-depth-grid">
      <div class="fc-depth-card">
        <p class="fc-depth-step">300k steps</p>
        <p class="fc-depth-line">🎯 <strong>Win Rate:</strong> <strong>{w300}</strong></p>
        <p class="fc-depth-line">📈 <strong>Avg Reward:</strong> <span class="fc-pos">{r300}</span></p>
      </div>
      <div class="fc-depth-card">
        <p class="fc-depth-step">1.2M steps</p>
        <p class="fc-depth-line">🎯 <strong>Win Rate:</strong> <strong>{w1200}</strong></p>
        <p class="fc-depth-line">📈 <strong>Avg Reward:</strong> <span class="fc-pos">{r1200}</span></p>
      </div>
    </div>
    <p class="fc-depth-conv">→ {conv}</p>
    <p class="fc-depth-insight">Instead of overtraining, we identified convergence early and optimized for efficiency.</p>
  </div>
  <p class="fc-compare-summary">{summary}</p>
  {foot}
</div>
</div>"""

ABOUT_PANEL_HTML = r"""
<div class="fc-tab-dashboard fc-dashboard-wrap">
  <div class="fc-dashboard-grad fc-dashboard-grad--about">
    <h2 class="fc-dashboard-title">About this lab</h2>
    <p class="fc-dashboard-sub" style="margin-bottom: 18px;">Scannable quick reference. Same rules and API as the <strong>Play</strong> tab—only the presentation differs.</p>
    <div class="fc-about-grid">
      <article class="fc-glass" role="region" aria-label="Goal">
        <div class="fc-glass-h">
          <h3 class="fc-glass-t">Goal</h3>
        </div>
        <ul>
          <li>Manage a fixed token budget</li>
          <li>Reveal clues to learn about the pick</li>
          <li>Decide: commit, or refresh the candidate</li>
        </ul>
      </article>
      <article class="fc-glass" role="region" aria-label="Actions">
        <div class="fc-glass-h">
          <h3 class="fc-glass-t">Actions</h3>
        </div>
        <ul>
          <li><strong>Reveal Low</strong> — cheap information</li>
          <li><strong>Reveal High</strong> — pricier, higher signal</li>
          <li><strong>Commit</strong> — lock in your final decision</li>
          <li><strong>Refresh</strong> — end episode and skip the candidate (alias: skip)</li>
        </ul>
      </article>
      <article class="fc-glass" role="region" aria-label="Strategy">
        <div class="fc-glass-h">
          <h3 class="fc-glass-t">Strategy</h3>
        </div>
        <ul>
          <li>Balance cost against information</li>
          <li>Don’t burn tokens for no gain</li>
          <li>Commit when you are confident, refresh when the profile looks wrong</li>
        </ul>
      </article>
    </div>
  </div>
</div>
"""


def _attr_display_name(raw_key: str) -> str:
    k = str(raw_key).strip().lower().replace(" ", "_")
    if k in ATTR_LABELS:
        return ATTR_LABELS[k]
    base = str(raw_key).strip()
    return base.upper()[:8] if base else "CLUE"


def _clue_label_value(raw: str) -> tuple[str, str, bool]:
    """(label, value, hidden) for card face."""
    if raw == HIDDEN:
        return ("", "???", True)
    try:
        t = ast.literal_eval(raw)
        if isinstance(t, (list, tuple)) and len(t) == 2:
            k, v = t
            return (_attr_display_name(str(k)), str(v).strip(), False)
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


def _artifact_png_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / filename


def _insight_section_header_html(title: str, subtitle: str) -> str:
    return (
        '<div class="fc-insight-head">'
        f'<h3 class="fc-insight-title">{_html_escape(title)}</h3>'
        f'<p class="fc-insight-subtitle">{_html_escape(subtitle)}</p>'
        '<span class="fc-insight-badge">Generated after training (offline evaluation)</span>'
        "</div>"
    )


# (filename, section title, subtitle) for Training Insights tab
TRAINING_INSIGHT_SECTIONS: list[tuple[str, str, str]] = [
    (
        "reward_curve.png",
        "Learning Progress",
        "Agent improves reward over time during training",
    ),
    (
        "win_rate_vs_random.png",
        "Before vs After Training",
        "Trained agent significantly outperforms random baseline",
    ),
]


def _render_six_clues(
    revealed: tuple[str, ...] | list[str] | None,
    highlight: int | None,
) -> str:
    if not revealed or len(revealed) < 6:
        revealed = [HIDDEN] * 6
    shown = sum(1 for x in revealed if x != HIDDEN)
    parts: list[str] = [
        '<div class="fc-encore-board">'
        '<div class="fc-board-head">'
        '<div class="fc-board-title">CLUES</div>'
        f'<div class="fc-board-count">{shown}/6</div>'
        "</div>"
        '<div class="fc-clue-grid-host">'
    ]
    for i in range(6):
        raw = revealed[i]  # type: ignore[index]
        label, value, hidden = _clue_label_value(raw)
        is_low = i < 3
        tier = "LOW" if is_low else "HIGH"
        tier_cls = "is-low" if is_low else "is-high"
        new_cls = " is-new" if highlight is not None and i == highlight and not hidden else ""
        if hidden:
            parts.append(
                f'<div class="fc-encore-card {tier_cls} is-hidden">'
                f'<span class="fc-card-tier">{tier}</span>'
                '<div class="fc-card-hidden"><span class="ico">🔭</span>'
                '<span class="q">? ? ?</span></div>'
                "</div>"
            )
            continue
        vhtml = _html_escape(value)
        lhtml = _html_escape(label)
        parts.append(
            f'<div class="fc-encore-card {tier_cls} is-revealed{new_cls}">'
            f'<span class="fc-card-tier">{tier}</span>'
            '<div class="fc-card-revealed">'
            f'<div class="fc-card-clue">{vhtml}</div>'
            f'<div class="fc-card-attr">{lhtml}</div>'
            "</div>"
            "</div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


COUNTER_FOOTER_IDLE = (
    "<div class='fc-play-footer'>"
    "<div class='fc-counter-row'>"
    "Low: <span class='fc-mute' style='font-weight:600;'>—</span> &nbsp;·&nbsp; "
    "High: <span class='fc-mute' style='font-weight:600;'>—</span></div>"
    "<p class='fc-mute' style='margin:0 0 8px;font-size:0.86rem;'>"
    "Awaiting episode start.</p>"
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
    cl = "#FF4444" if lo == 0 else "#888888"
    ch = "#FF4444" if hi == 0 else "#FF6666"
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
        "info": dict(getattr(o, "info", {})),
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
    active = " fc-play-log-active" if last_action is not None else ""
    detail = (
        ""
        if last_action is None
        else f"<p class='fc-mute fc-play-log-detail log-content'>{_html_escape(outcome)}</p>"
    )
    return (
        f"<div class='fc-card-log fc-play-log log-panel'>"
        f"<p class='fc-oneline-log log-content{active}'>{line}</p>"
        f"<p class='fc-play-log-meta log-content'><span class='fc-mute'>"
        f"Step reward</span> <span class='{cls}'>{rc}</span> · "
        f"<span class='fc-mute'>Tokens</span> "
        f"{o.tokens} / {MAX_TOKENS} · <span class='fc-mute'>Step</span> {o.step_number}</p>"
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
        "<p class='fc-flow-line fc-play-flow-live'>"
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
            hcls = (
                "fc-reward-pos"
                if c > 1e-6
                else "fc-reward-neg"
                if c < -1e-6
                else "fc-reward-neu"
            )
            parts.append(
                f"<div class='fc-hist-card fc-hist-card--final'>"
                f"<span class='fc-hist-st'>Episode total</span>"
                f"Final cumulative reward: <strong class='{hcls}'>{sc}</strong></div>"
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


def _append_episode_trace(
    prior: list[dict] | list | None,
    o,
) -> list[dict]:
    info = getattr(o, "info", None) or {}
    an = info.get("action_name", "")
    if an in ("", "—", None):
        return list(prior or [])
    row = {
        "step": int(info.get("step_number", o.step_number)),
        "action": str(an),
        "reward": float(info.get("step_reward", o.reward)),
        "reason": str(info.get("reason", "")).strip(),
    }
    return list(prior or []) + [row]


def _episode_trace_html(rows: list[dict] | list | None, episode_done: bool) -> str:
    rows = list(rows or [])
    if not rows and not episode_done:
        body = '<p class="fc-trace-empty">No actions taken yet.</p>'
    else:
        lines: list[str] = []
        for r in rows:
            rw = float(r["reward"])
            if rw > 1e-9:
                rs, ccls = f"+{rw:.2f}", "fc-trace-pos"
            elif rw < -1e-9:
                rs, ccls = f"{rw:.2f}", "fc-trace-neg"
            else:
                rs, ccls = f"{rw:.2f}", "fc-trace-neu"
            an_esc = _html_escape(str(r.get("action", "—")))
            stn = int(r.get("step", 0))
            sub = ""
            reas = str(r.get("reason", "")).strip()
            if reas:
                sub = f'<div class="fc-trace-reason">{_html_escape(reas)}</div>'
            lines.append(
                f'<div class="fc-trace-step">'
                f'<div class="fc-trace-line">Step {stn} · {an_esc} → <span class="{ccls}">{rs}</span></div>'
                f"{sub}</div>"
            )
        body = '<div class="fc-trace-lines">' + "".join(lines) + "</div>"
    final = ""
    if episode_done and rows:
        tot = sum(float(x["reward"]) for x in rows)
        if tot > 1e-9:
            ts, tcls = f"+{tot:.2f}", "fc-trace-pos"
        elif tot < -1e-9:
            ts, tcls = f"{tot:.2f}", "fc-trace-neg"
        else:
            ts, tcls = f"{tot:.2f}", "fc-trace-neu"
        final = (
            f'<div class="fc-trace-final">'
            f'<div class="fc-trace-final-lbl">Final result</div>'
            f"Total reward: <span class=\"{tcls}\">{ts}</span></div>"
        )
    return (
        '<div class="fc-trace-panel">'
        f'<div class="fc-history-scroll">{body}{final}</div>'
        "</div>"
    )


def _update_live_stats(prev: dict | None, o) -> dict:
    """Cumulative total_reward += step_reward; tokens and steps from info. No-op step keeps prior stats (freeze)."""
    info = getattr(o, "info", None) or {}
    if info.get("action_name") == "—":
        return dict(prev) if prev else {**LIVE_STATS_DEFAULT}
    p = {**LIVE_STATS_DEFAULT, **(prev or {})}
    tr = float(info.get("step_reward", o.reward))
    return {
        "current_tokens": int(info.get("tokens_left", o.tokens)),
        "total_reward": float(p.get("total_reward", 0.0)) + tr,
        "step_count": int(info.get("step_number", o.step_number)),
    }


def _stat_html_tokens_left(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    cur = int(s["current_tokens"])
    cls = "neg" if cur <= 20 else ""
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">🪙</span>'
        '<span class="fc-chip-lbl">Tokens</span>'
        f'<span class="fc-chip-val {cls}">{cur}</span>'
        "</div>"
    )


def _stat_html_steps(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    steps = int(s["step_count"])
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">📈</span>'
        '<span class="fc-chip-lbl">Steps</span>'
        f'<span class="fc-chip-val">{steps}</span>'
        "</div>"
    )


def _stat_html_reward(stats: dict | None) -> str:
    s = {**LIVE_STATS_DEFAULT, **(stats or {})}
    tot = float(s["total_reward"])
    if tot > 1e-9:
        cls, reward_str = "pos", f"+{tot:.2f}"
    elif tot < -1e-9:
        cls, reward_str = "neg", f"{tot:.2f}"
    else:
        cls, reward_str = "", f"{tot:.2f}"
    return (
        '<div class="fc-stat-chip">'
        '<span class="fc-chip-ico">💰</span>'
        '<span class="fc-chip-lbl">Reward</span>'
        f'<span class="fc-chip-val {cls}">{reward_str}</span>'
        "</div>"
    )


def _live_stats_html(stats: dict | None) -> str:
    """Single-row HTML (all four stats); Play tab uses four gr.HTML outputs instead."""
    return (
        '<div class="stats-row">'
        + _stat_html_tokens_left(stats)
        + _stat_html_steps(stats)
        + _stat_html_reward(stats)
        + "</div>"
    )


def _compute_confidence_level(live: dict | None) -> str:
    """Heuristic from running total reward and step count; frontend-only (no env change)."""
    s = {**LIVE_STATS_DEFAULT, **(live or {})}
    st = int(s.get("step_count", 0))
    tr = float(s.get("total_reward", 0.0))
    if st == 0:
        return "UNKNOWN"
    if tr > 0.8 and st >= 2:
        return "HIGH"
    if tr > 0:
        return "MEDIUM"
    return "LOW"


def _confidence_html(level: str) -> str:
    tip = "Confidence is based on reward trend and actions taken"
    u = (level or "UNKNOWN").upper()
    if u == "UNKNOWN":
        vcls = "confidence-txt--unk"
        fill_cls = "confidence-fill--unk"
        text = "UNKNOWN"
    elif u == "HIGH":
        vcls = "confidence-txt--hi"
        fill_cls = "confidence-fill--hi"
        text = "HIGH"
    elif u == "MEDIUM":
        vcls = "confidence-txt--med"
        fill_cls = "confidence-fill--med"
        text = "MEDIUM"
    else:
        vcls = "confidence-txt--lo"
        fill_cls = "confidence-fill--lo"
        text = "LOW"
    et = _html_escape(tip)
    tx = _html_escape(text)
    return (
        f'<div class="confidence-shell" title="{et}">'
        f'<div class="confidence-lbl">Confidence</div>'
        f'<div class="confidence-track"><div class="confidence-fill {fill_cls}"></div></div>'
        f'<div class="confidence-txt {vcls}" title="{et}">{tx}</div>'
        f"</div>"
    )


def build_blocks() -> gr.Blocks:
    with gr.Blocks(
        title="FC Decision Lab",
        theme=gr.themes.Base(),
        css=CSS_STRING,
    ) as demo:
        gr.HTML(GRADIO_APP_FONT_LINKS, elem_id="fc-app-font-links")
        with gr.Tabs():
            with gr.Tab("Play"):
                with gr.Column(
                    elem_id="fc-play-root",
                    elem_classes=["play-tab", "fc-play-page"],
                ):
                    gr.HTML(PLAY_TAB_HEAD_INJECT, elem_id="fc-play-style-inject")
                    st = gr.State()  # type: ignore[var-annotated]  # { "env", "pre_obs" }
                    episode_trace = gr.State([])  # type: ignore[var-annotated]  # list[dict] step log
                    live_stats = gr.State(dict(LIVE_STATS_DEFAULT))  # type: ignore[var-annotated]
                    # Before first action per episode → UNKNOWN; after steps → HIGH|MEDIUM|LOW
                    confidence_level = gr.State("UNKNOWN")

                    gr.HTML(
                        STATIC_HEADER,
                        elem_id="fc-play-header",
                        elem_classes=["fc-play-header-slot", "play-header"],
                    )

                    ls0 = dict(LIVE_STATS_DEFAULT)
                    with gr.Group(elem_classes=["card", "panel-stats"]):
                        gr.HTML("<div class='section-title'>Live Stats</div>")
                        with gr.Row(elem_classes=["stats-row"]):
                            stat_html_tok_left = gr.HTML(
                                _stat_html_tokens_left(ls0),
                                elem_id="fc-play-stat-tokens-left",
                                elem_classes=["stat-block"],
                            )
                            stat_html_steps = gr.HTML(
                                _stat_html_steps(ls0),
                                elem_id="fc-play-stat-steps",
                                elem_classes=["stat-block"],
                            )
                            stat_html_reward = gr.HTML(
                                _stat_html_reward(ls0),
                                elem_id="fc-play-stat-reward",
                                elem_classes=["stat-block"],
                            )

                    with gr.Group(elem_classes=["card", "panel-board"]):
                        gr.HTML("<div class='section-title'>Game Board</div>")
                        b_reset = gr.Button(
                            "Start new episode",
                            elem_id="fc-play-btn-start",
                            elem_classes=[
                                "start-btn",
                                "fc-btn--start",
                                "gr-button",
                                "gr-button-primary",
                            ],
                        )
                        card_block = gr.HTML(
                            _render_six_clues((HIDDEN,) * 6, None),
                            elem_id="fc-play-cards",
                        )
                        footer_status = gr.HTML(
                            COUNTER_FOOTER_IDLE,
                            elem_id="fc-play-footer-tokens",
                            elem_classes=["fc-footer-wrap"],
                        )

                    with gr.Group(elem_classes=["card", "panel-confidence"]):
                        gr.HTML("<div class='section-title'>Confidence</div>")
                        confidence_display = gr.HTML(
                            _confidence_html("UNKNOWN"),
                            elem_id="fc-play-confidence",
                            elem_classes=["fc-conf-outer"],
                            visible=False,
                        )

                    with gr.Group(elem_classes=["card", "panel-actions"]):
                        gr.HTML("<div class='section-title'>Actions</div>")
                        with gr.Row(elem_classes=["btn-grid"]):
                            b_low = gr.Button(
                                "REVEAL LOW",
                                interactive=False,
                                elem_id="fc-play-btn-low",
                                elem_classes=[
                                    "btn-low",
                                    "gr-button",
                                    "fc-btn--low",
                                    "gr-button-primary",
                                ],
                            )
                            b_high = gr.Button(
                                "REVEAL HIGH",
                                interactive=False,
                                elem_id="fc-play-btn-high",
                                elem_classes=[
                                    "btn-high",
                                    "gr-button",
                                    "fc-btn--high",
                                    "gr-button-primary",
                                ],
                            )
                        with gr.Row(elem_classes=["btn-grid"]):
                            b_skip = gr.Button(
                                "REFRESH",
                                interactive=False,
                                elem_id="fc-play-btn-refresh",
                                elem_classes=[
                                    "btn-ghost",
                                    "gr-button",
                                    "fc-btn--refresh",
                                    "gr-button-primary",
                                ],
                            )
                            b_commit = gr.Button(
                                "COMMIT",
                                interactive=False,
                                elem_id="fc-play-btn-commit",
                                elem_classes=[
                                    "btn-commit",
                                    "gr-button",
                                    "fc-btn--commit",
                                    "gr-button-primary",
                                ],
                            )

                    with gr.Group(elem_classes=["card", "panel-last"]):
                        gr.HTML("<div class='section-title'>Last Action</div>")
                        last_block = gr.HTML(
                            "<div class='fc-card-log fc-play-log log-panel'>"
                            "<p class='fc-mute log-content' style='margin:0'>No action yet.</p></div>",
                            elem_id="fc-play-last-action",
                            elem_classes=["log-panel"],
                        )

                    with gr.Group(elem_classes=["card", "panel-history"]):
                        gr.HTML("<div class='section-title'>Episode History</div>")
                        episode_trace_display = gr.HTML(
                            _episode_trace_html([], False),
                            elem_id="fc-play-episode-trace",
                            elem_classes=["fc-trace-outer", "log-panel"],
                        )
                    flow = gr.HTML(
                        "<p class='fc-flow-line status-hint'>"
                        "Press <strong>Start new episode</strong> to begin.</p>",
                        elem_id="fc-play-flow",
                        elem_classes=["status-hint"],
                    )
                _out_play = [
                    st,
                    episode_trace,
                    live_stats,
                    confidence_level,
                    card_block,
                    stat_html_tok_left,
                    stat_html_steps,
                    stat_html_reward,
                    confidence_display,
                    last_block,
                    episode_trace_display,
                    flow,
                    footer_status,
                    b_low,
                    b_high,
                    b_skip,
                    b_commit,
                ]
                n_out = 17
                n_skip_updates = n_out - 1

                def on_start() -> tuple:
                    e = FCEnvEnvironment()
                    o = e.reset()
                    snap = _snapshot(o)
                    s0: dict = {"env": e, "pre_obs": snap}
                    bup = _button_state_from_obs(o)
                    tr_empty: list[dict] = []
                    ls0 = dict(LIVE_STATS_DEFAULT)
                    c0 = _compute_confidence_level(ls0)
                    return (
                        s0,
                        tr_empty,
                        ls0,
                        c0,
                        _render_six_clues(o.revealed_clues, None),
                        _stat_html_tokens_left(ls0),
                        _stat_html_steps(ls0),
                        _stat_html_reward(ls0),
                        _confidence_html(c0),
                        _last_step_html(
                            o,
                            "Episode started — your move.",
                            None,
                        ),
                        _episode_trace_html(tr_empty, False),
                        _flow_badge(o, None),
                        _play_footer_html(o),
                    ) + bup

                def on_step(
                    s: dict | None,
                    user_action: int,
                    ep_tr: list | None,
                    live: dict | None,
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
                    new_tr = _append_episode_trace(ep_tr, o)
                    new_live = _update_live_stats(live, o)
                    c_new = _compute_confidence_level(new_live)
                    return (
                        next_s,
                        new_tr,
                        new_live,
                        c_new,
                        _render_six_clues(o.revealed_clues, new_idx),
                        _stat_html_tokens_left(new_live),
                        _stat_html_steps(new_live),
                        _stat_html_reward(new_live),
                        _confidence_html(c_new),
                        _last_step_html(
                            o,
                            _outcome_text(user_action, pre_dict, o, new_idx),
                            user_action,
                        ),
                        _episode_trace_html(new_tr, o.done),
                        _flow_badge(o, user_action),
                        _play_footer_html(o),
                    ) + bup

                b_reset.click(
                    on_start,
                    inputs=None,
                    outputs=_out_play,
                )
                b_low.click(
                    lambda s, t, lv: on_step(s, 0, t, lv),
                    inputs=[st, episode_trace, live_stats],
                    outputs=_out_play,
                )
                b_high.click(
                    lambda s, t, lv: on_step(s, 1, t, lv),
                    inputs=[st, episode_trace, live_stats],
                    outputs=_out_play,
                )
                b_commit.click(
                    lambda s, t, lv: on_step(s, 2, t, lv),
                    inputs=[st, episode_trace, live_stats],
                    outputs=_out_play,
                )
                b_skip.click(
                    lambda s, t, lv: on_step(s, 3, t, lv),
                    inputs=[st, episode_trace, live_stats],
                    outputs=_out_play,
                )
            with gr.Tab("Compare"):
                gr.HTML(
                    _compare_panel_html(), elem_classes=["fc-tab-compare"]
                )

            with gr.Tab("Training Insights"):
                with gr.Column(elem_classes=["fc-insights-page"]):
                    for _img_name, _stitle, _ssub in TRAINING_INSIGHT_SECTIONS:
                        _ap = _artifact_png_path(_img_name)
                        with gr.Group(elem_classes=["fc-insight-card"]):
                            gr.HTML(_insight_section_header_html(_stitle, _ssub))
                            if _ap.is_file():
                                gr.Image(
                                    value=str(_ap),
                                    show_label=False,
                                    show_download_button=False,
                                )
                            else:
                                gr.HTML(
                                    '<p class="fc-insight-miss">Training graph not available</p>',
                                    elem_classes=["fc-insight-miss-box"],
                                )

            with gr.Tab("About"):
                gr.HTML(ABOUT_PANEL_HTML, elem_classes=["fc-tab-about"])

    return demo
