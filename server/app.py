"""ASGI app: OpenEnv API + Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

import gradio as gr

from core.env_server import create_fastapi_app
from environment import FCEnvEnvironment
from gradio_ui import build_blocks

_env = FCEnvEnvironment()
app = create_fastapi_app(_env, title="FC OpenEnv", version="0.1.0")


@app.get("/")
def service_index() -> dict:
    return {
        "service": "fc-openenv",
        "description": "Strategic clue-budgeting RL environment (OpenEnv-style API + UI).",
        "endpoints": {
            "api": {
                "reset": "POST /reset",
                "step": "POST /step  body: {action: 0-3}",
                "state": "GET /state",
                "health": "GET /health",
                "docs": "GET /docs",
            },
            "mcp": {"tools_list": "GET /tools/list", "tools_call": "POST /tools/call"},
            "ui": "GET /ui  (Gradio: actions, observations, state traces, plot artifacts)",
        },
    }


app = gr.mount_gradio_app(app, build_blocks(), path="/ui")
