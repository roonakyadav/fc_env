"""ASGI app: OpenEnv API + Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

import gradio as gr
from fastapi.responses import RedirectResponse

from core.env_server import create_fastapi_app
from environment import FCEnvEnvironment
from gradio_ui import build_blocks

_env = FCEnvEnvironment()
app = create_fastapi_app(_env, title="FC OpenEnv", version="0.1.0")


@app.get("/")
def root() -> RedirectResponse:
    """Hugging Face Space opens / by default; send users to the Gradio UI."""
    return RedirectResponse(url="/ui", status_code=302)


app = gr.mount_gradio_app(app, build_blocks(), path="/ui")
