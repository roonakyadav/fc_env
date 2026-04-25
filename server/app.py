"""ASGI app: OpenEnv API + Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app

from core.env_server import create_fastapi_app
from environment import FCEnvEnvironment
from gradio_ui import build_blocks

_env = FCEnvEnvironment()
app = create_fastapi_app(_env, title="FC OpenEnv", version="0.1.0")


@app.get("/")
def root() -> RedirectResponse:
    """Hugging Face Space opens / by default; send users to the Gradio UI."""
    return RedirectResponse(url="/ui/", status_code=302)


@app.get("/test-ui")
def test_ui() -> dict[str, str]:
    return {"status": "ui route reachable"}


gradio_app = build_blocks()
app = mount_gradio_app(app, gradio_app, path="/ui/")
