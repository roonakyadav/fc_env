"""
ASGI entry for Hugging Face Spaces and local dev.

Run: ``uvicorn app:app --host 0.0.0.0 --port 7860`` or ``python app.py``

Hugging Face Spaces set the ``PORT`` environment variable; we respect it in ``__main__``.
"""

import os

import uvicorn

from server.app import app  # noqa: F401  — re-export for uvicorn

__all__ = ["app"]


if __name__ == "__main__":
    _port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=_port)
