"""HTTP client for a deployed FC OpenEnv Space (or local uvicorn)."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class FCEvOpenEnvClient:
    """Calls /reset, /step, /state, /health on a running OpenEnv-compatible server."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FCEvOpenEnvClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def reset(self) -> dict[str, Any]:
        r = self._client.post("/reset")
        r.raise_for_status()
        return r.json()

    def step(self, action: int) -> dict[str, Any]:
        r = self._client.post("/step", json={"action": int(action)})
        r.raise_for_status()
        return r.json()

    def state(self) -> dict[str, Any]:
        r = self._client.get("/state")
        r.raise_for_status()
        return r.json()

    def tools_list(self) -> dict[str, Any]:
        r = self._client.get("/tools/list")
        r.raise_for_status()
        return r.json()

    def tools_call(self, name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        r = self._client.post(
            "/tools/call",
            json={"name": name, "arguments": arguments or {}},
        )
        r.raise_for_status()
        return r.json()
