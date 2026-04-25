from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, TYPE_CHECKING, Callable, Optional  # noqa: I001

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from environment import FCEnvEnvironment


class StepActionRequest(BaseModel):
    action: int = Field(..., description="0=LOW,1=HIGH,2=STOP,3=SKIP")


class ObservationResponseModel(BaseModel):
    revealed_clues: list[str]
    tokens: int
    step_number: int
    low_remaining: int
    high_remaining: int
    done: bool
    reward: float


class StateResponseModel(BaseModel):
    episode_id: str
    step_count: int
    tokens: int
    done: bool
    low_revealed: int
    high_revealed: int


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def create_fastapi_app(
    env: "FCEnvEnvironment",
    *,
    title: str = "FC OpenEnv",
    version: str = "0.1.0",
    on_startup: Optional[Callable[[], None]] = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if on_startup is not None:
            on_startup()
        yield

    app = FastAPI(
        title=title,
        version=version,
        description="FIFA/FC card reveal negotiation environment (OpenEnv API).",
        lifespan=lifespan,
    )
    # Avoid sharing unsynchronized episode state in concurrent requests: use per-request
    # env in thread pool or lock. We run single-worker uvicorn; lock is enough.
    from threading import Lock  # local import to keep import graph light

    _lock = Lock()
    with _lock:
        app.state.env = env

    def _obs_to_dict(o) -> dict:
        return {
            "revealed_clues": list(o.revealed_clues),
            "tokens": o.tokens,
            "step_number": o.step_number,
            "low_remaining": o.low_remaining,
            "high_remaining": o.high_remaining,
            "done": o.done,
            "reward": o.reward,
        }

    @app.get(
        "/health",
        tags=["ops"],
        summary="Health check for hosting / probes",
    )
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/reset", response_model=ObservationResponseModel, tags=["env"])
    def reset_episode() -> dict[str, Any]:
        with _lock:
            obs = app.state.env.reset()
        return _obs_to_dict(obs)

    @app.post("/step", response_model=ObservationResponseModel, tags=["env"])
    def take_step(req: StepActionRequest) -> dict[str, Any]:
        with _lock:
            from models import Action  # import here to keep server deps explicit

            try:
                a = Action(action=req.action)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                ) from e
            obs = app.state.env.step(a)
        return _obs_to_dict(obs)

    @app.get("/state", response_model=StateResponseModel, tags=["env"])
    def get_state() -> dict[str, Any]:
        with _lock:
            s = app.state.env.state()
        return {
            "episode_id": s.episode_id,
            "step_count": s.step_count,
            "tokens": s.tokens,
            "done": s.done,
            "low_revealed": s.low_revealed,
            "high_revealed": s.high_revealed,
        }

    @app.get(
        "/tools/list",
        tags=["mcp"],
        summary="MCP-style tool discovery (subset)",
    )
    def tools_list() -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": "fc_env.reset",
                    "description": "Start a new episode; POST /reset",
                },
                {
                    "name": "fc_env.step",
                    "description": "Execute action; POST /step with {action: int}",
                },
                {
                    "name": "fc_env.state",
                    "description": "Episode metadata; GET /state",
                },
            ]
        }

    @app.post(
        "/tools/call",
        tags=["mcp"],
        summary="MCP-style tool dispatch to HTTP environment actions",
    )
    def tools_call(req: ToolCallRequest) -> dict[str, Any]:
        if req.name in ("fc_env.reset", "reset"):
            with _lock:
                obs = app.state.env.reset()
            return {"content": _obs_to_dict(obs)}
        if req.name in ("fc_env.state", "state"):
            with _lock:
                s = app.state.env.state()
            return {
                "content": {
                    "episode_id": s.episode_id,
                    "step_count": s.step_count,
                    "tokens": s.tokens,
                    "done": s.done,
                    "low_revealed": s.low_revealed,
                    "high_revealed": s.high_revealed,
                }
            }
        if req.name in ("fc_env.step", "step"):
            act = req.arguments.get("action", req.arguments.get("a"))
            if act is None:
                raise HTTPException(
                    status_code=400,
                    detail="Missing arguments.action for fc_env.step",
                )
            with _lock:
                from models import Action

                try:
                    obs = app.state.env.step(Action(action=int(act)))
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
            return {"content": _obs_to_dict(obs)}
        raise HTTPException(
            status_code=404, detail=f"Unknown tool: {req.name}. See GET /tools/list"
        )

    return app
