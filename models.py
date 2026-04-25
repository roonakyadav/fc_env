from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Action:
    action: int

    def __post_init__(self) -> None:
        if self.action not in (0, 1, 2, 3):
            raise ValueError("action must be one of: 0 (LOW), 1 (HIGH), 2 (STOP), 3 (SKIP)")


@dataclass(frozen=True)
class Observation:
    revealed_clues: Tuple[str, ...]
    tokens: int
    step_number: int
    low_remaining: int
    high_remaining: int
    done: bool
    reward: float


@dataclass(frozen=True)
class State:
    episode_id: str
    step_count: int
    tokens: int
    done: bool
    low_revealed: int
    high_revealed: int
