from typing import List

from pydantic import BaseModel


class FcAction(BaseModel):
    action: int


class FcObservation(BaseModel):
    revealed_clues: List[str]
    tokens: int
    step_number: int
    low_remaining: int
    high_remaining: int
    done: bool
    reward: float
