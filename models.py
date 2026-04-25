from pydantic import BaseModel
from typing import List


class FcAction(BaseModel):
    action: int  # 0-5 reveal, 6 stop


class FcObservation(BaseModel):
    revealed_clues: List[str]
    tokens: int
    step_number: int
    low_remaining: int
    high_remaining: int
    done: bool
    reward: float