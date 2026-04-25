import random
from dataclasses import dataclass
from typing import List, Tuple
from uuid import uuid4

from models import Action, Observation, State


@dataclass(frozen=True)
class PlayerProfile:
    ovr: int
    program: str
    tradable: bool
    nationality: str
    position: str
    extra_label: str
    extra_value: str


class FCEnvEnvironment:
    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps
        self._episode_id = str(uuid4())
        self._step_count = 0
        self.tokens = 100
        self.done = False
        self.last_action_was_skip = False
        self.is_trap = False
        self.player = PlayerProfile(88, "REGULAR", True, "England", "CM", "team", "Liverpool")
        self.low_clues: List[Tuple[str, str]] = []
        self.high_clues: List[Tuple[str, str]] = []
        self.low_revealed = [False, False, False]
        self.high_revealed = [False, False, False]
        self.state_snapshot = State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            tokens=self.tokens,
            done=self.done,
            low_revealed=0,
            high_revealed=0,
        )
        self.reset()

    def reset(self) -> Observation:
        self._episode_id = str(uuid4())
        self._step_count = 0
        self.tokens = 100
        self.done = False
        self.last_action_was_skip = False
        self.player, self.is_trap = self._generate_player()
        self.low_clues, self.high_clues = self._generate_clues(self.player, self.is_trap)
        self.low_revealed = [False, False, False]
        self.high_revealed = [False, False, False]
        self._update_state()
        return self._observation(reward=0.0)

    def step(self, action: Action) -> Observation:
        if self.done:
            return self._observation(reward=0.0)

        act = action.action
        dense_reward = 0.0

        if act in (0, 1) and self.tokens <= 0:
            self.done = True
            act = 2
            self._step_count += 1
        elif act == 0:
            dense_reward += self._reveal_low_clue()
        elif act == 1:
            dense_reward += self._reveal_high_clue()
        elif act == 2:
            self.done = True
            self._step_count += 1
            dense_reward -= 0.05
        elif act == 3:
            self.done = True
            self.last_action_was_skip = True
            self._step_count += 1
            dense_reward += 0.10 if self.is_trap else -0.20

        if self.tokens <= 0:
            self.done = True
            dense_reward -= 0.10
        if self._step_count >= self.max_steps:
            self.done = True
        if all(self.low_revealed) and all(self.high_revealed):
            self.done = True

        self.tokens = max(self.tokens, 0)
        terminal_reward = self._terminal_reward() if self.done else 0.0
        total_reward = dense_reward + terminal_reward
        self._update_state()
        return self._observation(reward=float(total_reward))

    def state(self) -> State:
        return self.state_snapshot

    def _observation(self, reward: float) -> Observation:
        return Observation(
            revealed_clues=tuple(self._visible_clues()),
            tokens=self.tokens,
            step_number=self._step_count,
            low_remaining=self.low_revealed.count(False),
            high_remaining=self.high_revealed.count(False),
            done=self.done,
            reward=float(reward),
        )

    def _update_state(self) -> None:
        self.state_snapshot = State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            tokens=self.tokens,
            done=self.done,
            low_revealed=sum(self.low_revealed),
            high_revealed=sum(self.high_revealed),
        )

    def _visible_clues(self) -> List[str]:
        low_visible = [str(self.low_clues[i]) if self.low_revealed[i] else "HIDDEN" for i in range(3)]
        high_visible = [str(self.high_clues[i]) if self.high_revealed[i] else "HIDDEN" for i in range(3)]
        return low_visible + high_visible

    def _reveal_low_clue(self) -> float:
        unrevealed = [i for i, is_shown in enumerate(self.low_revealed) if not is_shown]
        if not unrevealed:
            self.tokens -= 2
            return -0.10
        index = random.choice(unrevealed)
        self.low_revealed[index] = True
        self.tokens -= 5
        self._step_count += 1
        return 0.08

    def _reveal_high_clue(self) -> float:
        unrevealed = [i for i, is_shown in enumerate(self.high_revealed) if not is_shown]
        if not unrevealed:
            self.tokens -= 5
            return -0.15
        remaining = len(unrevealed)
        if remaining == 3:
            cost = 15
        elif remaining == 2:
            cost = 20
        else:
            cost = 25
        index = random.choice(unrevealed)
        self.high_revealed[index] = True
        self.tokens -= cost
        self._step_count += 1
        # Reward richer clues but discourage over-spending.
        return 0.15 - (cost / 200.0)

    def _terminal_reward(self) -> float:
        revealed = sum(self.low_revealed) + sum(self.high_revealed)
        if revealed < 2:
            return -1.0
        if self.last_action_was_skip:
            return 1.0 if self.is_trap else -1.0

        quality = self._estimated_quality()
        if self.is_trap:
            # Trap cards should be skipped; committing resources is penalized.
            return -0.6 + (self.tokens / 200.0) - (revealed / 20.0)

        score = 0.0
        score += 0.6 * quality
        score += self.tokens / 120.0
        score += 0.2 if sum(self.high_revealed) >= 1 else -0.1
        if revealed >= 5:
            score -= 0.1
        return max(-1.5, min(2.0, score))

    def _estimated_quality(self) -> float:
        ovr = self.player.ovr
        if ovr >= 97:
            base = 1.0
        elif ovr >= 94:
            base = 0.7
        elif ovr >= 91:
            base = 0.4
        else:
            base = 0.1
        if self.player.program == "ICON":
            base += 0.1
        if not self.player.tradable:
            base += 0.05
        return min(1.2, base)

    def _get_ovr_bucket(self, ovr: int) -> str:
        if 88 <= ovr <= 90:
            return "88-90"
        if 91 <= ovr <= 93:
            return "91-93"
        if 94 <= ovr <= 96:
            return "94-96"
        if 97 <= ovr <= 99:
            return "97-99"
        return "Unknown"

    def _generate_player(self) -> Tuple[PlayerProfile, bool]:
        is_trap = random.random() < 0.30
        nationality = random.choice(["Brazil", "France", "Argentina", "England", "Spain"])
        position = random.choice(["ST", "CM", "CB", "GK"])
        tradable = random.choice([True, False])

        ovr_bias = 0
        if nationality in ("Brazil", "France"):
            ovr_bias += 1
        if position == "ST":
            ovr_bias += 1
        if not tradable:
            ovr_bias += 1

        if is_trap:
            return (
                PlayerProfile(
                    ovr=random.randint(88, 91),
                    program="ICON",
                    tradable=tradable,
                    nationality=nationality,
                    position=position,
                    extra_label="era",
                    extra_value=random.choice(["90s", "2000s", "2010s"]),
                ),
                True,
            )

        program = random.choice(["ICON", "HERO", "REGULAR"])
        if program == "ICON":
            ovr = min(99, random.randint(95, 99) + min(ovr_bias, 1))
            return (
                PlayerProfile(
                    ovr=ovr,
                    program=program,
                    tradable=tradable,
                    nationality=nationality,
                    position=position,
                    extra_label="era",
                    extra_value=random.choice(["90s", "2000s", "2010s"]),
                ),
                False,
            )
        if program == "HERO":
            ovr = min(96, random.randint(92, 96) + min(ovr_bias, 1))
            return (
                PlayerProfile(
                    ovr=ovr,
                    program=program,
                    tradable=tradable,
                    nationality=nationality,
                    position=position,
                    extra_label="era",
                    extra_value=random.choice(["90s", "2000s", "2010s"]),
                ),
                False,
            )

        ovr = min(94, random.randint(88, 94) + ovr_bias)
        return (
            PlayerProfile(
                ovr=ovr,
                program=program,
                tradable=tradable,
                nationality=nationality,
                position=position,
                extra_label="team",
                extra_value=random.choice(
                    ["Real Madrid", "Barcelona", "Manchester City", "Liverpool", "Bayern Munich"]
                ),
            ),
            False,
        )

    def _generate_clues(
        self, player: PlayerProfile, is_trap: bool
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        noisy_ovr = random.choice([95, 97, 98]) if is_trap else player.ovr + random.choice([-1, 0, 1])
        low_clues: List[Tuple[str, str]] = [
            ("nationality", player.nationality),
            ("position", player.position),
            ("tradable", str(player.tradable)),
        ]
        if is_trap:
            i = random.randint(0, 2)
            if i == 0:
                low_clues[i] = ("nationality", random.choice(["Brazil", "France", "Argentina", "England", "Spain"]))
            elif i == 1:
                low_clues[i] = ("position", random.choice(["ST", "CM", "CB", "GK"]))
            else:
                low_clues[i] = ("tradable", str(random.choice([True, False])))

        high_clues: List[Tuple[str, str]] = [
            ("program", player.program),
            ("ovr_range", self._get_ovr_bucket(noisy_ovr)),
            (player.extra_label, player.extra_value),
        ]
        random.shuffle(low_clues)
        random.shuffle(high_clues)
        return low_clues, high_clues
