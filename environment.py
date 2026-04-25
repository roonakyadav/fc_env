import random
from dataclasses import dataclass
from typing import List, Tuple
from uuid import uuid4

from models import Action, Observation, State

# Display names for step().info (matches Gradio)
STEP_ACTION_NAMES: dict[int, str] = {
    0: "Reveal Low",
    1: "Reveal High",
    2: "Commit",
    3: "Refresh",
}


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
        return self._observation(0.0)

    def step(self, action: Action) -> Observation:
        if self.done:
            return self._observation(
                0.0,
                {
                    "tokens_left": int(self.tokens),
                    "step_reward": 0.0,
                    "step_number": int(self._step_count),
                    "action_name": "—",
                    "reason": "",
                },
            )

        act = action.action
        reward = 0.0

        # ---------- ACTIONS ----------
        if act == 0:
            reward += self._reveal_low_clue()
        elif act == 1:
            reward += self._reveal_high_clue()
        elif act == 2:
            self.done = True
            self._step_count += 1
        elif act == 3:
            self.done = True
            self.last_action_was_skip = True
            self._step_count += 1

        # ---------- TERMINATION ----------
        if self.tokens <= 0:
            self.done = True

        if self._step_count >= self.max_steps:
            self.done = True

        if all(self.low_revealed) and all(self.high_revealed):
            self.done = True

        # ---------- FINAL REWARD ----------
        if self.done:
            reward += self._terminal_reward()

        self.tokens = max(self.tokens, 0)
        self._update_state()
        # info: step_reward and counts for API / Live Stats UI (reward math unchanged)
        reason = self._get_reward_reason(act, float(reward))
        info = {
            "tokens_left": int(self.tokens),
            "step_reward": float(reward),
            "step_number": int(self._step_count),
            "action_name": STEP_ACTION_NAMES.get(act, f"Action {act}"),
            "reason": reason,
        }
        return self._observation(reward, info)

    def _get_reward_reason(self, action: int, reward: float) -> str:
        """Short UI explanation; does not change reward values."""
        if action == 0:  # Reveal Low
            return "Cheap info gained"
        if action == 1:  # Reveal High
            return "Strong signal (higher cost)"
        if action == 2:  # Commit
            if reward > 0:
                return "Correct decision"
            return "Bad commit timing"
        if action == 3:  # Refresh
            return "Skipped candidate (penalty applied)"
        return ""

    def state(self) -> State:
        return self.state_snapshot

    # ==============================
    # 🔥 REWARD SYSTEM (FIXED)
    # ==============================

    def _terminal_reward(self) -> float:
        revealed = sum(self.low_revealed) + sum(self.high_revealed)
        token_ratio = self.tokens / 100.0
        quality = self._estimated_quality()

        # ---------- SKIP ----------
        if self.last_action_was_skip:
            if self.is_trap:
                return 1.0 + 0.3 * token_ratio
            else:
                return -1.2

        # ---------- TRAP COMMIT ----------
        if self.is_trap:
            return -1.3

        # ---------- FORCE INFORMATION USAGE ----------
        if revealed == 0:
            return -1.0   # 🚨 prevents instant commit
        if revealed == 1:
            return -0.5

        # ---------- NORMAL COMMIT ----------
        reward = 0.0

        reward += 1.0 * quality               # main signal
        reward += 0.25 * token_ratio          # reduced importance

        if revealed >= 2:
            reward += 0.2                     # reward for using info
        else:
            reward -= 0.3

        if sum(self.high_revealed) >= 1:
            reward += 0.1                     # encourage high clue

        return max(-1.5, min(1.5, reward))

    # ==============================
    # 🔍 REVEAL ACTIONS
    # ==============================

    def _reveal_low_clue(self) -> float:
        unrevealed = [i for i in range(3) if not self.low_revealed[i]]
        if not unrevealed:
            self.tokens -= 2
            return -0.2

        idx = random.choice(unrevealed)
        self.low_revealed[idx] = True

        self.tokens -= 5
        self._step_count += 1

        return 0.12   # 🔥 increased

    def _reveal_high_clue(self) -> float:
        unrevealed = [i for i in range(3) if not self.high_revealed[i]]
        if not unrevealed:
            self.tokens -= 5
            return -0.3

        remaining = len(unrevealed)

        cost = 15 if remaining == 3 else 20 if remaining == 2 else 25

        idx = random.choice(unrevealed)
        self.high_revealed[idx] = True

        self.tokens -= cost
        self._step_count += 1

        # 🔥 increased rewards
        if remaining == 3:
            return 0.15
        elif remaining == 2:
            return 0.12
        else:
            return 0.10

    # ==============================
    # 🧠 QUALITY
    # ==============================

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

    # ==============================
    # 🎲 GENERATION (UNCHANGED)
    # ==============================

    def _generate_player(self):
        is_trap = random.random() < 0.30

        nationality = random.choice(["Brazil", "France", "Argentina", "England", "Spain"])
        position = random.choice(["ST", "CM", "CB", "GK"])
        tradable = random.choice([True, False])

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
            ovr = random.randint(95, 99)
        elif program == "HERO":
            ovr = random.randint(92, 96)
        else:
            ovr = random.randint(88, 94)

        return (
            PlayerProfile(
                ovr=ovr,
                program=program,
                tradable=tradable,
                nationality=nationality,
                position=position,
                extra_label="team",
                extra_value=random.choice(["Real Madrid", "Barcelona", "City", "Liverpool"]),
            ),
            False,
        )

    def _generate_clues(self, player, is_trap):
        low = [
            ("nationality", player.nationality),
            ("position", player.position),
            ("tradable", str(player.tradable)),
        ]

        high = [
            ("program", player.program),
            ("ovr", str(player.ovr)),
            (player.extra_label, player.extra_value),
        ]

        random.shuffle(low)
        random.shuffle(high)

        return low, high

    # ==============================
    # 📦 STATE / OBS
    # ==============================

    def _visible_clues(self):
        low = [str(self.low_clues[i]) if self.low_revealed[i] else "HIDDEN" for i in range(3)]
        high = [str(self.high_clues[i]) if self.high_revealed[i] else "HIDDEN" for i in range(3)]
        return low + high

    def _observation(self, reward, step_info: dict | None = None):
        inf: dict = {} if step_info is None else dict(step_info)
        return Observation(
            revealed_clues=tuple(self._visible_clues()),
            tokens=self.tokens,
            step_number=self._step_count,
            low_remaining=self.low_revealed.count(False),
            high_remaining=self.high_revealed.count(False),
            done=self.done,
            reward=float(reward),
            info=inf,
        )

    def _update_state(self):
        self.state_snapshot = State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            tokens=self.tokens,
            done=self.done,
            low_revealed=sum(self.low_revealed),
            high_revealed=sum(self.high_revealed),
        )