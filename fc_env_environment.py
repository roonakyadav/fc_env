# FC Mobile RL Environment (in-process)
import random
from dataclasses import dataclass
from uuid import uuid4

from schemas import FcAction, FcObservation


@dataclass
class State:
    episode_id: str
    step_count: int = 0


class FcEnvironment:
    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.max_steps = 8
        self.reset()

    def reset(self) -> FcObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.player, self.is_trap = self._generate_player()
        self.low_clues, self.high_clues = self._generate_clues(self.player, self.is_trap)

        self.low_revealed = [False] * 3
        self.high_revealed = [False] * 3
        self.high_clues_taken = 0

        self.tokens = 100
        self.done = False
        self.last_action_was_skip = False

        low_remaining = self.low_revealed.count(False)
        high_remaining = self.high_revealed.count(False)

        return FcObservation(
            revealed_clues=self._get_visible_clues(),
            tokens=self.tokens,
            step_number=sum(self.low_revealed) + sum(self.high_revealed),
            low_remaining=low_remaining,
            high_remaining=high_remaining,
            done=False,
            reward=0.0,
        )

    def _get_rarity(self, ovr):
        if 97 <= ovr <= 99:
            return "ELITE"
        if 94 <= ovr <= 96:
            return "RARE"
        if 91 <= ovr <= 93:
            return "UNCOMMON"
        return "COMMON"

    def _generate_player(self):
        is_trap = random.random() < 0.3

        nationality = random.choice(["Brazil", "France", "Argentina", "England", "Spain"])
        position = random.choice(["ST", "CM", "CB", "GK"])
        tradable = random.choice([True, False])

        ovr_bias = 0
        if nationality in ["Brazil", "France"]:
            ovr_bias += 1
        if position == "ST":
            ovr_bias += 1
        if not tradable:
            ovr_bias += 1

        if is_trap:
            program = "ICON"
            ovr = random.randint(88, 91)
            self.rarity = "ELITE"
            extra_field = {"era": random.choice(["2000s", "2010s", "90s"])}
        else:
            program = random.choice(["ICON", "HERO", "REGULAR"])
            if program == "ICON":
                ovr = random.randint(95, 99) + min(ovr_bias, 1)
                ovr = min(ovr, 99)
                extra_field = {"era": random.choice(["2000s", "2010s", "90s"])}
            elif program == "HERO":
                ovr = random.randint(92, 96) + min(ovr_bias, 1)
                ovr = min(ovr, 96)
                extra_field = {"era": random.choice(["2000s", "2010s", "90s"])}
            else:
                ovr = random.randint(88, 94) + ovr_bias
                ovr = min(ovr, 94)
                extra_field = {
                    "team": random.choice(
                        ["Real Madrid", "Barcelona", "Manchester City", "Liverpool", "Bayern Munich"]
                    )
                }
            self.rarity = self._get_rarity(ovr)

        player = {
            "ovr": ovr,
            "program": program,
            "tradable": tradable,
            "nationality": nationality,
            "position": position,
        }
        player.update(extra_field)
        return player, is_trap

    def _get_ovr_bucket(self, ovr):
        if 88 <= ovr <= 90:
            return "88-90"
        if 91 <= ovr <= 93:
            return "91-93"
        if 94 <= ovr <= 96:
            return "94-96"
        if 97 <= ovr <= 99:
            return "97-99"
        return "Unknown"

    def _generate_clues(self, player, is_trap):
        if is_trap:
            noisy_ovr = random.choice([95, 97, 98])
        else:
            noisy_ovr = player["ovr"] + random.choice([-1, 0, 1])

        low_clues = [
            ["nationality", player["nationality"]],
            ["position", player["position"]],
            ["tradable", player["tradable"]],
        ]

        if is_trap:
            idx_to_flip = random.randint(0, 2)
            if idx_to_flip == 0:
                low_clues[0][1] = random.choice(["Brazil", "France", "Argentina", "England", "Spain"])
            elif idx_to_flip == 1:
                low_clues[1][1] = random.choice(["ST", "CM", "CB", "GK"])
            else:
                low_clues[2][1] = random.choice([True, False])

        low_clues = [tuple(c) for c in low_clues]

        high_clues = [
            ("program", player["program"]),
            ("ovr_range", self._get_ovr_bucket(noisy_ovr)),
        ]
        if player["program"] == "REGULAR":
            high_clues.append(("team", player["team"]))
        else:
            high_clues.append(("era", player["era"]))

        random.shuffle(low_clues)
        random.shuffle(high_clues)
        return low_clues, high_clues

    def _get_visible_clues(self):
        low_visible = [
            str(self.low_clues[i]) if self.low_revealed[i] else "HIDDEN" for i in range(3)
        ]
        high_visible = [
            str(self.high_clues[i]) if self.high_revealed[i] else "HIDDEN" for i in range(3)
        ]
        return low_visible + high_visible

    def step(self, action: FcAction) -> FcObservation:
        if self.done:
            low_remaining = self.low_revealed.count(False)
            high_remaining = self.high_revealed.count(False)
            return FcObservation(
                revealed_clues=self._get_visible_clues(),
                tokens=self.tokens,
                step_number=sum(self.low_revealed) + sum(self.high_revealed),
                low_remaining=low_remaining,
                high_remaining=high_remaining,
                done=True,
                reward=0.0,
            )

        act = action.action

        if act in [0, 1] and self.tokens <= 0:
            self.done = True
            act = 2
            self._state.step_count += 1
        else:
            if act == 0:
                unrevealed = [i for i, r in enumerate(self.low_revealed) if not r]
                if unrevealed:
                    idx = random.choice(unrevealed)
                    self.low_revealed[idx] = True
                    self.tokens -= 5
                    self._state.step_count += 1
                else:
                    self.tokens -= 2

            elif act == 1:
                unrevealed = [i for i, r in enumerate(self.high_revealed) if not r]
                if unrevealed:
                    remaining_high = len(unrevealed)
                    if remaining_high == 3:
                        cost = 15
                    elif remaining_high == 2:
                        cost = 20
                    else:
                        cost = 25

                    idx = random.choice(unrevealed)
                    self.high_revealed[idx] = True
                    self.high_clues_taken += 1
                    self.tokens -= cost
                    self._state.step_count += 1
                else:
                    self.tokens -= 5

            elif act == 2:
                self.done = True
                self._state.step_count += 1

            elif act == 3:
                self.done = True
                self.last_action_was_skip = True
                self._state.step_count += 1

        if self.tokens <= 0:
            self.done = True

        if self._state.step_count >= self.max_steps:
            self.done = True

        if all(self.low_revealed) and all(self.high_revealed):
            self.done = True

        self.tokens = max(self.tokens, 0)

        reward = 0.0

        if self.done:
            reward = self._calculate_reward()

        low_remaining = self.low_revealed.count(False)
        high_remaining = self.high_revealed.count(False)

        return FcObservation(
            revealed_clues=self._get_visible_clues(),
            tokens=self.tokens,
            step_number=sum(self.low_revealed) + sum(self.high_revealed),
            low_remaining=low_remaining,
            high_remaining=high_remaining,
            done=self.done,
            reward=reward,
        )

    def _calculate_reward(self):
        if self.last_action_was_skip:
            if self.is_trap:
                return 20.0
            return -30.0

        if self.tokens == 0 and not self.last_action_was_skip:
            return -20.0

        if self.is_trap:
            correct_solution = False
            wrong_solution = True
        else:
            correct_solution = True
            wrong_solution = False

        ovr = self.player["ovr"]

        if 97 <= ovr <= 99:
            base = 120
        elif 94 <= ovr <= 96:
            base = 80
        elif 91 <= ovr <= 93:
            base = 40
        else:
            base = 10

        spent = (100 - self.tokens) * 1.2
        efficiency_bonus = self.tokens * 0.1

        low_used = sum(self.low_revealed)
        high_used = sum(self.high_revealed)
        revealed_count = low_used + high_used

        if revealed_count < 2:
            return -40.0

        early_stop_penalty = 0
        uncertainty_penalty = 0

        if revealed_count == 2:
            uncertainty_penalty = 5

        saturation_penalty = 20 if revealed_count >= 5 else 0

        high_info_bonus = 4 if high_used >= 1 else 0
        low_usage_bonus = min(low_used, 2)

        noise = 0

        reward = (
            base
            - spent
            + efficiency_bonus
            - early_stop_penalty
            - uncertainty_penalty
            - saturation_penalty
            + low_usage_bonus
            + high_info_bonus
            + noise
        )

        if correct_solution:
            reward += 50
        if wrong_solution:
            reward -= 50

        visible_clues = str(self._get_visible_clues())
        if "97-99" in visible_clues:
            reward += 15
        elif "94-96" in visible_clues:
            reward += 8

        if "ICON" in visible_clues:
            reward += 6

        return float(reward)

    @property
    def state(self) -> State:
        return self._state
