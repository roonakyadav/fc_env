import gymnasium as gym
from gymnasium import spaces
import numpy as np

from fc_env_environment import FcEnvironment
from schemas import FcAction


class FCGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = FcEnvironment()

        # Actions: 0=LOW, 1=HIGH, 2=STOP, 3=SKIP
        self.action_space = spaces.Discrete(4)

        # Observation: 6 clues + tokens + step + low_rem + high_rem
        self.observation_space = spaces.Box(
            low=0.0,
            high=5.0,
            shape=(10,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._convert_obs(obs), {}

    def step(self, action):
        obs = self.env.step(FcAction(action=action))

        return (
            self._convert_obs(obs),
            obs.reward,
            obs.done,
            False,  # truncated
            {}
        )

    def _convert_obs(self, obs):
        import numpy as np

        encoded = []

        for clue in obs.revealed_clues:
            if clue == "HIDDEN":
                encoded.append(-1.0)
            else:
                value = 1.0

                if "ICON" in clue:
                    value = 5.0
                elif "HERO" in clue:
                    value = 4.0
                elif "97-99" in clue:
                    value = 3.5
                elif "94-96" in clue:
                    value = 3.0
                elif "91-93" in clue:
                    value = 2.5

                # ADDITIVE signals (don't use elif)
                if "False" in clue:
                    value += 0.3  # non-tradable
                if "ST" in clue:
                    value += 0.2

                encoded.append(value)

        encoded.append((obs.tokens / 100)*5)
        encoded.append((obs.step_number / 10)*5)
        encoded.append(obs.low_remaining / 3)
        encoded.append(obs.high_remaining / 3)

        return np.array(encoded, dtype=np.float32)