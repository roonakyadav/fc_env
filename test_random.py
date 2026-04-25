from server.fc_env_environment import FcEnvironment
from models import FcAction
import random

env = FcEnvironment()

episodes = 200
total_reward = 0
max_reward = -999

for i in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action = random.randint(0, 3)
        obs = env.step(FcAction(action=action))
        done = obs.done

    total_reward += obs.reward
    max_reward = max(max_reward, obs.reward)

print("Average Reward:", total_reward / episodes)
print("Max Reward:", max_reward)