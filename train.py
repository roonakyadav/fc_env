from stable_baselines3 import PPO
from gym_wrapper import FCGymEnv

env = FCGymEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/ppo_fc_env"
)

model.learn(total_timesteps=300000)

model.save("./models/ppo_fc_env")