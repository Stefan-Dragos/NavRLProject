from NavEnvironment import NavEnv
from NavEnvironmentV2 import NavEnvV2
from NavEnvironmentV3 import NavEnvV3
from NavEnvironmentV4 import NavEnvV4
from NavEnvironmentV4 import NavEnvV4_Custom

import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = NavEnvV4_Custom(8, 7, 10, 600, "human")

#model = PPO("MlpPolicy", env=env, verbose=1)
#model.learn(total_timesteps=10000)

env.metadata["render_fps"] = 8

observation, info = env.reset()

for i in range(1000):

    print(observation)
    obs = np.array(observation ,dtype = np.float32)
    print(obs.dtype)

    action = env.action_space.sample()

    observation, reward, term, trunc, info = env.step(action)

    print("---------------------------------------------")

    #print(info)
    #print(env.angularVel)

    if term or trunc:
        observation, info = env.reset(seed=2)

env.showRewardPlot(1000)

