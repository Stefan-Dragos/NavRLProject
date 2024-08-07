from NavEnvironment import NavEnv
from NavEnvironmentV2 import NavEnvV2
from NavEnvironmentV3 import NavEnvV3
from NavEnvironmentV4 import NavEnvV4
from NavEnvironmentV4 import NavEnvV4_Custom

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import TD3

env = NavEnvV4_Custom(10, 5, 10,600,"human")
env2 = NavEnvV4_Custom(5, 5, 10, 600, "human")

#env.metadata["render_fps"] = 4

model = SAC.load("FINAL_MODELS/FINALPPOent0.0001_lr0.0003", env=env)  #use the same model on varying amounts of hazards

obs1, info = env.reset()
obs2, info = env2.reset()

for i in range(1000):

    action1, _ = model.predict(obs1, deterministic=True)
    action2, _ = model.predict(obs2, deterministic=True)

    obs1, reward, term1, trunc1, info = env.step(action1)
    obs2, reward, term2, trunc2, info = env2.step(action2)

    #print(reward)
    #print(f" Distance penalty {env.agentDistanceTo(env.target_position) / 500} ")
    #print(obs)

    if term1 or trunc1:
        print("-------RESET--1------")
        obs1, info = env.reset()

    if term2 or trunc2:
        print("-------RESET--2------")
        obs2, info = env2.reset()