from NavEnvironment import NavEnv
from NavEnvironmentV2 import NavEnvV2
from NavEnvironmentV3 import NavEnvV3
from NavEnvironmentV4 import NavEnvV4
from NavEnvironmentV4 import NavEnvV4_Custom

import gymnasium as gym
from stable_baselines3 import PPO

env = NavEnvV4_Custom(10, 5, 10,600,"human")

#env.metadata["render_fps"] = 4

model = PPO.load("FINAL_MODELS/FINALPPOent0.0001_lr0.0003", env=env)

obs, info = env.reset()

for i in range(1000):

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, term, trunc, info = env.step(action)

    #print(reward)
    #print(f" Distance penalty {env.agentDistanceTo(env.target_position) / 500} ")
    print(obs)

    if term or trunc:
        print("-------RESET------")
        obs, info = env.reset()