from NavEnvironment import NavEnv
from NavEnvironmentV2 import NavEnvV2
from NavEnvironmentV3 import NavEnvV3
from NavEnvironmentV4 import NavEnvV4
from NavEnvironmentV4 import NavEnvV4_Custom

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import DDPG

from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3 import TD3
#from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback   #check back in with model while training
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import torch.utils.tensorboard

import os
import numpy as np

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


#TODO: Train with more hazards for more efficient model
#Gives it more difficult situations to learn better (eg. 8 hazard)
#TODO: Train multiple different agents (A2C, DDPG, SAC, TD3) and apply HER algorithm with some of these agents
#Save all data in new board, and save tests under "NavEnvV4_board"

def make_env(rank, seed = 0):

    def _init():
        env = NavEnvV4(5, 10, 600, None)  #make specific env here
        #env = MaxAndSkipEnv(env, 4)
        env.seed = seed + rank
        return env
    
    set_random_seed(seed)
    return _init


def main():
    log_dir = "logdirRL/"
    os.makedirs(log_dir, exist_ok=True)

    num_envs = 4

    #------------------HyperParameters-----------------------------------
    timeSteps = 500000
    learningRate = 0.0003
    entC = 0.0001
    #--------------------------------------------------------------------

    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(num_envs)]), "logdirRl/monitor")

    model = DDPG("MlpPolicy", env, verbose=1, learning_rate=learningRate, tensorboard_log="./VaryModels_board/")

    print("--------Started Learning-----------")

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model.learn(total_timesteps=timeSteps, callback=callback, tb_log_name=f"DDPG_LR{learningRate}_Steps{timeSteps}_5H")  #_ent{entC}

    model.save("SavedRLModels/DDPG_lr0.0003")

    print("---------Finished Learning------------")


if __name__ == "__main__":
    main()