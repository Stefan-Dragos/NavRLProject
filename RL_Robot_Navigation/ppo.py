from network import FeedForwardNN
import gymnasium as gym
from torch.distributions import MultivariateNormal
import torch

class PPO:

    def __init__(self, env):
        #initalize env
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        #init hyperparameters
        self._init_hyperparameters()

        #get actor nn, obs --> action
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        #get critic nn, obs --> value
        self.critic = FeedForwardNN(self.obs_dim, 1)

        #Multivariate Normal Distribution (Gaussian) matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)



    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.timesteps_per_episode = 1600

        self.gamma = 0.95  #calculating rewards to go, discount factor


    def get_action(self, obs):

        mean = self.actor(obs)

        dist = MultivariateNormal(mean, self.cov_mat)  #create a gaussian distribution with mean and covariance matrix

        action = dist.sample()   #sample from distribution and get log probability
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()  #detach unnecessary information


    def learn(self, totalTimeSteps):
        currentTimeStep = 0  #current time step simulated

        while currentTimeStep < totalTimeSteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            V = self.evaluate(batch_obs)   #get obs value function

            A_k = batch_rtgs - V.detach()   #get advantage function 

            #normalize advantage function --> improved performance
            A_k = (A_k - A_k.mean())

    #data collection
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        timeStep = 0

        # go through each episode
        while timeStep < self.timesteps_per_batch:

            ep_rewards = []

            obs = self.env.reset()
            done = False

            for episodeTimeStep in range(self.timesteps_per_episode):
                timeStep += 1  #increase total batch timestep

                action, log_propability = self.get_action(obs)   #get action with actor

                obs, rew, done, trunc, info = self.env.step(action)  #deploy action

                #save data
                ep_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_propability)

                if done:
                    break
            
            batch_lens.append(episodeTimeStep + 1)  #add one bc steps start at 0
            batch_rews.append(ep_rewards)

        #reshape data into tensors, will need them like this for later
        batch_obs = torch.tensor(batch_obs, dtype=float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        #calculate rewards to go
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        #go backwards to maintain same order in batch rtgs
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0

            for rew in reversed(ep_rews):

                discounted_reward = rew + discounted_reward * self.gamma

                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, torch.float)

        return batch_rtgs


    def evaluate(self, batch_obs):
        V = self.critic(batch_obs.squeeze)
        return V