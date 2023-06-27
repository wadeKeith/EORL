import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from agent_env import AgentEnv
from ppo import PPOContinuous

env = AgentEnv()

actor_lr = 1e-5
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.95
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.manual_seed(811)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间



agent_test = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
agent_test.actor.load_state_dict(torch.load("ppo_continuous_actor1.pth"))

#test
state,done = env.reset()
while not done:
    action = agent_test.take_action(state)
    next_obs, reward, done, info = env.step(action)
    env.render()