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
torch.manual_seed(4444)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间


agent_test = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
agent_test.actor.load_state_dict(torch.load("E:\Github\EORL\model\ppo_continuous_97.pth"))

# test
reward_all = []
for i in range(10):
    state, done = env.reset()
    reward_ls = []

    while not done:
        action = agent_test.take_action(state)
        next_obs, reward, done, info = env.step(action)
        reward_ls.append(reward)
        env.render()
    reward_all.append(np.sum(reward_ls))
    print("reward: ", np.mean(reward_all[-10:]))