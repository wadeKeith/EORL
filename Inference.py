import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from agent_env import AgentEnv
from ppo import PolicyNetContinuous

env = AgentEnv()

actor_lr = 2e-5
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


agent_test =  PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
agent_test.load_state_dict(torch.load("E:\Github\EORL\model\ppo_continuous_51.pth"))

# test
reward_all = []

state = env.reset()
state = torch.tensor(state, dtype=torch.float).to(device)
done = False
reward_ls = []
num = 0
while not done:
    action,_ = agent_test.forward(state)
    action = action.clamp(-1.0, 1.0)
    action = action.cpu().detach().numpy().tolist()
    # print(action)
    next_obs, reward, done, info = env.step(action)
    reward_ls.append(reward)
    # env.render()
    state = next_obs
    state = torch.tensor(state, dtype=torch.float).to(device)
    num += 1

reward_all.append(np.sum(reward_ls))
print("reward: ", np.mean(reward_all[-10:]))
print("num: ", num)