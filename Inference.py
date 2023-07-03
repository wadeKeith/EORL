import torch
import numpy as np
from agent_env import AgentEnv,map_action
from ppo import PolicyNetContinuous

env = AgentEnv()


hidden_dim = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(4444)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间


agent_test =  PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
agent_test.load_state_dict(torch.load("E:\Github\EORL\model\ppo_continuous_6.pth"))

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
    action = map_action(action,env.action_space)
    # print(action)
    next_obs, reward, done, info = env.step(action)
    # print(next_obs)
    reward_ls.append(reward)
    env.render()
    state = next_obs
    state = torch.tensor(state, dtype=torch.float).to(device)
    num += 1

reward_all.append(np.sum(reward_ls))
print("reward: ", np.mean(reward_all[-10:]))
print("num: ", num)