import torch
import numpy as np
from agent_env import AgentEnv
from ppo import PolicyNetContinuous
from utils import loadclass

env = AgentEnv()


seed = 4443
hidden_dim = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间

model_num = 0
agent_test = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
agent_test.load_state_dict(torch.load("./model/ppo_continuous_%d.pkl" % model_num))
agent_test.eval()

env.state_norm = loadclass("./model/text_file/state_norm_%d" % model_num)
env.state_norm_flag = False
# test
reward_all = []

state = env.reset()
done = False
reward_ls = []
num = 1
while not done:
    # print(state)
    state = torch.tensor(state, dtype=torch.float).to(device)
    action, _ = agent_test(state)
    action = action.clamp(-1.0, 1.0)
    # print(action)
    action = action.cpu().detach().numpy().tolist()
    # print(action)
    next_obs, reward, done, info = env.step(action)
    reward_ls.append(reward)
    env.render()
    state = next_obs
    num += 1

print("reward: ", np.sum(reward_ls))
print("num: ", num)
