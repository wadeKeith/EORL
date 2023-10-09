import torch
import numpy as np
from agent_env import AgentEnv
from ppo import PolicyNetContinuous
from vehicle_utils import  pb_cal
import matplotlib.pyplot as plt

env = AgentEnv()


seed = 429
hidden_dim = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间

model_num = 89
agent_test = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
agent_test.load_state_dict(torch.load("./model/ppo_continuous_%d.pkl" % model_num))
agent_test.eval()


# test
reward_all = []

state = env.reset()

done = False
reward_ls = []
pb_ls = []
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
    pb = pb_cal(
                env.env.vehicle.motor_eff_2d,
                env.env.vehicle.force,
                env.env.vehicle.x_dot,
                env.env.vehicle.soc,
                env.env.vehicle.r_w,
                env.env.vehicle.battery_eff_dis_1d,
                env.env.vehicle.battery_eff_cha_1d,
            )[0]
    pb_ls.append(pb/1000)
    reward_ls.append(reward)
    env.render()
    state = next_obs
    num += 1
# plt.plot(np.linspace(1, len(pb_ls), len(pb_ls)), pb_ls)
# plt.xlabel("time(0.1s)")
# plt.ylabel("power(kW)")
# plt.show()
energy = 0
for i in range(0, len(pb_ls)):
    energy = pb_ls[i] * 0.1 + energy
print("energy: ", energy/3600)
print("reward: ", np.sum(reward_ls))
print("num: ", num)
print(info)
