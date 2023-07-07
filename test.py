import torch
import numpy as np
from ppo import PolicyNetContinuous




def evluation_policy(env,hidden_dim,device,model_num):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent_test = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
    agent_test.load_state_dict(torch.load("./model/ppo_continuous_%d.pkl" % model_num))
    agent_test.eval()

    state = env.reset()
    done = False
    reward_ls = []
    num = 1
    while not done:
        state = torch.tensor(state, dtype=torch.float).to(device)
        action, _ = agent_test(state)
        action = action.clamp(-1.0, 1.0)
        action = action.cpu().detach().numpy().tolist()
        next_obs, reward, done, info = env.step(action)
        reward_ls.append(reward)
        # env.render()
        state = next_obs
        num += 1

    print("reward: ", np.sum(reward_ls))
    print("num: ", num)




