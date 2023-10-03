import torch

from agent_env import AgentEnv
from ppo import PPOContinuous, train_on_policy_agent

have_model = 0
render_flag = 0

seed = 1111


actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 2000
hidden_dim = 64
gamma = 0.9999
lmbda = 0.95
epochs = 100
eps = 0.1
entropy_coef = 0.001
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

env = AgentEnv()

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, entropy_coef)

if have_model:
    agent.actor.load_state_dict(torch.load("E:\Github\EORL\ppo_continuous_base.pkl"))
# train
return_list = train_on_policy_agent(env, agent, num_episodes, render_flag)
