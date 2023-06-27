import torch

from agent_env import AgentEnv
from ppo import PPOContinuous, train_on_policy_agent

have_model = True
render_flag = False


env = AgentEnv()

actor_lr = 5e-5
critic_lr = 5e-4
num_episodes = 20000
hidden_dim = 128
gamma = 0.95
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(811)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

if have_model:
    agent.actor.load_state_dict(torch.load("ppo_continuous_actor.pth"))
# train
return_list = train_on_policy_agent(env, agent, num_episodes,render_flag)
# save model
torch.save(agent.actor.state_dict(), "ppo_continuous_actor1.pth")


