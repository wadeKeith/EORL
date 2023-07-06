import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from agent_env import AgentEnv


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOContinuous:
    """处理连续动作的PPO算法"""

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,entropy_coef):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.lr_a =actor_lr
        self.lr_c = critic_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr,eps=1e-5)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.entropy_coef = entropy_coef

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        action = action.clamp(-1.0, 1.0)
        # print(action)
        return action.cpu().numpy().tolist()[0]

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.float).view(-1, 2).to(self.device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = rewards  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        updata_size = 128
        for _ in range((states.size()[0]) // updata_size + 1):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            dist_entropy = action_dists.entropy().sum(1, keepdim=True)  # 计算熵
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2)- self.entropy_coef * dist_entropy # 计算actor的损失加入了熵
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.mean().backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.epochs)
        lr_c_now = self.lr_c * (1 - total_steps / self.epochs)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    adv = torch.tensor(advantage_list, dtype=torch.float).view(-1, 1)
    # adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
    return adv


def train_on_policy_agent(env, agent, num_episodes, render_flag=False):
    return_list = []
    num_ls = []
    transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
    max_return = 0
    for i in range(100):
        agent.lr_decay(i)
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            info_display = {'out of road': 0,
                            'speed illegal': 0,
                            'motor cant provide': 0,
                            'arrive': 0,
                            'collision': 0, }
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0

                state = env.reset()
                done = False
                num = 0
                while not done:
                    # print("state: ", state)
                    action = agent.take_action(state)
                    # print("action: ", action)
                    next_state, reward, done, info = env.step(action)
                    if render_flag:
                        env.render()
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["dones"].append(done)
                    state = next_state
                    episode_return += reward
                    num += 1

                return_list.append(episode_return)
                num_ls.append(num)
                for key,value in info.items():
                    if key=='out of road' and value == 1:
                        info_display['out of road'] += 1
                    elif key=='speed illegal' and value == 1:
                        info_display['speed illegal'] += 1
                    elif key=='motor cant provide' and value == 1:
                        info_display['motor cant provide'] += 1
                    elif key=='arrive' and value == 1:
                        info_display['arrive'] += 1
                    elif key=='collision' and value == 1:
                        info_display['collision'] += 1
                # if np.mean(return_list[-10:]) > max_return:
                #     torch.save(agent.actor.state_dict(),
                #                "./model/ppo_continuous_%d.pth" % i)
                #     max_return = np.mean(return_list[-10:])
                agent.update(transition_dict)
                transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                            "num_steps": "%.3f" % np.mean(num_ls[-10:]),
                            'learning rate': agent.actor_optimizer.param_groups[0]['lr'],
                            'info': info_display
                        }
                    )
                pbar.update(1)
        torch.save(agent.actor.state_dict(), "./model/ppo_continuous_%d.pkl" % i)
    return return_list

