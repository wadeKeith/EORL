import numpy as np
from gymnasium.spaces import Box
import math
from road_env import RoadEnv


def dictobs2listobs(obs):
    list_obs = []
    for key in obs.keys():
        if key == 'dmin':
            [list_obs.append(i) for i in obs[key]]
        else:
            list_obs.append(obs[key])
    return list_obs
def normalize_obs(obs,state_upper):
    new_obs = dictobs2listobs(obs)
    final_obs = []
    for i in range(len(new_obs)):
        final_obs.append(new_obs[i]/state_upper[i])
    return final_obs

def map_action(action,action_space):
    if action[0] <= 0 and action[1] <= 0:
        action = [abs(action_space.low[0]) * action[0], abs(action_space.low[1]) * action[1]]
    elif action[0] <= 0 and action[1] >= 0:
        action = [abs(action_space.low[0]) * action[0], abs(action_space.high[1]) * action[1]]
    elif action[0] >= 0 and action[1] <= 0:
        action = [abs(action_space.high[0]) * action[0], abs(action_space.low[1]) * action[1]]
    elif action[0] >= 0 and action[1] >= 0:
        action = [abs(action_space.high[0]) * action[0], abs(action_space.high[1]) * action[1]]
    return action
class AgentEnv(object):
    def __init__(self) -> None:
        self.env = RoadEnv()
        self.max_length =math.sqrt(self.env.road_width**2+self.env.road_length**2)
        # define observation space and action space
        self.state_lower = np.array([0, 0, 0, -1, math.radians(-25), -2*math.pi,0,-4800,0,0,0,0,0,0,0,0,0])
        self.state_upper = np.array([self.env.road_length,
                                     self.env.road_width*self.env.road_num,
                                     50,
                                     1,
                                     math.radians(25),
                                     2*math.pi,
                                     1,
                                     4800,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length,
                                     self.max_length])
        self.observation_space = Box(low=self.state_lower, high=self.state_upper, shape=(8+9,), dtype=np.float32)
        self.action_space = Box(low=np.array([-2.0, -math.radians(30)]), high=np.array([2.0, math.radians(30)]), dtype=np.float32)


    def reset(self):
        obs,done = self.env.reset()
        # obs_norm = normalize_obs(obs,self.state_upper)
        return normalize_obs(obs,self.state_upper),done

    def step(self, action):
        action = map_action(action,self.action_space)
        # print(action)
        next_obs, reward, done, info = self.env.step(action)
        return normalize_obs(next_obs,self.state_upper), reward, done, info

    def render(self):
        self.env.render()


if __name__ == "__main__":
    env = AgentEnv()
    obs,done = env.reset()
    print(obs)
    # a =np.array([0, 0, 0, -1, math.radians(-25), -math.inf,0,-4800,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf])
    # print(a.shape)
    # b =np.array([1000.0, env.env.road_width*env.env.road_num,50,1,math.radians(25),-math.inf,1,4800,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf])
    # print(b.shape)
    # print('')

    while not done:
        action = [0.1, 0.1]
        next_obs, reward, done, info = env.step(action)
        env.render()
        # print(next_obs, reward, done, info)
