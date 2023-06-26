import numpy as np
from gymnasium.spaces import Box

from road_env import RoadEnv


def dictobs2listobs(obs):
    list_obs = []
    for key in obs.keys():
        list_obs.append(obs[key])
    return list_obs


class AgentEnv(object):
    def __init__(self) -> None:
        # define observation space and action space
        self.observation_space = Box(low=-1.0, high=2.0, shape=(8,), dtype=np.float32)
        # self.observation_space = Box(low=np.array([0, 0, -10, -10, -10, ]), high=np.array([1000.0, 7.0]), dtype=np.float32)
        self.action_space = Box(low=np.array([-3.0, -2.0]), high=np.array([2.0, 2.0]), dtype=np.float32)
        self.env = RoadEnv()

    def reset(self):
        obs = self.env.reset()
        return np.array(dictobs2listobs(obs))

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return np.array(dictobs2listobs(next_obs)), reward, done, info


if __name__ == "__main__":
    env = AgentEnv()
    obs = env.reset()
    done = False
    while not done:
        action = [0.1, 0.1]
        next_obs, reward, done, info = env.step(action)
        print(next_obs, reward, done, info)
