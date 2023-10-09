import numpy as np
from gymnasium.spaces import Box
import math
from road_env import RoadEnv


def dictobs2listobs(obs,state_lower,state_upper):
    list_obs = []
    for key in obs.keys():
        if key == "xy_direction":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "xy_direction_next":
            for i in obs[key]:
                list_obs.append(i)
        else:
            list_obs.append(obs[key])
    for i in range(len(list_obs)):
        if i==1 or i==9:
            list_obs[i] = (list_obs[i] - state_lower[i]) / (state_upper[i] - state_lower[i])
        else:
            list_obs[i] = list_obs[i] / state_upper[i]
    return list_obs

def map_action(action, action_space):
    if action[0] <= 0 and action[1] <= 0:
        action = [
            abs(action_space.low[0]) * action[0],
            abs(action_space.low[1]) * action[1],
        ]
    elif action[0] <= 0 and action[1] >= 0:
        action = [
            abs(action_space.low[0]) * action[0],
            abs(action_space.high[1]) * action[1],
        ]
    elif action[0] >= 0 and action[1] <= 0:
        action = [
            abs(action_space.high[0]) * action[0],
            abs(action_space.low[1]) * action[1],
        ]
    elif action[0] >= 0 and action[1] >= 0:
        action = [
            abs(action_space.high[0]) * action[0],
            abs(action_space.high[1]) * action[1],
        ]
    assert action in action_space, "action is not in action space"
    return action


class AgentEnv(object):
    def __init__(self) -> None:
        self.env = RoadEnv()
        self.max_length = self.env.max_distant
        # define observation space and action space
        self.state_lower = np.array(
            [
                0,
                self.env.road_init_width,
                0,
                -1,
                math.radians(-25),
                -math.pi,
                0,
                math.radians(-5),
                0,
                self.env.road_init_width,
                0,
                -1,
                math.radians(-25),
                -math.pi,
                0,
                math.radians(-5),
            ]
            + [0,0]*16 
        )
        self.state_upper = np.array(
            [
                self.env.road_length,
                self.env.road_width * self.env.road_num+self.env.road_init_width,
                50,
                1,
                math.radians(25),
                math.pi,
                1,
                math.radians(5),
                self.env.road_length,
                self.env.road_width * self.env.road_num+self.env.road_init_width,
                50,
                1,
                math.radians(25),
                math.pi,
                1,
                math.radians(5),
            ]
            + [self.env.road_length,self.env.road_width+self.env.road_init_width]*16
        )
        self.observation_space = Box(
            low=self.state_lower,
            high=self.state_upper,
            shape=(len(self.state_lower),),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.array([-2, -math.radians(5)]),
            high=np.array([1.5, math.radians(5)]),
            dtype=np.float32,
        )
    def reset(self):
        obs = self.env.reset()
        obs = dictobs2listobs(obs,self.state_lower, self.state_upper)
        return obs

    def step(self, action):
        action = map_action(action, self.action_space)
        # print(action)
        next_obs, reward, done, info = self.env.step(action)
        # print(next_obs['x_dot'])
        # print(next_obs['x'])
        # print(reward)
        
        next_obs = dictobs2listobs(next_obs,self.state_lower, self.state_upper)
        # print(next_obs)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()


if __name__ == "__main__":
    env = AgentEnv()
    obs = env.reset()
    done = False
    # print(obs)
    # a =np.array([0, 0, 0, -1, math.radians(-25), -math.inf,0,-4800,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf,-math.inf])
    # print(a.shape)
    # b =np.array([1000.0, env.env.road_width*env.env.road_num,50,1,math.radians(25),-math.inf,1,4800,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf,math.inf])
    # print(b.shape)
    # print('')

    while not done:
        action = [0.5, 0]
        next_obs, reward, done, info = env.step(action)
        print(next_obs)
        env.render()
        # print(next_obs, reward, done, info)
        print(info)
