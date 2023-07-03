import numpy as np
from gymnasium.spaces import Box
import math
from road_env import RoadEnv


def dictobs2listobs(obs, upper):
    list_obs = []
    for key in obs.keys():
        if key == "dmin":
            [list_obs.append(i / 1000) for i in obs[key]]
        elif key == "x_min":
            [list_obs.append(i / obs["dmin"][obs[key].index(i)]) for i in obs[key]]
        elif key == "y_min":
            [list_obs.append(i / obs["dmin"][obs[key].index(i)]) for i in obs[key]]
        elif key == "dmin_next":
            [list_obs.append(i / 1000) for i in obs[key]]
        elif key == "x_min_next":
            [list_obs.append(i / obs["dmin_next"][obs[key].index(i)]) for i in obs[key]]
        elif key == "y_min_next":
            [list_obs.append(i / obs["dmin_next"][obs[key].index(i)]) for i in obs[key]]
        else:
            list_obs.append(obs[key] / upper[key])
    return list_obs


# def normalize_obs(obs, state_upper):
#     new_obs = dictobs2listobs(obs)
#     final_obs = [new_obs[i] / state_upper[i] for i in range(len(new_obs))]
#     return final_obs


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
                0,
                0,
                -1,
                math.radians(-25),
                -math.pi,
                0,
                -4800,
                0,
                0,
                0,
                -1,
                math.radians(-25),
                -math.pi,
                0,
                -4800,
            ]
        )
        self.state_upper = np.array(
            [
                self.env.road_length,
                self.env.road_width * self.env.road_num,
                50,
                1,
                math.radians(25),
                math.pi,
                1,
                4800,
                self.env.road_length,
                self.env.road_width * self.env.road_num,
                50,
                1,
                math.radians(25),
                math.pi,
                1,
                4800,
            ]
        )
        self.normlization = {
            "x": self.env.road_length,
            "y": self.env.road_width * self.env.road_num,
            "x_dot": 50,
            "y_dot": 1,
            "phi": math.radians(25),
            "omega": math.pi,
            "soc": 1,
            "force": 4800,
            "x_next": self.env.road_length,
            "y_next": self.env.road_width * self.env.road_num,
            "x_dot_next": 50,
            "y_dot_next": 1,
            "phi_next": math.radians(25),
            "omega_next": math.pi,
            "soc_next": 1,
            "force_next": 4800,
        }
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
        return dictobs2listobs(obs, self.normlization)

    def step(self, action):
        action = map_action(action, self.action_space)
        # print(action)
        next_obs, reward, done, info = self.env.step(action)
        # print(next_obs['x_dot'])
        # print(next_obs['x'])
        # print(reward)
        return dictobs2listobs(next_obs, self.normlization), reward, done, info

    def render(self):
        self.env.render()


if __name__ == "__main__":
    env = AgentEnv()
    obs = env.reset()
    done = False
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
