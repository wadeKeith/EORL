import math
import numpy as np


class VehicleEnv(object):
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.x_dot = None
        self.y_dot = None
        self.phi = None  # 角度
        self.omega = None  # 角速度

        # next time step
        self.x_next = None
        self.y_next = None
        self.x_dot_next = None
        self.y_dot_next = None
        self.phi_next = None  # 下一时刻角度
        self.omega_next = None  # 下一时刻角速度

        self.x_ddot = None
        self.delta_t = 1  # simulate time step

        # parameters for vehicle
        self.m = 2500
        self.c_f = -4e5
        self.c_r = -1.6e6
        self.D = self.c_f + self.c_r
        self.a_v = 3.4
        self.b_v = 1.4
        self.K = self.a_v * self.c_f - self.b_v * self.c_r
        self.W = self.a_v**2 * self.c_f + self.b_v**2 * self.c_r
        self.I_zz = 9.3e5

    def reset(self):
        self.x = 0
        self.y = 0
        self.x_dot = 0
        self.y_dot = 0
        self.phi = 0  # 角度
        self.omega = 0  # 角速度

        # next time step
        self.x_next = 0
        self.y_next = 0
        self.x_dot_next = 0
        self.y_dot_next = 0
        self.phi_next = 0  # 角度
        self.omega_next = 0  # 角速度
        return np.array([self.x_next, self.y_next, self.x_dot_next, self.y_dot_next, self.phi_next, self.omega_next])

    def step(self, action):
        assert isinstance(action, list), "action must be a list"
        self.x_next = self.x + self.delta_t * (self.x_dot * math.cos(self.phi) - self.y_dot * math.sin(self.phi))
        self.y_next = self.y + self.delta_t * (self.x_dot * math.sin(self.phi) + self.y_dot * math.cos(self.phi))
        self.x_dot_next = self.x_dot + self.delta_t * (action[0] - self.y_dot * self.omega)
        self.y_dot_next = (
            2 * self.m * self.x_dot * self.y_dot
            + self.D * self.delta_t * self.y_dot
            + 2 * self.K * self.omega * self.delta_t
            - 2 * self.m * self.x_dot**2 * self.omega * self.delta_t
            - 2 * self.c_f * self.x_dot * action[1] * self.delta_t
        ) / (2 * self.m * self.x_dot - self.D * self.delta_t)
        self.phi_next = self.phi + self.delta_t * self.omega
        self.omega_next = (
            2 * self.I_zz * self.x_dot * self.omega
            + self.W * self.delta_t * self.omega
            + 2 * self.K * self.y_dot * self.delta_t
            - 2 * self.x_dot * self.a_v * self.c_f * action[1] * self.delta_t
        ) / (2 * self.I_zz * self.x_dot - self.W * self.delta_t)

        # update state and relate info
        self.x = self.x_next
        self.y = self.y_next
        self.x_dot = self.x_dot_next
        self.y_dot = self.y_dot_next
        self.phi = self.phi_next
        self.omega = self.omega_next

        # Return State, Reward, Done, Info
        return_state = np.array(
            [self.x_next, self.y_next, self.x_dot_next, self.y_dot_next, self.phi_next, self.omega_next]
        )
        reward = 0
        done = False
        info = {}
        return return_state, reward, done, info


if __name__ == "__main__":
    env = VehicleEnv()

    obs = env.reset()
    done = False

    obs_lists = []

    for i in range(10000):
        obs_lists.append(obs)
        if i == 0:
            action = [1, 0]
        else:
            action = [0, 0]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs

    x_plot = [obs[0] for obs in obs_lists]
    y_plot = [obs[1] for obs in obs_lists]
    x_dot_plot = [obs[2] for obs in obs_lists]
    y_dot_plot = [obs[3] for obs in obs_lists]
    phi_plot = [obs[4] for obs in obs_lists]
    omega_plot = [obs[5] for obs in obs_lists]

    from utils import plot_list

    plot_list(x_plot, "x_plot")
    plot_list(y_plot, "y_plot")
    plot_list(x_dot_plot, "x_dot_plot")
    plot_list(y_dot_plot, "y_dot_plot")
    plot_list(phi_plot, "phi_plot")
    plot_list(omega_plot, "omega_plot")
    print("")
