import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from vehicle_utils import bat_dynamic, pb_cal


class VehicleEnv(object):
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.x_dot = None
        self.y_dot = None
        self.phi = None  # 角度
        self.omega = None  # 角速度
        self.force = None  # driving force
        self.soc = None  # state of charge

        # next time step
        self.x_next = None
        self.y_next = None
        self.x_dot_next = None
        self.y_dot_next = None
        self.phi_next = None  # 下一时刻角度
        self.omega_next = None  # 下一时刻角速度
        self.force_next = None  # 下一时刻驱动力
        self.soc_next = None  # 下一时刻电池电量

        # self.x_ddot = None
        self.delta_t = 0.05  # simulate time step

        # parameters for vehicle
        self.m = 1400
        self.c_f = -4e5
        self.c_r = -1.6e6
        self.D = self.c_f + self.c_r
        self.a_v = 3.4
        self.b_v = 1.4
        self.K = self.a_v * self.c_f - self.b_v * self.c_r
        self.W = self.a_v**2 * self.c_f + self.b_v**2 * self.c_r
        self.I_zz = 9.3e5
        self.g = 9.8
        self.tau_r = 0.01
        self.tau_a = 0.3
        self.rho_a = 1.223
        self.A_f = 2.0
        self.r_w = 0.25
        self.motor_eff_speed = np.array([0, 1000, 2000, 3000, 4000])
        self.motor_eff_torque = np.linspace(-1200, 1200, 21)
        self.motor_eff_eff = (
            np.array(
                [
                    [0.7, 0.78, 0.85, 0.86, 0.81],
                    [0.7, 0.78, 0.86, 0.87, 0.82],
                    [0.7, 0.79, 0.86, 0.88, 0.85],
                    [0.7, 0.80, 0.86, 0.89, 0.87],
                    [0.7, 0.81, 0.87, 0.90, 0.88],
                    [0.7, 0.82, 0.88, 0.90, 0.9],
                    [0.7, 0.82, 0.87, 0.90, 0.91],
                    [0.7, 0.82, 0.86, 0.90, 0.91],
                    [0.7, 0.81, 0.85, 0.89, 0.91],
                    [0.7, 0.77, 0.82, 0.87, 0.88],
                    [0.7, 0.75, 0.8, 0.85, 0.85],
                    [0.7, 0.77, 0.82, 0.87, 0.88],
                    [0.7, 0.81, 0.85, 0.89, 0.91],
                    [0.7, 0.82, 0.86, 0.90, 0.91],
                    [0.7, 0.82, 0.87, 0.90, 0.91],
                    [0.7, 0.82, 0.88, 0.90, 0.9],
                    [0.7, 0.81, 0.87, 0.90, 0.88],
                    [0.7, 0.80, 0.86, 0.89, 0.87],
                    [0.7, 0.79, 0.86, 0.88, 0.85],
                    [0.7, 0.78, 0.86, 0.87, 0.82],
                    [0.7, 0.78, 0.85, 0.86, 0.81],
                ]
            )
            * 1.09
        )
        self.motor_eff_2d = interpolate.RegularGridInterpolator(
            (self.motor_eff_torque, self.motor_eff_speed), self.motor_eff_eff
        )
        self.bat_eff_soc = np.array([0.1, 0.9])
        self.bat_eff_dis = np.array([0.65, 0.9])
        self.bat_eff_cha = np.array([0.9, 0.7])
        self.battery_eff_dis_1d = interpolate.RegularGridInterpolator((self.bat_eff_soc,), self.bat_eff_dis)
        self.battery_eff_cha_1d = interpolate.RegularGridInterpolator((self.bat_eff_soc,), self.bat_eff_cha)
        self.bat_q = 25 * 1000 * 3600  # 电池容量 25kwh

        self.theta = None

    def update_theta(self, theta):
        self.theta = theta

    def reset(self):
        self.x = 5
        self.y = 3.75*3/2
        self.x_dot = 30
        self.y_dot = 0
        self.phi = 0  # 角度
        self.omega = 0  # 角速度
        self.soc = 0.6  # state of charge
        self.force = (
            0 * self.m
            + self.m * self.g * self.tau_r * math.cos(self.theta)
            + self.m * self.g * math.sin(self.theta)
            + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
        )

        # next time step
        self.x_next = 0
        self.y_next = 0
        self.x_dot_next = 0
        self.y_dot_next = 0
        self.phi_next = 0  # 角度
        self.omega_next = 0  # 角速度
        self.soc_next = 0.6  # state of charge

        return {
            "x": self.x_next,
            "y": self.y_next,
            "x_dot": self.x_dot_next,
            "y_dot": self.y_dot_next,
            "phi": self.phi_next,
            "omega": self.omega_next,
            "soc": self.soc_next,
            "force": self.force,
        }

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # Limit Action 这里放入强化时去掉，变成软约束即可
        if self.x_dot > 50:
            action[0] = min(action[0], 0)
        elif self.x_dot < 0:
            action[0] = max(action[0], 0)
        elif self.phi>math.radians(5):
            action[1] = min(action[1], 0)
        elif self.phi<math.radians(-5):
            action[1] = max(action[1], 0)
        # Syetem Dynamics
        self.x_next = self.x + self.delta_t * (self.x_dot * abs(math.cos(self.phi)) - self.y_dot * abs(math.sin(self.phi)))
        self.y_next = self.y + self.delta_t * (self.x_dot * abs(math.sin(self.phi))+ self.y_dot * abs(math.cos(self.phi)))
        self.x_dot_next = self.x_dot + self.delta_t * (action[0] + self.y_dot * self.omega)

        self.y_dot_next = (
            self.m * self.x_dot * self.y_dot
            + self.K * self.omega * self.delta_t
            - self.m * self.x_dot**2 * self.omega * self.delta_t
            - self.c_f * self.x_dot * action[1] * self.delta_t
        ) / (self.m * self.x_dot - self.D * self.delta_t)
        self.phi_next = self.phi + self.delta_t * self.omega
        self.omega_next = (
            self.I_zz * self.x_dot * self.omega
            + self.K * self.y_dot * self.delta_t
            - self.x_dot * self.a_v * self.c_f * action[1] * self.delta_t
        ) / (self.I_zz * self.x_dot - self.W * self.delta_t)
        self.force = (
            action[0] * self.m
            + self.m * self.g * self.tau_r * math.cos(self.theta)
            + self.m * self.g * math.sin(self.theta)
            + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
        )
        pb = pb_cal(
            self.motor_eff_2d,
            self.force,
            self.x_dot,
            self.soc,
            self.r_w,
            self.battery_eff_dis_1d,
            self.battery_eff_cha_1d,
        )
        self.soc_next = bat_dynamic(
            self.soc,
            self.force * self.r_w,
            self.x_dot * 60 / (2 * math.pi * self.r_w),
            self.battery_eff_dis_1d,
            self.battery_eff_cha_1d,
            self.delta_t,
            self.bat_q,
        )[0]

        # update state and relate info
        self.x = self.x_next
        self.y = self.y_next
        self.x_dot = self.x_dot_next
        self.y_dot = self.y_dot_next
        self.phi = self.phi_next
        self.omega = self.omega_next
        self.soc = self.soc_next

        # Return State, Reward, Done, Info
        return_state = {
            "x": self.x_next,
            "y": self.y_next,
            "x_dot": self.x_dot_next,
            "y_dot": self.y_dot_next,
            "phi": self.phi_next,
            "omega": self.omega_next,
            "soc": self.soc_next,
            "force": self.force,
        }

        reward = pb * self.delta_t
        done = False
        info = {}
        return return_state, reward, done, info


if __name__ == "__main__":
    env = VehicleEnv()

    obs = env.reset()
    done = False

    obs_lists = []
    reward_lists = []
    for i in range(10000):
        obs_lists.append(obs)
        if i <= 10:
            action = [1, math.pi / 4]
        # elif 400<=i < 500:
        #     action = [0, 1]
        else:
            action = [-0.001, 0]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        reward_lists.append(reward)
    Reward = sum(reward_lists)
    x_plot = [obs["x"] for obs in obs_lists]
    y_plot = [obs["y"] for obs in obs_lists]
    x_dot_plot = [obs["x_dot"] for obs in obs_lists]
    y_dot_plot = [obs["y_dot"] for obs in obs_lists]
    phi_plot = [obs["phi"] for obs in obs_lists]
    omega_plot = [obs["omega"] for obs in obs_lists]

    from utils import plot_list

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.plot(x_plot)
    plt.title("x_plot")
    plt.subplot(2, 3, 2)
    plt.plot(y_plot)
    plt.title("y_plot")
    plt.subplot(2, 3, 3)
    plt.plot(x_dot_plot)
    plt.title("x_dot_plot")
    plt.subplot(2, 3, 4)
    plt.plot(y_dot_plot)
    plt.title("y_dot_plot")
    plt.subplot(2, 3, 5)
    plt.plot(phi_plot)
    plt.title("phi_plot")
    plt.subplot(2, 3, 6)
    plt.plot(omega_plot)
    plt.title("omega_plot")
    plt.show()
    print("")
    plt.figure(2)
    plt.plot(x_plot, y_plot)
    plt.title("x_plot,y_plot")
    plt.show()
