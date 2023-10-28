# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from vehicle_utils import bat_dynamic, pb_cal


class VehicleEnv(object):
    def __init__(self, road_width, road_length, road_num,road_init_width,road_gradient_func):
        self.x = None
        self.y = None
        self.x_dot = None
        self.y_dot = None
        self.phi = None  # 角度
        self.omega = None  # 角速度
        self.force = None  # driving force
        self.soc = None  # state of charge
        self.theta = None
        
        # next time step
        self.x_next = None
        self.y_next = None
        self.x_dot_next = None
        self.y_dot_next = None
        self.phi_next = None  # 下一时刻角度
        self.omega_next = None  # 下一时刻角速度
        self.soc_next = None  # 下一时刻电池电量
        self.theta_next = None

        # self.x_ddot = None
        self.road_gradient_func = road_gradient_func
        self.delta_t = 0.1  # simulate time step

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
        self.min_torque = -1200
        self.max_torque = 1200
        self.motor_eff_speed = np.array([0, 1000, 2000, 3000, 4000])
        self.motor_eff_torque = np.linspace(self.min_torque, self.max_torque, 21)
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



        self.road_width = road_width
        self.road_init_width = road_init_width
        self.road_num = road_num
        self.road_length = road_length
        self.car_length = self.a_v + self.b_v
        self.car_width = 1.95072


    def reset(self, x, y, x_dot, y_dot, phi, omega, soc):
        self.x = x
        self.y = y
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.phi = phi  # 角度
        self.omega = omega  # 角速度
        self.soc = soc  # state of charge
        self.theta = math.radians(self.road_gradient_func([self.x, self.y])[0])

        # next time step
        self.x_next = x
        self.y_next = y
        self.x_dot_next = x_dot
        self.y_dot_next = y_dot
        self.phi_next = phi  # 角度
        self.omega_next = omega  # 角速度
        self.soc_next = soc  # state of charge
        self.theta_next = self.theta
        return {
            "x": self.x,
            "y": self.y,
            "x_dot": self.x_dot,
            "y_dot": self.y_dot,
            "phi": self.phi,
            "omega": self.omega,
            "soc": self.soc,
            "theta": self.theta,
            "x_next": self.x_next,
            "y_next": self.y_next,
            "x_dot_next": self.x_dot_next,
            "y_dot_next": self.y_dot_next,
            "phi_next": self.phi_next,
            "omega_next": self.omega_next,
            "soc_next": self.soc_next,
            "theta_next": self.theta_next,
        }

    def step(self, action):
        assert isinstance(action, list), "action must be a list"
        # Syetem Dynamics
        self.x_next = self.x + self.delta_t * (self.x_dot * math.cos(self.phi) - self.y_dot * math.sin(self.phi))
        self.y_next = self.y + self.delta_t * (self.x_dot * math.sin(self.phi) + self.y_dot * math.cos(self.phi))
        # 开出道路惩罚
        done_outofroad = (
            0 if self.road_init_width < self.y_next < self.road_init_width+self.road_width * self.road_num else 1
        )  # 0表示没有开出道路，1表示开出道路
        x_dot_next = self.x_dot + self.delta_t * (action[0] + self.y_dot * self.omega)
        # 速度约束
        if 0 < x_dot_next < 50:
            done_overacceration = 0
        elif x_dot_next >= 50:
            action[0] = 0
            done_overacceration = 1
        else:
            action[0] = 0
            done_overacceration = 1
        self.x_dot_next = self.x_dot + self.delta_t * (action[0] + self.y_dot * self.omega)
        # 到终点的奖励
        if self.x_next >= self.road_length:
            done_arrive = 1
        else:
            done_arrive = 0
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
        force = (
            action[0] * self.m
            + self.m * self.g * self.tau_r * math.cos(self.theta)
            + self.m * self.g * math.sin(self.theta)
            + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
        )
        if self.min_torque / self.r_w <= force <= self.max_torque / self.r_w:
            self.force = (
                action[0] * self.m
                + self.m * self.g * self.tau_r * math.cos(self.theta)
                + self.m * self.g * math.sin(self.theta)
                + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
            )
            done_motor_cant_provide = 0
        elif force < self.min_torque / self.r_w:
            self.force = self.min_torque / self.r_w
            done_motor_cant_provide = 0
        else:
            self.force = self.max_torque / self.r_w
            done_motor_cant_provide = 1
        try:
            pb = pb_cal(
                self.motor_eff_2d,
                self.force,
                self.x_dot,
                self.soc,
                self.r_w,
                self.battery_eff_dis_1d,
                self.battery_eff_cha_1d,
            )[0]
        except:
            print(
                "Motor cant offer the acceration or velocity or SOC ",
                "Force:",
                self.force,
                "velocity:",
                self.x_dot,
                "SOC:",
                self.soc,
            )

        self.soc_next = bat_dynamic(
            self.motor_eff_2d,
            self.r_w,
            self.soc,
            self.force,
            self.x_dot,
            self.battery_eff_dis_1d,
            self.battery_eff_cha_1d,
            self.delta_t,
            self.bat_q,
        )[0]
        if done_outofroad or done_arrive:
            self.theta_next = math.radians(self.road_gradient_func([self.x, self.y])[0])
        else:
            self.theta_next = math.radians(self.road_gradient_func([self.x_next, self.y_next])[0])
        # Return State, Reward, Done, Info
        return_state = {
            "x": self.x,
            "y": self.y,
            "x_dot": self.x_dot,
            "y_dot": self.y_dot,
            "phi": self.phi,
            "omega": self.omega,
            "soc": self.soc,
            "theta": self.theta,
            "x_next": self.x_next,
            "y_next": self.y_next,
            "x_dot_next": self.x_dot_next,
            "y_dot_next": self.y_dot_next,
            "phi_next": self.phi_next,
            "omega_next": self.omega_next,
            "soc_next": self.soc_next,
            "theta_next": self.theta_next,
        }
        # update state and relate info

        self.x = self.x_next
        self.y = self.y_next
        self.x_dot = self.x_dot_next
        self.y_dot = self.y_dot_next
        self.phi = self.phi_next
        self.omega = self.omega_next
        self.soc = self.soc_next
        self.theta = self.theta_next

        if done_outofroad or done_overacceration or done_motor_cant_provide or done_arrive:
            done = 1
            info = {'out of road':done_outofroad,
                    'speed illegal':done_overacceration,
                    'motor cant provide':done_motor_cant_provide,
                    'arrive':done_arrive}
        else:
            done = 0
            info = {}
        # speed_reward = np.sin((2*np.pi/((20-0)*2))*(self.x_dot_next)) if 0<self.x_dot_next<20 else -1
        speed_reward = 0
        if done_arrive:
            reward_arrive = 100    # 到达终点奖励
        else:
            reward_arrive = 0
        reward = (
            speed_reward+reward_arrive+math.sin((math.pi/2/self.road_length)*self.x_next)
            +5*abs(np.sin((2*np.pi/((self.road_width)/3))*(self.y_next-(self.road_init_width))))
            - 1*pb*self.delta_t / (self.bat_q/3600)
            - 0.1 * abs(action[0])
            - 0.1 * abs(action[1])
        )
        
        return return_state, reward, done, info


# if __name__ == "__main__":
