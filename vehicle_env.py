import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from vehicle_utils import bat_dynamic, pb_cal


class VehicleEnv(object):
    def __init__(self,road_width,road_length,road_num,car_length):
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

        self.theta = None

        self.road_width = road_width
        self.road_num = road_num
        self.road_length = road_length
        self.car_length = car_length
        self.car_width = self.road_width / 3 * 2

    def update_theta(self, theta):
        self.theta = theta

    def reset(self,x,y,x_dot,y_dot,phi,omega,soc):
        self.x = x
        self.y = y
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.phi = phi  # 角度
        self.omega = omega  # 角速度
        self.soc = soc  # state of charge
        self.force = (
            0 * self.m
            + self.m * self.g * self.tau_r * math.cos(self.theta)
            + self.m * self.g * math.sin(self.theta)
            + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
        )

        # # next time step
        # self.x_next = 0
        # self.y_next = 0
        # self.x_dot_next = 0
        # self.y_dot_next = 0
        # self.phi_next = 0  # 角度
        # self.omega_next = 0  # 角速度
        # self.soc_next = 0.6  # state of charge
        if 0 < self.x < 1000 and 0 < self.y < self.road_width * self.road_num:
            done = False
        else:
            done = True
        return {
            "x": self.x,
            "y": self.y,
            "x_dot": self.x_dot,
            "y_dot": self.y_dot,
            "phi": self.phi,
            "omega": self.omega,
            "soc": self.soc,
            "force": self.force,
        },done

    def step(self, action):
        assert isinstance(action, list), "action must be a list"
        # Syetem Dynamics
        self.x_next = self.x + self.delta_t * (self.x_dot * math.cos(self.phi) - self.y_dot * math.sin(self.phi))
        self.y_next = self.y + self.delta_t * (self.x_dot * math.sin(self.phi)+ self.y_dot * math.cos(self.phi))
        if 0<self.x_next<1000 and 0<self.y_next<self.road_width*self.road_num:
            done_outofroad = False
        else:
            done_outofroad = True
        x_dot_next = self.x_dot + self.delta_t * (action[0] + self.y_dot * self.omega)
        if 0<=x_dot_next <= 50 :
            self.x_dot_next = self.x_dot + self.delta_t * (action[0] + self.y_dot * self.omega)
            done_overacceration = False
        elif x_dot_next > 50:
            self.x_dot_next = 50
            done_overacceration = True
        else:
            self.x_dot_next = 0
            done_overacceration = True
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
        if self.min_torque/self.r_w <= force <=self.max_torque/self.r_w:
            self.force = (
                action[0] * self.m
                + self.m * self.g * self.tau_r * math.cos(self.theta)
                + self.m * self.g * math.sin(self.theta)
                + 0.5 * self.rho_a * self.A_f * self.tau_a * self.x_dot**2
            )
            done_motor_cant_provide = False
        elif force < self.min_torque/self.r_w:
            self.force = self.min_torque/self.r_w
            done_motor_cant_provide = False
        else:
            self.force = self.max_torque/self.r_w
            done_motor_cant_provide = True
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
            print('Motor cant offer the acceration or velocity or SOC ','Force:',self.force,'velocity:',self.x_dot,'SOC:',self.soc)

        self.soc_next = bat_dynamic(
            self.motor_eff_2d,
            self.r_w,
            self.soc,
            self.force,
            self.x_dot ,
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
        if done_outofroad or done_overacceration or done_motor_cant_provide:
            reward = -1
            done = True
            info = {}
        else:
            reward = -1*pb/(self.max_torque/self.r_w*50)+math.exp(-abs(self.soc-0.6))+2*math.exp(-abs(self.x-1000+self.y-self.road_width*self.road_num/2))
            done = False
            info = {}
        return return_state, reward, done, info


# if __name__ == "__main__":

