# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Polygon


from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build
from utils import coordination, direction_distance


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None
        with open("sv_dic_all.json", "r") as file:
            self.surrounding_vehicles_all = json.load(file)
        self.surrounding_vehicles_initial_time = 3500
        self.time_step = None
        # parameters for road
        self.road_width = 23
        self.road_init_width = 10
        self.road_length = 100
        self.road_num = 1
        self.road_gradient_fun = road_curvature_gradient_build(self.road_length, self.road_width, self.road_num,self.road_init_width)

        # parameters for ego vehicle
        self.vehicle = VehicleEnv(
            road_width=self.road_width,
            road_length=self.road_length,
            road_num=self.road_num,
            road_init_width=self.road_init_width,
            road_gradient_func = self.road_gradient_fun,
        )
        self.vehicle_obs = None  # vehicle observation
        self.ego_x_initial = 0
        self.ego_y_initial = self.road_init_width + self.road_width / 2
        self.ego_x_dot_initial = 15
        self.ego_y_dot_initial = 0
        self.ego_phi_initial = 0
        self.ego_omega_initial = 0
        self.ego_soc_initial = 0.7


        self.dmin = None
        self.dmin_next = None
        self.max_distant = math.sqrt(self.road_length**2 + (self.road_num * self.road_width+self.road_init_width) ** 2)
        self.max_distances = [self.road_length,self.road_num * self.road_width+self.road_init_width]


    def reset(self):
        self.surrounding_vehicles = self.surrounding_vehicles_all['%d'%self.surrounding_vehicles_initial_time]  # surrounding vehicles
        sv_time = self.surrounding_vehicles_initial_time
        self.time_step = self.surrounding_vehicles_initial_time
        self.road_curvature = 0  # 曲率
        self.vehicle_obs = self.vehicle.reset(
            self.ego_x_initial,
            self.ego_y_initial,
            self.ego_x_dot_initial,
            self.ego_y_dot_initial,
            self.ego_phi_initial,
            self.ego_omega_initial,
            self.ego_soc_initial,
        )
        self.dmin_next,_ = direction_distance(
            [
                self.vehicle_obs["x_next"],
                self.vehicle_obs["y_next"],
                self.vehicle_obs["phi_next"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
            self.max_distances,
        )
        self.dmin = self.dmin_next
        self.vehicle_obs["xy_direction"] = self.dmin
        self.vehicle_obs["xy_direction_next"] = self.dmin_next
        return self.vehicle_obs

    def step(self, action):
        assert isinstance(action, list), "action must be a list"
        self.time_step = self.time_step + 1
        next_vehicle_obs, reward_ego, done_ego, info_ego = self.vehicle.step(action)
        self.surrounding_vehicles = self.surrounding_vehicles_all['%d'%self.time_step]  # surrounding vehicles
        sv_time = self.time_step

        self.dmin_next,flag_collision = direction_distance(
            [
                next_vehicle_obs["x_next"],
                next_vehicle_obs["y_next"],
                next_vehicle_obs["phi_next"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
            self.max_distances,
        )
        # print(self.dmin_next)
        next_vehicle_obs["xy_direction"] = self.dmin

        next_vehicle_obs["xy_direction_next"] = self.dmin_next
        self.dmin = self.dmin_next
        self.vehicle_obs = next_vehicle_obs
        if flag_collision <= 0:
            done_road = 1
        else:
            done_road = 0
        reward = reward_ego
        if done_ego or done_road:
            done = 1
            info_ego['collision'] = done_road
            info = info_ego
        else:
            done = 0
            info = {}
        return next_vehicle_obs, reward, done, info

    def render(self):
        self.plot_road()

    def plot_road(self):
        plt.cla()
        lateralPos = self.surrounding_vehicles["Local_X"]
        longitudePos = self.surrounding_vehicles["Local_Y"]
        id = self.surrounding_vehicles["Vehicle_ID"]
        length = self.surrounding_vehicles["v_Length"]
        width = self.surrounding_vehicles["v_Width"]
        v_class = self.surrounding_vehicles["v_Class"]
        # 绘制左右两边的黑色实线
        # view_road_len = 30
        plt.plot(
            [0, self.road_length],
            [3.0356616-2.5908/2, 3.0356616-2.5908/2],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [0, self.road_length],
            [31.5472632+2.5908/2, 31.5472632+2.5908/2],
            color="black",
            linewidth=2,
        )


        # 小车参数
        ego_parmeters = [
            self.vehicle_obs["x"],
            self.vehicle_obs["y"],
            self.vehicle_obs["phi"],
            self.vehicle.car_length,
            self.vehicle.car_width,
        ]
        car_box = coordination(ego_parmeters)
        rectangle_ego = Polygon(car_box, closed=True, edgecolor="red", linewidth=2, facecolor="none")

       

        for i in range(len(id)):
            surrounding_vehicle_parmeters = [
                longitudePos[i],
                lateralPos[i],
                0,
                length[i],
                width[i],
            ]
            surrounding_vehicle_box = coordination(surrounding_vehicle_parmeters)

            # 创建Polygon对象
            if v_class[i] == 1:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="red",
                    linewidth=2,
                    facecolor="none",
                )
            elif v_class[i] == 2:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="green",
                    linewidth=2,
                    facecolor="none",
                )
            elif v_class[i] == 3:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="blue",
                    linewidth=2,
                    facecolor="none",
                )

            # 将小车形状添加到图形中
            plt.gca().add_patch(rectangle)
            plt.text(longitudePos[i], lateralPos[i], str(id[i]), fontsize=10, color='red', style='italic')

        # 将小车形状添加到图形中
        plt.gca().add_patch(rectangle_ego)
        plt.pause(self.vehicle.delta_t)




if __name__ == "__main__":
    road_env = RoadEnv()
    obs = road_env.reset()
    done = False
    obs_lists = []
    reward_lists = []
    while not done:
        # print(obs['dmin'])
        obs_lists.append(obs)
        action = [1, 0]
        next_obs, reward, done, info = road_env.step(action)
        obs = next_obs
        reward_lists.append(reward)

        road_env.plot_road()

    print("reward:", sum(reward_lists))
    print("obs:", obs_lists[-1])
    print("done:", done)
    print("info:", info)

