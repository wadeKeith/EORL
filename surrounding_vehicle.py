import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from utils import coordination


def find_max_num_sv(surrounding_vehicles):
    frames = np.unique(surrounding_vehicles["Frame_ID"])
    max_num = 0
    for frame in frames:
        num = surrounding_vehicles[surrounding_vehicles["Frame_ID"] == frame].shape[0]
        if num > max_num:
            max_num = num
    return max_num


class SV_env(object):
    def __init__(self) -> None:
        self.surrounding_vehicle_all = pd.read_csv("New_NGSIM.csv")
        self.surrounding_vehicle_total_time = np.unique(self.surrounding_vehicle_all["Frame_ID"]).shape[0]
        self.surrounding_vehicle_max_num = find_max_num_sv(self.surrounding_vehicle_all)
        self.frames = np.unique(self.surrounding_vehicle_all["Frame_ID"])
        self.inital_time = 3000
        self.surrounding_vehicles_lat_min = min(self.surrounding_vehicle_all["Local_X"])
        self.surrounding_vehicles_lat_max = max(self.surrounding_vehicle_all["Local_X"])
        self.surrounding_vehicles_lon_min = 0
        self.surrounding_vehicles_lon_max = 300
        self.surrounding_vehicle_width_max = max(self.surrounding_vehicle_all["v_Width"])
        # parameters for road
        self.road_width = 23
        self.road_num = 1
        self.time_step = self.inital_time
        self.surrounding_vehicles = None
        self.deltat = 0.01


    def reset(self):
        self.time_step = self.inital_time
        self.surrounding_vehicles = self.surrounding_vehicle_all[(self.surrounding_vehicle_all["Frame_ID"]==self.inital_time) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]>=self.surrounding_vehicles_lon_min) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]<=self.surrounding_vehicles_lon_max)]  # surrounding vehicles



        return self.surrounding_vehicles

    def step(self):
        self.time_step = self.time_step + 1
        next_surrounding_vehicles = self.surrounding_vehicle_all[(self.surrounding_vehicle_all["Frame_ID"]==self.time_step) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]>=self.surrounding_vehicles_lon_min) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]<=self.surrounding_vehicles_lon_max)]
        self.surrounding_vehicles = next_surrounding_vehicles
        return self.surrounding_vehicles

    def render(self):
        self.plot_road()

    def plot_road(self):
        plt.cla()
        lateralPos = self.surrounding_vehicles["Local_X"]
        longitudePos = self.surrounding_vehicles["Local_Y"]
        id = self.surrounding_vehicles["Vehicle_ID"]
        len = self.surrounding_vehicles["v_Length"]
        width = self.surrounding_vehicles["v_Width"]
        v_class = self.surrounding_vehicles["v_Class"]
        # 绘制左右两边的黑色实线
        # view_road_len = 30
        plt.plot(
            [self.surrounding_vehicles_lon_min, self.surrounding_vehicles_lon_max],
            [self.surrounding_vehicles_lat_min-self.surrounding_vehicle_width_max/2, self.surrounding_vehicles_lat_min-self.surrounding_vehicle_width_max/2],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [self.surrounding_vehicles_lon_min, self.surrounding_vehicles_lon_max],
            [self.surrounding_vehicles_lat_max+self.surrounding_vehicle_width_max/2, self.surrounding_vehicles_lat_max+self.surrounding_vehicle_width_max/2],
            color="black",
            linewidth=2,
        )


        # 小车参数

       

        for i in range(0,self.surrounding_vehicles.shape[0]):
            surrounding_vehicle_parmeters = [
                longitudePos.values[i],
                lateralPos.values[i],
                0,
                len.values[i],
                width.values[i],
            ]
            surrounding_vehicle_box = coordination(surrounding_vehicle_parmeters)

            # 创建Polygon对象
            if v_class.values[i] == 1:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="red",
                    linewidth=2,
                    facecolor="none",
                )
            elif v_class.values[i] == 2:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="green",
                    linewidth=2,
                    facecolor="none",
                )
            elif v_class.values[i] == 3:
                rectangle = Polygon(
                    surrounding_vehicle_box,
                    closed=True,
                    edgecolor="blue",
                    linewidth=2,
                    facecolor="none",
                )

            # 将小车形状添加到图形中
            plt.gca().add_patch(rectangle)
            plt.text(longitudePos.values[i], lateralPos.values[i], str(id.values[i]), fontsize=10, color='red', style='italic')

        # 将小车形状添加到图形中
        plt.pause(self.deltat)


if __name__ == "__main__":
    sv_env = SV_env()
    obs = sv_env.reset()
    for i in range(1, 200):
        obs_new = sv_env.step()
        obs = obs_new
        sv_env.render()

    obs = sv_env.reset()
    for i in range(1, 200):
        obs_new = sv_env.step()
        obs = obs_new
        sv_env.render()

    print('a')

