import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build
from utils import coordination, e_s_distance
from surrounding_vehicle import SV_env


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None
        self.surrounding_vehicles_all = SV_env()
        # parameters for road
        self.road_width = 23
        self.road_init_width = 10
        self.road_length = 300
        self.road_num = 1
        self.road_gradient_fun = road_curvature_gradient_build(self.road_length, self.road_width, self.road_num,self.road_init_width)

        # parameters for ego vehicle
        self.vehicle = VehicleEnv(
            road_width=self.road_width,
            road_length=self.road_length,
            road_num=self.road_num,
            road_init_width=self.road_init_width,
        )
        self.vehicle_obs = None  # vehicle observation
        self.ego_x_initial = 0
        self.ego_y_initial = self.road_init_width + self.road_width / 2
        self.ego_x_dot_initial = 20
        self.ego_y_dot_initial = 0
        self.ego_phi_initial = 0
        self.ego_omega_initial = 0
        self.ego_soc_initial = 0.7


        self.dmin = None
        self.x_min = None
        self.y_min = None
        self.dmin_next = None
        self.x_min_next = None
        self.y_min_next = None
        self.max_distant = math.sqrt(self.road_length**2 + (self.road_num * self.road_width+self.road_init_width) ** 2)


    def reset(self):
        self.surrounding_vehicles = self.surrounding_vehicles_all.reset()  # surrounding vehicles
        self.road_curvature = 0  # 曲率
        self.road_gradient = self.road_gradient_fun([self.ego_x_initial, self.ego_y_initial])[0]
        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        self.vehicle_obs = self.vehicle.reset(
            self.ego_x_initial,
            self.ego_y_initial,
            self.ego_x_dot_initial,
            self.ego_y_dot_initial,
            self.ego_phi_initial,
            self.ego_omega_initial,
            self.ego_soc_initial,
        )
        self.dmin_next, self.x_min_next, self.y_min_next = e_s_distance(
            [
                self.vehicle_obs["x_next"],
                self.vehicle_obs["y_next"],
                self.vehicle_obs["phi_next"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
        )
        self.dmin = self.dmin_next
        self.x_min = self.x_min_next
        self.y_min = self.y_min_next
        self.vehicle_obs["dmin"] = self.dmin
        self.vehicle_obs["x_min"] = self.x_min
        self.vehicle_obs["y_min"] = self.y_min
        self.vehicle_obs["dmin_next"] = self.dmin_next
        self.vehicle_obs["x_min_next"] = self.x_min_next
        self.vehicle_obs["y_min_next"] = self.y_min_next
        return self.vehicle_obs

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # update road gradient
        try:
            self.road_gradient = self.road_gradient_fun([self.vehicle_obs["x_next"], self.vehicle_obs["y_next"]])[0]
        except:
            print("out of road and x:", self.vehicle_obs["x_next"], "y:", self.vehicle_obs["y_next"])

        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        next_vehicle_obs, reward_ego, done_ego, info_ego = self.vehicle.step(action)
        next_surrounding_vehicles = self.surrounding_vehicles_all.step()  # surrounding vehicles
        self.surrounding_vehicles = next_surrounding_vehicles

        self.dmin_next, self.x_min_next, self.y_min_next = e_s_distance(
            [
                next_vehicle_obs["x_next"],
                next_vehicle_obs["y_next"],
                next_vehicle_obs["phi_next"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
        )
        next_vehicle_obs["dmin"] = self.dmin
        next_vehicle_obs["x_min"] = self.x_min
        next_vehicle_obs["y_min"] = self.y_min
        next_vehicle_obs["dmin_next"] = self.dmin_next
        next_vehicle_obs["x_min_next"] = self.x_min_next
        next_vehicle_obs["y_min_next"] = self.y_min_next
        self.dmin = self.dmin_next
        self.x_min = self.x_min_next
        self.y_min = self.y_min_next
        self.vehicle_obs = next_vehicle_obs
        if min(self.dmin_next) <= 0:
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
        len = self.surrounding_vehicles["v_Length"]
        width = self.surrounding_vehicles["v_Width"]
        v_class = self.surrounding_vehicles["v_Class"]
        # 绘制左右两边的黑色实线
        # view_road_len = 30
        plt.plot(
            [self.surrounding_vehicles_all.surrounding_vehicles_lon_min, self.surrounding_vehicles_all.surrounding_vehicles_lon_max],
            [self.surrounding_vehicles_all.surrounding_vehicles_lat_min-self.surrounding_vehicles_all.surrounding_vehicle_width_max/2, self.surrounding_vehicles_all.surrounding_vehicles_lat_min-self.surrounding_vehicles_all.surrounding_vehicle_width_max/2],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [self.surrounding_vehicles_all.surrounding_vehicles_lon_min, self.surrounding_vehicles_all.surrounding_vehicles_lon_max],
            [self.surrounding_vehicles_all.surrounding_vehicles_lat_max+self.surrounding_vehicles_all.surrounding_vehicle_width_max/2, self.surrounding_vehicles_all.surrounding_vehicles_lat_max+self.surrounding_vehicles_all.surrounding_vehicle_width_max/2],
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
        plt.gca().add_patch(rectangle_ego)
        plt.pause(self.vehicle.delta_t)




if __name__ == "__main__":
    road_env = RoadEnv()
    obs = road_env.reset()
    done = False
    obs_lists = []
    reward_lists = []
    while not done:
        obs_lists.append(obs)
        action = [2, 0]
        next_obs, reward, done, info = road_env.step(action)
        obs = next_obs
        reward_lists.append(reward)

        road_env.plot_road()

    print("reward:", sum(reward_lists))
    print("obs:", obs_lists[-1])
    print("done:", done)
    print("info:", info)
