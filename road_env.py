import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build
from utils import coordination, e_s_distance, dynamic_surrounding_vehicle


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None

        # parameters for road
        self.road_width = 3.75
        self.road_length = 1000
        self.road_num = 3
        self.road_gradient_fun = road_curvature_gradient_build(
            self.road_length, self.road_width, self.road_num
        )

        # parameters for ego vehicle
        self.vehicle = VehicleEnv(
            road_width=self.road_width,
            road_length=self.road_length,
            road_num=self.road_num,
        )
        self.vehicle_obs = None  # vehicle observation
        self.ego_x_initial = 5
        self.ego_y_initial = self.road_width / 2 * 3
        self.ego_x_dot_initial = 20
        self.ego_y_dot_initial = 0
        self.ego_phi_initial = 0
        self.ego_omega_initial = 0
        self.ego_soc_initial = 0.7
        self.surrounding_velocity = 10
        self.surrounding_vehicles = {
            "1": {
                "x": self.road_length / 5,
                "y": self.road_width / 2,
                "phi": 0,
                "v": self.surrounding_velocity,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "2": {
                "x": self.road_length / 5 * 3,
                "y": self.road_width / 2,
                "phi": 0,
                "v": self.surrounding_velocity,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "3": {
                "x": self.road_length / 5 * 3,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": self.surrounding_velocity,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "4": {
                "x": self.road_length / 5 * 2,
                "y": self.road_width / 2 * 3,
                "phi": 0,
                "v": self.surrounding_velocity,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "5": {
                "x": self.road_length / 5 ,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": self.surrounding_velocity,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
        }  # surrounding vehicles
        self.dmin = None
        self.dmin_next = None
        self.max_distant = math.sqrt(
            self.road_length**2 + (self.road_num * self.road_width) ** 2
        )

    def reset(self):
        self.road_curvature = 0  # 曲率
        self.road_gradient = self.road_gradient_fun(
            [self.ego_x_initial, self.ego_y_initial]
        )[0]
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
        self.dmin = e_s_distance(
            [
                self.vehicle_obs["x"],
                self.vehicle_obs["y"],
                self.vehicle_obs["phi"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
        )
        self.vehicle_obs["dmin"] = self.dmin

        return self.vehicle_obs

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # update road gradient
        try:
            self.road_gradient = self.road_gradient_fun(
                [self.vehicle_obs["x"], self.vehicle_obs["y"]]
            )[0]
        except:
            print(
                "out of road and x:", self.vehicle_obs["x"], "y:", self.vehicle_obs["y"]
            )

        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        next_vehicle_obs, reward_ego, done_ego, info = self.vehicle.step(action)
        next_surrounding_vehicles = dynamic_surrounding_vehicle(
            self.surrounding_vehicles, self.vehicle.delta_t
        )
        self.surrounding_vehicles = next_surrounding_vehicles

        self.dmin_next = e_s_distance(
            [
                next_vehicle_obs["x"],
                next_vehicle_obs["y"],
                next_vehicle_obs["phi"],
                self.vehicle.car_length,
                self.vehicle.car_width,
            ],
            self.surrounding_vehicles,
        )
        next_vehicle_obs["dmin"] = self.dmin_next
        if min(self.dmin_next) <= 0:
            done_road = 0
            # reward_collision = min(self.dmin_next)*1000
            reward_collision = 0
            info = {}
        else:
            done_road = 0
            reward_collision = min(self.dmin_next)/10
            info = {}
        self.dmin = self.dmin_next
        self.vehicle_obs = next_vehicle_obs
        reward = (reward_ego + reward_collision)/100
        if done_ego or done_road:
            done = 1
        else:
            done = 0
        return next_vehicle_obs, reward, done, info

    def render(self):
        self.plot_road()

    def plot_road(self):
        plt.cla()
        x_ls = []
        for key, value in self.surrounding_vehicles.items():
            x_ls.append(value["x"])
        x_ls.append(self.vehicle_obs["x"])
        x_min = min(x_ls)
        x_max = max(x_ls)
        # 绘制左右两边的黑色实线
        # view_road_len = 30
        plt.plot(
            [x_min - 5, x_max + 5],
            [0, 0],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [x_min - 5, x_max + 5],
            [self.road_num * self.road_width, self.road_num * self.road_width],
            color="black",
            linewidth=2,
        )

        # 绘制车道线
        for i in range(1, self.road_num):
            lane_y = i * self.road_width
            plt.plot(
                [x_min - 5, x_max + 5],
                [lane_y, lane_y],
                linestyle="dashed",
                color="black",
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
        rectangle_ego = Polygon(
            car_box, closed=True, edgecolor="red", linewidth=2, facecolor="none"
        )

        for key, value in self.surrounding_vehicles.items():
            surrounding_vehicle_parmeters = [
                value["x"],
                value["y"],
                value["phi"],
                value["car_length"],
                value["car_width"],
            ]
            surrounding_vehicle_box = coordination(surrounding_vehicle_parmeters)

            # 创建Polygon对象
            rectangle = Polygon(
                surrounding_vehicle_box,
                closed=True,
                edgecolor="blue",
                linewidth=2,
                facecolor="none",
            )

            # 将小车形状添加到图形中
            plt.gca().add_patch(rectangle)

        # 将小车形状添加到图形中
        plt.gca().add_patch(rectangle_ego)
        plt.pause(self.vehicle.delta_t)


if __name__ == "__main__":
    road_env = RoadEnv()
    obs = road_env.reset()
    done = False
    obs_lists = []
    reward_lists = []
    for i in range(10000):
        obs_lists.append(obs)
        action = [4, 0]
        next_obs, reward, done, info = road_env.step(action)
        obs = next_obs
        reward_lists.append(reward)

        road_env.plot_road()
        # print(obs['force'])
    print("reward:", sum(reward_lists))
