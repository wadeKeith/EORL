import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build
from utils import coordination, e_s_distance


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None

        # parameters for road
        self.road_width = 3.75
        self.road_length = 1000
        self.road_num = 3
        self.road_gradient_fun = road_curvature_gradient_build(self.road_length, self.road_width, self.road_num)

        # parameters for ego vehicle
        self.vehicle = VehicleEnv(
            road_width=self.road_width, road_length=self.road_length, road_num=self.road_num, car_length=5
        )
        self.vehicle_obs = None  # vehicle observation
        self.ego_x_initial = 5
        self.ego_y_initial = self.road_width / 2 * 3
        self.ego_x_dot_initial = 20
        self.ego_y_dot_initial = 0
        self.ego_phi_initial = 0
        self.ego_omega_initial = 0
        self.ego_soc_initial = 0.7

        self.surrounding_vehicles = {
            "1": {
                "x": self.road_length / 5,
                "y": self.road_width / 2,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "2": {
                "x": self.road_length / 5 * 3,
                "y": self.road_width / 2,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "3": {
                "x": self.road_length / 5 * 5,
                "y": self.road_width / 2,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "4": {
                "x": self.road_length / 5 * 2,
                "y": self.road_width / 2 * 3,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "5": {
                "x": self.road_length / 5 * 4,
                "y": self.road_width / 2 * 3,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "6": {
                "x": self.road_length / 4,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "7": {
                "x": self.road_length / 4 * 2,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "8": {
                "x": self.road_length / 4 * 3,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
            "9": {
                "x": self.road_length / 12 * 11,
                "y": self.road_width / 2 * 5,
                "phi": 0,
                "v": 0,
                "car_length": 5,
                "car_width": self.road_width / 3 * 2,
            },
        }  # surrounding vehicles
        self.dmin = None
        self.dmin_next = None
        self.max_distant = math.sqrt(self.road_length**2 + (self.road_num * self.road_width) ** 2)

    def reset(self):
        self.road_curvature = 0  # 曲率
        self.road_gradient = self.road_gradient_fun([self.ego_x_initial, self.ego_y_initial])[0]
        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        self.vehicle_obs, done_ego = self.vehicle.reset(
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
        if min(self.dmin) <= 0:
            done_road = True
        else:
            done_road = False
        if done_ego or done_road:
            done = True
        else:
            done = False

        return self.vehicle_obs, done

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # update road gradient
        try:
            self.road_gradient = self.road_gradient_fun([self.vehicle_obs["x"], self.vehicle_obs["y"]])[0]
        except:
            print("out of road and x:", self.vehicle_obs["x"], "y:", self.vehicle_obs["y"])

        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        next_vehicle_obs, reward_ego, done_ego, info = self.vehicle.step(action)
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
            done_road = True
            reward_collision = min(self.dmin_next)
            info = {}
        else:
            done_road = False
            reward_collision = min(self.dmin_next) / self.max_distant  # 这里可以考虑将CBF加入这里，进行对奖励补偿
            info = {}

        self.vehicle_obs = next_vehicle_obs
        reward = reward_ego + reward_collision
        if done_ego or done_road:
            done = True
        else:
            done = False
        return next_vehicle_obs, reward, done, info

    def render(self):
        self.plot_road()

    def plot_road(self):
        plt.cla()

        # 绘制左右两边的黑色实线
        # view_road_len = 30
        plt.plot(
            [0, self.road_length],
            [0, 0],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [0, self.road_length],
            [self.road_num * self.road_width, self.road_num * self.road_width],
            color="black",
            linewidth=2,
        )

        # 绘制车道线
        for i in range(1, self.road_num):
            lane_y = i * self.road_width
            plt.plot(
                [0, self.road_length],
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
        rectangle_ego = Polygon(car_box, closed=True, edgecolor="red", linewidth=2, facecolor="none")

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
            rectangle = Polygon(surrounding_vehicle_box, closed=True, edgecolor="red", linewidth=2, facecolor="none")

            # 将小车形状添加到图形中
            plt.gca().add_patch(rectangle)

        # 将小车形状添加到图形中
        plt.gca().add_patch(rectangle_ego)
        plt.pause(0.1)


if __name__ == "__main__":
    road_env = RoadEnv()
    obs, done = road_env.reset()

    obs_lists = []
    reward_lists = []
    for i in range(10000):
        obs_lists.append(obs)
        action = [5, 1]
        next_obs, reward, done, info = road_env.step(action)
        obs = next_obs
        reward_lists.append(reward)

        road_env.plot_road()
        # print(obs['force'])
    print("reward:", sum(reward_lists))
