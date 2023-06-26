import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build


def rota_rect(box, phi, x, y):
    """
    :param box: 正矩形的四个顶点
    :param phi: 旋转角度
    :param x: 旋转中心(x,y)
    :param y: 旋转中心(x,y)
    :return: 旋转矩形的四个顶点坐标
    """
    # 旋转矩形
    box_matrix = np.array(box) - np.repeat(np.array([[x, y]]), len(box), 0)
    phi = -phi / 180.0 * np.pi
    rota_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]], np.float32)

    new_box = box_matrix.dot(rota_matrix) + np.repeat(np.array([[x, y]]), len(box), 0)
    return new_box


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None

        # parameters for road
        self.road_width = 3.75
        self.road_length = 1000
        self.road_num = 3
        self.road_gradient_fun = road_curvature_gradient_build(self.road_length, self.road_width, self.road_num)

        self.vehicle = VehicleEnv()
        self.vehicle_obs = None  # vehicle observation

    def reset(self):
        self.road_curvature = 0  # 曲率
        self.road_gradient = self.road_gradient_fun([0, 0])[0]
        # update theta
        self.vehicle.update_theta(self.road_gradient)
        self.vehicle_obs = self.vehicle.reset()
        return self.vehicle_obs

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # update road gradient
        self.road_gradient = self.road_gradient_fun([self.vehicle_obs["x"], self.vehicle_obs["y"]])[0]

        # update theta
        self.vehicle.update_theta(self.road_gradient)
        next_vehicle_obs, reward, done, info = self.vehicle.step(action)
        self.vehicle_obs = next_vehicle_obs
        return next_vehicle_obs, reward, done, info

    def render(self):
        pass

    def plot_road(self, obs):
        plt.cla()
        # 定义道路总长和车道数量
        road_length = 1000
        num_lanes = 3

        # 绘制左右两边的黑色实线
        view_road_len = 30
        plt.plot(
            [self.vehicle_obs["x"] - view_road_len, self.vehicle_obs["x"] + view_road_len],
            [0, 0],
            color="black",
            linewidth=2,
        )
        plt.plot(
            [self.vehicle_obs["x"] - view_road_len, self.vehicle_obs["x"] + view_road_len],
            [num_lanes * 3, num_lanes * 3],
            color="black",
            linewidth=2,
        )

        # 绘制车道线
        for i in range(1, num_lanes):
            lane_y = i * 3
            plt.plot(
                [self.vehicle_obs["x"] - view_road_len, self.vehicle_obs["x"] + view_road_len],
                [lane_y, lane_y],
                linestyle="dashed",
                color="black",
            )

        # 小车参数
        car_length = 2
        car_width = 1
        car_x = self.vehicle_obs["x"]  # 小车的x轴位置
        car_y = self.vehicle_obs["y"]  # 小车的y轴位置
        car_angle = self.vehicle_obs["phi"]  # 小车与x轴的夹角

        # 基于小车位置和旋转角度计算小车的四个点坐标
        car_box = [
            [car_x - car_length / 2, car_y - car_width / 2],
            [car_x + car_length / 2, car_y - car_width / 2],
            [car_x + car_length / 2, car_y + car_width / 2],
            [car_x - car_length / 2, car_y + car_width / 2],
        ]

        car_box = rota_rect(car_box, car_angle, car_x, car_y)

        # 创建Polygon对象
        rectangle = Polygon(car_box, closed=True, edgecolor="red", linewidth=2, facecolor="none")

        # 将小车形状添加到图形中
        plt.gca().add_patch(rectangle)
        plt.pause(0.1)


if __name__ == "__main__":
    road_env = RoadEnv()

    obs = road_env.reset()
    done = False

    obs_lists = []
    reward_lists = []
    for i in range(10000):
        obs_lists.append(obs)
        if i <= 10:
            import math

            action = [1, math.pi / 4]
        # elif 400<=i < 500:
        #     action = [0, 1]
        else:
            action = [0.5, 0]
        next_obs, reward, done, info = road_env.step(action)
        obs = next_obs
        reward_lists.append(reward)

        road_env.plot_road(obs)
