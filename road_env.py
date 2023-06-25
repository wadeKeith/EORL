import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from road_curvature_gradient_build import road_curvature_gradient_build


def rota_rect(box, theta, x, y):
    """
    :param box: 正矩形的四个顶点
    :param theta: 旋转角度
    :param x: 旋转中心(x,y)
    :param y: 旋转中心(x,y)
    :return: 旋转矩形的四个顶点坐标
    """
    # 旋转矩形
    box_matrix = np.array(box) - np.repeat(np.array([[x, y]]), len(box), 0)
    theta = -theta / 180.0 * np.pi
    rota_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)

    new_box = box_matrix.dot(rota_matrix) + np.repeat(np.array([[x, y]]), len(box), 0)
    return new_box


class RoadEnv(object):
    def __init__(self) -> None:
        self.road_curvature = None
        self.road_gradient = None
        # next time step
        self.road_curvature_next = None
        self.road_gradient_next = None

        # parameters for road
        self.road_width = 3.75
        self.road_length = 1000
        self.road_num = 3
        self.road_gradient_cal = road_curvature_gradient_build(self.road_length, self.road_width, self.road_num)

    def reset(self):
        self.road_curvature = 0
        self.road_gradient = self.road_gradient_cal([0, 0])[0]
        return self.road_curvature, self.road_gradient

    def step(self, x, y):
        self.road_curvature_next = 0
        self.road_gradient_next = self.road_gradient_cal([x, y])[0]
        return self.road_curvature_next, self.road_gradient_next

    def render(self):
        pass

    def plot_road(self):
        # 定义道路总长和车道数量
        road_length = 1000
        num_lanes = 3

        # 绘制左右两边的黑色实线
        plt.plot([0, road_length], [0, 0], color="black", linewidth=2)
        plt.plot([0, road_length], [num_lanes * 3, num_lanes * 3], color="black", linewidth=2)

        # 绘制车道线
        for i in range(1, num_lanes):
            lane_y = i * 3
            plt.plot([0, road_length], [lane_y, lane_y], linestyle="dashed", color="black")

        # 小车参数
        car_length = 2
        car_width = 1
        car_x = 500  # 小车的x轴位置
        car_y = 4  # 小车的y轴位置
        car_angle = 30  # 小车与x轴的夹角

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
        plt.xlim(490, 510)
        plt.show()


if __name__ == "__main__":
    road_env = RoadEnv()
    road_env.reset()
    print(road_env.step(500, 4))
    road_env.plot_road()
