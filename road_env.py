import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon


from vehicle_env import VehicleEnv
from road_curvature_gradient_build import road_curvature_gradient_build
from utils import coordination,e_s_distance





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

        self.surrounding_vehicles = {
            "1": {"x": self.road_length/5, "y": self.road_width/2, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "2": {"x": self.road_length/5*3, "y": self.road_width/2, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "3": {"x": self.road_length/5*5, "y": self.road_width/2, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "4": {"x": self.road_length/5*2, "y": self.road_width/2*3, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "5": {"x": self.road_length/5*4, "y": self.road_width/2*3, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "6": {"x": self.road_length / 4 , "y": self.road_width / 2*5, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "7": {"x": self.road_length / 4*2, "y": self.road_width / 2*5, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "8": {"x": self.road_length / 4*3, "y": self.road_width / 2*5, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},
            "9": {"x": self.road_length / 12 * 11, "y": self.road_width / 2*5, "phi": 0, "v": 0,'car_length':5,'car_width':self.road_width/3*2},

        }  # surrounding vehicles
        self.dmin =None
        self.dmin_next = None
    def reset(self):
        self.road_curvature = 0  # 曲率
        self.road_gradient = self.road_gradient_fun([5, 3.75 * 3 / 2])[0]
        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        self.vehicle_obs = self.vehicle.reset()
        self.dmin = e_s_distance([self.vehicle_obs["x"],self.vehicle_obs["y"],self.vehicle_obs["phi"],self.vehicle.car_length,self.vehicle.car_width]
                                 , self.surrounding_vehicles)
        return self.vehicle_obs,self.dmin

    def step(self, action):
        assert isinstance(action, list), "action must be a list"

        # update road gradient
        self.road_gradient = self.road_gradient_fun([self.vehicle_obs["x"], self.vehicle_obs["y"]])[0]

        # update theta
        self.vehicle.update_theta(math.radians(self.road_gradient))
        next_vehicle_obs, reward, done, info = self.vehicle.step(action)
        self.vehicle_obs = next_vehicle_obs
        self.dmin_next = e_s_distance([self.vehicle_obs["x"], self.vehicle_obs["y"], self.vehicle_obs["phi"], self.vehicle.car_length, self.vehicle.car_width]
                                    , self.surrounding_vehicles)
        return [next_vehicle_obs,self.dmin_next], reward, done, info

    def render(self):
        pass

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


        ego_parmeters = [self.vehicle_obs["x"],self.vehicle_obs["y"],self.vehicle_obs["phi"],self.vehicle.car_length,self.vehicle.car_width ]
        car_box = coordination(ego_parmeters)
        rectangle_ego = Polygon(car_box, closed=True, edgecolor="red", linewidth=2, facecolor="none")

        for key, value in self.surrounding_vehicles.items():
            surrounding_vehicle_parmeters = [value["x"], value["y"], value["phi"], value["car_length"], value["car_width"]]
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

    obs,d_min = road_env.reset()
    done = False

    obs_lists = []
    d_min_lists = []
    reward_lists = []
    for i in range(10000):
        obs_lists.append(obs)
        d_min_lists.append(d_min)
        # if i <= 10:
        #     import math
        #
        #     action = [1, math.pi / 4]
        # # elif 400<=i < 500:
        # #     action = [0, 1]
        # else:
        #     action = [0.5, 0]
        action = [0, 0]
        [next_obs,d_min_next], reward, done, info = road_env.step(action)
        obs = next_obs
        d_min = d_min_next
        reward_lists.append(reward)

        road_env.plot_road()
        # print(obs['force'])
    print('')