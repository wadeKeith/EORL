import math
import numpy as np
from road_curvature_gradient_build import road_curvature_gradient_build


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
        self.road_gradient = self.road_gradient_cal([0, 0])
        return self.road_curvature, self.road_gradient
    def step(self, x, y):
        self.road_curvature_next = 0
        self.road_gradient_next = self.road_gradient_cal([x, y])[0]
        return self.road_curvature_next, self.road_gradient_next

if __name__ == "__main__":
    road_env = RoadEnv()
    road_env.reset()
    print(road_env.step(500, 4))




