import math
import numpy as np
from scipy import interpolate


def road_curvature_gradient_build(road_lenth, road_width, road_num):
    x_max = road_lenth
    y_max = road_width * road_num
    x_list = np.linspace(0, x_max, 1000)
    y_list = np.linspace(0, y_max, 1000)
    np.random.seed(1234)
    gradient_list = np.random.uniform(-math.pi / 9, math.pi / 9, (1000, 1000))
    gradient_cal = interpolate.RegularGridInterpolator(
        (x_list, y_list), gradient_list, method="linear", bounds_error=False, fill_value=None
    )
    return gradient_cal


if __name__ == "__main__":
    gradient_cal = road_curvature_gradient_build(1000, 3.75, 3)
    print(math.sin(gradient_cal([500, 4])))
