import math
import numpy as np
from scipy import interpolate


def road_curvature_gradient_build(road_lenth, road_width, road_num,road_init_width):
    x_max = road_lenth
    y_max = road_width * road_num
    x_list = np.linspace(0, x_max, 1000)
    y_list = np.linspace(road_init_width, road_init_width+y_max, 1000)
    np.random.seed(1234)
    gradient_list = np.random.uniform(-5, 5, (1000, 1000))
    gradient_cal = interpolate.RegularGridInterpolator((x_list, y_list), gradient_list, method="linear")
    return gradient_cal


if __name__ == "__main__":
    gradient_cal = road_curvature_gradient_build(1000, 3.75, 3)
    grad_ls = []
    for x in range(0, 1000, 1):
        for y in range(0, 11, 1):
            grad = gradient_cal([x, y])[0]
            grad_ls.append(grad)

    gradmax = max(grad_ls)
    print(grad_ls)
    print(min(grad_ls), gradmax)

    # gradient_cal = road_curvature_gradient_build(1000, 3.75, 3)
    # print(math.sin(gradient_cal([500, 4])))
