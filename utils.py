import matplotlib.pyplot as plt


def plot_list(list_data, plt_name):
    plt.plot(list_data)
    plt.title(plt_name)
    plt.show()


###########################################################################
import numpy as np
import math
from shapely import geometry


def sort_vertices(polygon):
    """Sorts vertices by polar angles.

    Args:
        polygon (list[list[float, float]]): list of polygon vertices

    Returns:
        list[list[float, float]]: list of polygon vertices sorted
    """
    cx, cy = polygon.mean(0)  # center of mass
    x, y = polygon.T
    angles = np.arctan2(y - cy, x - cx)
    indices = np.argsort(angles)
    return polygon[indices]


def crossprod(p1, p2):
    """Cross product of two vectors in 2R space.

    Args:
        p1 (list[float, float]): first vector
        p2 (list[float, float): second vector

    Returns:
        float: value of cross product
    """
    return p1[0] * p2[1] - p1[1] * p2[0]


def minkowskisum(pol1, pol2):
    """Calculate Minkowski sum of two convex polygons.

    Args:
        pol1 (np.ndarray[float, float]): first polygon
        pol2 (np.ndarray[float, float]): second polygon

    Returns:
        np.ndarray[np.ndarray[float, float]]: list of the Minkowski sum vertices
    """
    msum = []
    pol1 = sort_vertices(pol1)
    pol2 = sort_vertices(pol2)

    # sort vertices so that is starts with lowest y-value
    min1, min2 = np.argmin(pol1[:, 1]), np.argmin(pol2[:, 1])  # index of vertex with min y value
    pol1 = np.vstack((pol1[:min1], pol1[min1:]))
    pol2 = np.vstack((pol2[:min2], pol2[min2:]))

    i, j = 0, 0
    l1, l2 = len(pol1), len(pol2)
    # iterate through all the vertices
    while i < len(pol1) or j < len(pol2):
        msum.append(pol1[i % l1] + pol2[j % l2])
        cross = crossprod((pol1[(i + 1) % l1] - pol1[i % l1]), pol2[(j + 1) % l2] - pol2[j % l2])
        # using right-hand rule choose the vector with the lower polar angle and iterate this polygon's vertex
        if cross >= 0:
            i += 1
        if cross <= 0:
            j += 1

    return np.array(msum)


def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


def __point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = ((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
                x_min = x2
                y_min = y2
            else:
                distance = ix
                x_min = x1
                y_min = y1
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
            x_min = ix
            y_min = iy
        return distance, x_min, y_min


def distant_min(polygon1, polygon2):
    # calculate minkowski sum
    msum = minkowskisum(polygon1, polygon2 * -1)  # 两个多边形的minkowski sum   msum.shape = (n, 2) n为多边形的顶点数
    polygon_sum = geometry.Polygon([*msum, msum[0]])
    zero_point = geometry.Point(0, 0)
    if polygon_sum.contains(zero_point):
        min_distant = -1
        min_x = 0
        min_y = 0
    else:
        distant = []
        x_min_ls = []
        y_min_ls = []
        for i in range(msum.shape[0] - 1):
            min_distant, x_min, y_min = __point_to_line_distance(
                [0, 0], [msum[i, 0], msum[i, 1], msum[i + 1, 0], msum[i + 1, 1]]
            )
            distant.append(min_distant)
            x_min_ls.append(x_min)
            y_min_ls.append(y_min)
        min_distant, x_min, y_min = __point_to_line_distance(
            [0, 0],
            [
                msum[msum.shape[0] - 1, 0],
                msum[msum.shape[0] - 1, 1],
                msum[0, 0],
                msum[0, 1],
            ],
        )
        distant.append(min_distant)
        x_min_ls.append(x_min)
        y_min_ls.append(y_min)
        min_distant = min(distant)
        min_x = x_min_ls[distant.index(min(distant))]
        min_y = y_min_ls[distant.index(min(distant))]
    return min_distant, min_x, min_y


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


def coordination(car_parmeters):
    """
    :param x: 车辆中心x坐标
    :param y: 车辆中心y坐标
    :param phi: 车辆角度
    :param car_lenth: 车辆长度
    :param car_width: 车辆宽度
    :return: 车辆四个顶点坐标
    """
    x, y,phi, car_length, car_width = car_parmeters
    car_box = [
        [x - car_length, y - car_width / 2],
        [x , y - car_width / 2],
        [x , y + car_width / 2],
        [x - car_length, y + car_width / 2],
    ]

    car_box = rota_rect(car_box, phi, x, y)
    return car_box


def e_s_distance(ego_car_parmeters, surrounding_vehicles):
    """

    Args:
        ego_car_parmeters: [x, y, phi, car_length, car_width]
        surronding_car_parmeters: {1:{x, y, phi, car_length, car_width},       for surrounding vehicles
                                   2:{x, y, phi, car_length, car_width},
                                  ......}

    Returns: minize distance between ego car and surrounding vehicles  [d1, d2, d3, d4, d5, d6, d7, d8]

    """
    d_min = []
    x_min = []
    y_min = []
    ego_car_box = coordination(ego_car_parmeters)

    lateralPos = surrounding_vehicles["Local_X"]
    longitudePos = surrounding_vehicles["Local_Y"]
    id = surrounding_vehicles["Vehicle_ID"]
    len = surrounding_vehicles["v_Length"]
    width = surrounding_vehicles["v_Width"]
    v_class = surrounding_vehicles["v_Class"]
    for i in range(0,surrounding_vehicles.shape[0]):
        surrounding_vehicle_parmeters = [
                longitudePos.values[i],
                lateralPos.values[i],
                0,
                len.values[i],
                width.values[i],
        ]
        surrounding_vehicle_box = coordination(surrounding_vehicle_parmeters)
        min_distant, min_x, min_y = distant_min(ego_car_box, surrounding_vehicle_box)
        d_min.append(min_distant)
        x_min.append(min_x)
        y_min.append(min_y)
    return d_min, x_min, y_min

def direction_distance(ego_car_parmeters, surrounding_vehicles,max_distant):
    d_min_ls, x_min_ls, y_min_ls = e_s_distance(ego_car_parmeters, surrounding_vehicles)
    x_min_1 = []
    y_min_1 = []
    d_min_1 = []
    x_min_2 = []
    y_min_2 = []
    d_min_2 = []
    x_min_3 = []
    y_min_3 = []
    d_min_3 = []
    x_min_4 = []
    y_min_4 = []
    d_min_4 = []
    x_min_5 = []
    y_min_5 = []
    d_min_5 = []
    x_min_6 = []
    y_min_6 = []
    d_min_6 = []
    x_min_7 = []
    y_min_7 = []
    d_min_7 = []
    x_min_8 = []
    y_min_8 = []
    d_min_8 = []
    d_min_c = 1
    for i in range(0,len(d_min_ls)):
        if x_min_ls[i] < 0 and y_min_ls[i] < 0:
            x_min_1.append(x_min_ls[i])
            y_min_1.append(y_min_ls[i])
            d_min_1.append(d_min_ls[i])
        elif x_min_ls[i] < 0 and y_min_ls[i] > 0:
            x_min_2.append(x_min_ls[i])
            y_min_2.append(y_min_ls[i])
            d_min_2.append(d_min_ls[i])
        elif x_min_ls[i] > 0 and y_min_ls[i] < 0:
            x_min_3.append(x_min_ls[i])
            y_min_3.append(y_min_ls[i])
            d_min_3.append(d_min_ls[i])
        elif x_min_ls[i] > 0 and y_min_ls[i] > 0:
            x_min_4.append(x_min_ls[i])
            y_min_4.append(y_min_ls[i])
            d_min_4.append(d_min_ls[i])
        elif x_min_ls[i] == 0 and y_min_ls[i] < 0:
            x_min_5.append(x_min_ls[i])
            y_min_5.append(y_min_ls[i])
            d_min_5.append(d_min_ls[i])
        elif x_min_ls[i] == 0 and y_min_ls[i] > 0:
            x_min_6.append(x_min_ls[i])
            y_min_6.append(y_min_ls[i])
            d_min_6.append(d_min_ls[i])
        elif x_min_ls[i] < 0 and y_min_ls[i] == 0:
            x_min_7.append(x_min_ls[i])
            y_min_7.append(y_min_ls[i])
            d_min_7.append(d_min_ls[i])
        elif x_min_ls[i] > 0 and y_min_ls[i] == 0:
            x_min_8.append(x_min_ls[i])
            y_min_8.append(y_min_ls[i])
            d_min_8.append(d_min_ls[i])
        elif x_min_ls[i] == 0 and y_min_ls[i] == 0:
            d_min_c = -1
    if len(d_min_1) == 0:
        d_min_1_determine = [-300,-33]
    else:
        d_min_1_determine = [x_min_1[d_min_1.index(min(d_min_1))],y_min_1[d_min_1.index(min(d_min_1))]]
    if len(d_min_2) == 0:
        d_min_2_determine = [-300,33]
    else:
        d_min_2_determine = [x_min_2[d_min_2.index(min(d_min_2))],y_min_2[d_min_2.index(min(d_min_2))]]
    if len(d_min_3) == 0:
        d_min_3_determine = [300,-33]
    else:
        d_min_3_determine = [x_min_3[d_min_3.index(min(d_min_3))],y_min_3[d_min_3.index(min(d_min_3))]]
    if len(d_min_4) == 0:
        d_min_4_determine = [300,33]
    else:
        d_min_4_determine = [x_min_4[d_min_4.index(min(d_min_4))],y_min_4[d_min_4.index(min(d_min_4))]]
    if len(d_min_5) == 0:
        d_min_5_determine = [0,-33]
    else:
        d_min_5_determine = [x_min_5[d_min_5.index(min(d_min_5))],y_min_5[d_min_5.index(min(d_min_5))]]
    if len(d_min_6) == 0:
        d_min_6_determine = [0,33]
    else:
        d_min_6_determine = [x_min_6[d_min_6.index(min(d_min_6))],y_min_6[d_min_6.index(min(d_min_6))]]
    if len(d_min_7) == 0:
        d_min_7_determine = [-300,0]
    else:
        d_min_7_determine = [x_min_7[d_min_7.index(min(d_min_7))],y_min_7[d_min_7.index(min(d_min_7))]]
    if len(d_min_8) == 0:
        d_min_8_determine = [300,0]
    else:
        d_min_8_determine = [x_min_8[d_min_8.index(min(d_min_8))],y_min_8[d_min_8.index(min(d_min_8))]]
    d_min_determine = d_min_1_determine+d_min_2_determine+d_min_3_determine+d_min_4_determine+d_min_5_determine+d_min_6_determine+d_min_7_determine+d_min_8_determine
    return d_min_determine,d_min_c

    







if __name__ == "__main__":
    polygon1 = np.array([[2, 1], [4, 1], [4, 3], [2, 3]])
    polygon2 = np.array([[1, 2], [2, 1], [3, 2]])
    # polygon1 = np.array([[-4, 2], [-3, 1], [-6, 2]])
    # polygon2 = np.array([[2, 1], [4, 1], [4, 3], [2, 3]])
    msum = minkowskisum(polygon1, polygon2 * -1)
    # polygon_sum = geometry.Polygon([*msum, msum[0]])
    # zero_point = geometry.Point(0,0)
    # print(zero_point)
    # print(polygon_sum.contains(zero_point))
    # plt.figure(0)
    # plt.plot(polygon_sum.exterior.xy[0], polygon_sum.exterior.xy[1])
    # plt.scatter(zero_point.x, zero_point.y, c='r')
    # plt.show()
    d_min = distant_min(polygon1, polygon2)
    print(d_min)


def smooth(data, sm=100):
    smooth_data = []
    for d in data:
        y = np.ones(sm) * 1 / sm
        d = np.array(d).flatten()
        d = np.convolve(y, d, "same")
        smooth_data.append(d)
    return smooth_data
