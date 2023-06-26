import matplotlib.pyplot as plt


def plot_list(list_data, plt_name):
    plt.plot(list_data)
    plt.title(plt_name)
    plt.show()
###########################################################################
import numpy as np
import math

def sort_vertices(polygon):
    """Sorts vertices by polar angles.

    Args:
        polygon (list[list[float, float]]): list of polygon vertices

    Returns:
        list[list[float, float]]: list of polygon vertices sorted
    """
    cx, cy = polygon.mean(0)  # center of mass
    x, y = polygon.T
    angles = np.arctan2(y-cy, x-cx)
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
        cross = crossprod((pol1[(i+1) % l1] - pol1[i % l1]), pol2[(j+1) % l2] - pol2[j % l2])
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
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
        return distance


def distant_min(polygon1,polygon2):

    # calculate minkowski sum
    msum = minkowskisum(polygon1, polygon2* -1)
    distant = []
    for i in range(msum.shape[0]-1):
        distant.append(__point_to_line_distance([0,0],[msum[i,0],msum[i,1],msum[i+1, 0],msum[i+1, 1]]))
    distant.append(__point_to_line_distance([0,0],[msum[msum.shape[0]-1, 0],msum[msum.shape[0]-1, 1],msum[0, 0],msum[0, 1]]))
    return min(distant)



if __name__ == "__main__":
    # polygon1 = np.array([[1, 1], [4, 1], [4, 3], [1, 3]])
    # polygon2 = np.array([[8, 1], [14, 1], [14, 3], [8, 3]])
    polygon1 = np.array([[-4, 2], [-3, 1], [-6, 2]])
    polygon2 = np.array([[2, 1], [4, 1], [4, 3], [2, 3]])
    d_min = distant_min(polygon1, polygon2)
    print(d_min)
