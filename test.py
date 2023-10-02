import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

road_width = 23
road_init_width = 10
each_width = 23/6
t = np.linspace(0,50,1000)
y = np.sin((2*np.pi/((20-0)*2))*(t))

plt.plot(
            [0, 300],
            [road_init_width, road_init_width],
            color="black",
            linewidth=2,
        )
plt.plot(
    [0, 300],
    [33, 33],
    color="black",
    linewidth=2,
)

plt.plot(
    [0, 300],
    [33-each_width, 33-each_width],
    color="black",
    linewidth=2,
    linestyle='--'
)
plt.plot(
    [0, 300],
    [33-2*each_width, 33-2*each_width],
    color="black",
    linewidth=2,
    linestyle='--'
)
plt.plot(
    [0, 300],
    [33-3*each_width, 33-3*each_width],
    color="black",
    linewidth=2,
    linestyle='--'
)
plt.plot(
    [0, 300],
    [33-4*each_width, 33-4*each_width],
    color="black",
    linewidth=2,
    linestyle='--'
)
plt.plot(
    [0, 300],
    [33-5*each_width, 33-5*each_width],
    color="black",
    linewidth=2,
    linestyle='--'
)
plt.xlabel("Longitudinal distance(m)")
plt.ylabel("Lateral distance(m)")
plt.title("Road")
plt.ylim(0,road_width+road_init_width+5)
# plt.plot(t,y)
plt.show()
