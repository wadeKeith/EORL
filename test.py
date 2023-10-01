import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

road_width = 23
road_init_width = 10
t = np.linspace(0,50,1000)
y = np.sin((2*np.pi/((20-0)*2))*(t))
plt.plot(t,y)
plt.show()
