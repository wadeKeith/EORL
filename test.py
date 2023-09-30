import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_max_num_sv(surrounding_vehicles):
    frames = np.unique(surrounding_vehicles["Frame_ID"])
    max_num = 0
    for frame in frames:
        num = surrounding_vehicles[surrounding_vehicles["Frame_ID"] == frame].shape[0]
        if num > max_num:
            max_num = num
    return max_num
surrounding_vehicle = pd.read_csv("New_NGSIM.csv",low_memory=False)

surrounding_vehicle_total_time = np.unique(surrounding_vehicle["Frame_ID"]).shape[0]

num = find_max_num_sv(surrounding_vehicle)
for i in range(1, 100):
    surrounding_vehicle = surrounding_vehicle[surrounding_vehicle["Frame_ID"] != i]
print(surrounding_vehicle_total_time)
print(num)
