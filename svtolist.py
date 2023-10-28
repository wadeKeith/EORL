import pandas as pd
import numpy as np
import math
import json



class SVtolist_env(object):
    def __init__(self):
        self.surrounding_vehicle_all = pd.read_csv("New_NGSIM.csv")
        self.frames = np.unique(self.surrounding_vehicle_all["Frame_ID"])
        self.surrounding_vehicle_total_time = self.frames.shape[0]
        self.inital_time = self.frames[0]
        self.time_step = None
        self.surrounding_vehicles_lat_min = min(self.surrounding_vehicle_all["Local_X"])
        self.surrounding_vehicles_lat_max = max(self.surrounding_vehicle_all["Local_X"])
        self.surrounding_vehicles_lon_min = 0
        self.surrounding_vehicles_lon_max = 100
        self.surrounding_vehicle_width_max = max(self.surrounding_vehicle_all["v_Width"])
        # parameters f
        self.surrounding_vehicles = None



    def reset(self):
        self.time_step = self.inital_time
        self.surrounding_vehicles = self.surrounding_vehicle_all[(self.surrounding_vehicle_all["Frame_ID"]==self.inital_time) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]>=self.surrounding_vehicles_lon_min) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]<=self.surrounding_vehicles_lon_max)]  # surrounding vehicles



        return self.surrounding_vehicles,self.inital_time,self.surrounding_vehicle_total_time

    def step(self):
        self.time_step = self.time_step + 1
        self.surrounding_vehicles = self.surrounding_vehicle_all[(self.surrounding_vehicle_all["Frame_ID"]==self.time_step) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]>=self.surrounding_vehicles_lon_min) &
                                                                 (self.surrounding_vehicle_all["Local_Y"]<=self.surrounding_vehicles_lon_max)]
        return self.surrounding_vehicles,self.time_step



SV = SVtolist_env()
sv_dic = {}
sv_table_0,initial_time,total_time = SV.reset()
sv_table_dic_0 = sv_table_0.to_dict(orient='list')
sv_dic['%d'%initial_time] = sv_table_dic_0
for i in range(total_time-initial_time):
    sv_table,step_time = SV.step()
    sv_table_dic = sv_table.to_dict(orient='list')
    if sv_table_dic['Vehicle_ID'] == []:
        break
    sv_dic['%d'%step_time] = sv_table_dic

with open("sv_dic_all.json", "w") as file:
    json.dump(sv_dic, file)

