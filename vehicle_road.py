from vehicle_env import VehicleEnv
from road_env import RoadEnv
import matplotlib.pyplot as plt
import math

env = VehicleEnv()
road_env = RoadEnv()
obs_road = road_env.reset()
obs = env.reset(obs_road[1])
done = False

obs_lists = []
obs_road_lists = []
reward_lists = []
for i in range(10000):
    obs_lists.append(obs)
    obs_road_lists.append(obs_road[1])
    if i <= 10:
        action = [1, math.pi/4]
    # elif 400<=i < 500:
    #     action = [0, 1]
    else:
        action = [-0.001, 0]
    next_obs, reward, done, info = env.step(action, obs_road[1])
    obs = next_obs
    obs_road = road_env.step(obs[0], obs[1])
    reward_lists.append(reward)
Reward = sum(reward_lists)
x_plot = [obs[0] for obs in obs_lists]
y_plot = [obs[1] for obs in obs_lists]
x_dot_plot = [obs[2] for obs in obs_lists]
y_dot_plot = [obs[3] for obs in obs_lists]
phi_plot = [obs[4] for obs in obs_lists]
omega_plot = [obs[5] for obs in obs_lists]
soc_plot = [obs[6] for obs in obs_lists]
force_plot = [obs[7] for obs in obs_lists]
theta_plot = [theta for theta in obs_road_lists]
from utils import plot_list
plt.figure(1)
plt.subplot(2, 5, 1)
plt.plot(x_plot)
plt.title("x_plot")
plt.subplot(2, 5, 2)
plt.plot(y_plot)
plt.title("y_plot")
plt.subplot(2, 5, 3)
plt.plot(x_dot_plot)
plt.title("x_dot_plot")
plt.subplot(2, 5, 4)
plt.plot(y_dot_plot)
plt.title("y_dot_plot")
plt.subplot(2, 5, 5)
plt.plot(phi_plot)
plt.title("phi_plot")
plt.subplot(2, 5, 6)
plt.plot(omega_plot)
plt.title("omega_plot")
plt.subplot(2,5,7)
plt.plot(x_plot,y_plot)
plt.title("x_plot,y_plot")
plt.subplot(2,5,8)
plt.plot(theta_plot)
plt.title("theta_plot")
plt.subplot(2,5,9)
plt.plot(soc_plot)
plt.title("soc_plot")
plt.subplot(2,5,10)
plt.plot(force_plot)
plt.title("force_plot")
plt.show()
print("")

