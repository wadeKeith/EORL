from normalization import Normalization, RewardScaling

from road_env import RoadEnv

def dictobs2listobs(obs):
    list_obs = []
    for key in obs.keys():
        if key == "dmin":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "x_min":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "y_min":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "dmin_next":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "x_min_next":
            for i in obs[key]:
                list_obs.append(i)
        elif key == "y_min_next":
            for i in obs[key]:
                list_obs.append(i)
        else:
            list_obs.append(obs[key])
    return list_obs

env = RoadEnv()
state = env.reset()
state = dictobs2listobs(state)
state_norm = Normalization(16+30)
state = state_norm(state)
print(state)
done = False
while not done:
    next_obs, reward, done, info = env.step([1,0])
    next_obs = dictobs2listobs(next_obs)
    next_obs = state_norm(next_obs)
    print(next_obs)


