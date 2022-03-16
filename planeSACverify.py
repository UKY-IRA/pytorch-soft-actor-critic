import numpy as np
from model_plane import GaussianPolicy
from plane_env import Plane
import torch
import matplotlib.pyplot as plt

state_dict = torch.load('2022-03-10_23-58-59_SAC_Plane_Gaussian_autotune', map_location=torch.device('cpu'))
env = Plane()
model = GaussianPolicy(env.obs_state_len, env.action_space.shape[0], 256, action_space=env.action_space)
model.load_state_dict(state_dict['policy_state_dict'])

n = 0
rewards = []
crashed = 0
while n < 50:
    observation = env.reset()
    done = False
    while not done:
        encoded = observation
        encoded_tensor = torch.FloatTensor(encoded).unsqueeze(0)
        action, _, _ = model.sample(encoded_tensor)
        new_observation, reward, done, info = env.step(action)
        rewards.append(reward)
        observation = new_observation
    if env.t < env.maxtime:
        crashed += 1
    n += 1
print(f"Average Reward over {n} runs {sum(rewards)/n}, crash rate {crashed/50}")



# print(f"start state:{env.state}")
# print("state list:\n================")

for i in range(10):
    xs = []
    ys = []
    observation = env.reset()
    done = False
    while not done:
        xs.append(observation[0][0]*env.xdim)
        ys.append(observation[0][1]*env.ydim)
        encoded = observation
        encoded_tensor = torch.FloatTensor(encoded).unsqueeze(0)
        action, _, _ = model.sample(encoded_tensor)
        new_observation, reward, done, info = env.step(action)
        rewards.append(reward)
        observation = new_observation

    plt.scatter(xs[0], ys[0], marker='o')
    plt.pcolormesh(env.image.T, cmap="RdYlGn", alpha=0.2)
    plt.plot(xs, ys)
    plt.show()
