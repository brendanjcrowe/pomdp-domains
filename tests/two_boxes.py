import gymnasium as gym
import matplotlib.pyplot as plt

from pdomains import *

env=gym.make('pdomains-two-boxes-v0', rendering=True)
obs, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated or i % 10 == 0:
        obs, info = env.reset()
