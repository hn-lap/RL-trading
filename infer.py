from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.envs.registration import register
from stable_baselines3 import A2C, DQN, PPO

path_csv = "/home/eco0936_namnh/CODE/trading/datasets/eurusd.csv"
df = pd.read_csv(path_csv, parse_dates=True, index_col="Datetime")
df = df.sort_values("Datetime")
df_test = df["2023":].copy()
window_size = 10
start_index = window_size
end_index = 1000

register(
    id="forex-v100",
    entry_point="envs:ForexEnv",
    kwargs={
        "df": deepcopy(df_test),
        "window_size": window_size,
        "frame_bound": (start_index, end_index),
    },
)

env = gym.make("forex-v100")


model = DQN.load("DQN_model")
obs = env.reset()
while True:
    action, _states = model.predict(obs[np.newaxis, ...])
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()

import quantstats as qs

qs.extend_pandas()

net_worth = pd.Series(
    env.history["total_profit"], index=df.index[start_index + 1 : end_index]
)
returns = net_worth.pct_change().iloc[1:]

qs.reports.full(returns)
qs.reports.html(returns, output="a2c_quantstats.html")
