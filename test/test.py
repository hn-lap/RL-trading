import time
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import pandas as pd
from gym.envs.registration import register
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from agents.dqn import Agent
from envs import ForexEnv

path_csv = "/home/ai/notebook/survey_ml4t/data.csv"
df = pd.read_csv(path_csv, parse_dates=True, index_col="Date")
df = df.sort_values("Date")

window_size = 10
start_index = window_size
end_index = len(df)

register(
    id="forex-v100",
    entry_point="envs:ForexEnv",
    kwargs={"df": deepcopy(df), "window_size": 24, "frame_bound": (24, len(df))},
)

env_maker = lambda: gym.make(
    "forex-v100", df=df, window_size=window_size, frame_bound=(start_index, end_index)
)

env = DummyVecEnv([env_maker])

policy_kwargs = dict(net_arch=[64, "lstm", dict(vf=[128, 128, 128], pi=[64, 64])])
model = A2C("MlpLstmPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000)
