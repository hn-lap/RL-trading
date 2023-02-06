from copy import deepcopy
from sklearn.preprocessing import scale
import sys
import talib
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
from gym.envs.registration import register
from stable_baselines3 import A2C, DDPG, DQN, PPO,SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from envs import ForexEnv

model_base_agent = {"PPO": PPO, "A2C": A2C, "DQN": DQN, "DDPG": DDPG,'SAC':SAC}


def load_data(path_csv: str, window_size: int = 60):
    df = pd.read_csv(path_csv, parse_dates=True, index_col="Datetime")
    df = df.sort_values("Datetime")
    df['returns'] = df['Close'].pct_change()
    df['return_2'] = df['Close'].pct_change(2)
    df['return_5'] = df['Close'].pct_change(5)
    df['return_21'] = df['Close'].pct_change(21)
    df["SMA"] = talib.SMA(df['Close'], window_size)
    df["RSI"] = talib.RSI(df['Close'])
    df["TRIMA"] = talib.TRIMA(df['Close'])
    df['EMA'] = talib.EMA(df['Close'])
    df.fillna(0, inplace=True)
    r = df['returns'].copy()
    df = pd.DataFrame(scale(df),columns=df.columns,index=df.index)
    features = df.columns.drop("returns")
    df['returns'] = r
    df = df.loc[:,['returns'] + list(features)]
    return df


def create_envs(df,state: str,window_size: int = 60):
    register(
        id="forex-v100",
        entry_point="envs:ForexEnv",
        kwargs={
            "df": deepcopy(df),
            "window_size": window_size,
            "frame_bound": (window_size, len(df)),
        },
    )
    def my_process_data(env):
        start = env.frame_bound[0] - env.window_size
        end = env.frame_bound[1]
        prices = env.df.loc[:, "Close"].to_numpy()[start:end]
        signal_features = env.df.loc[:, ["Close",'returns','return_2','return_5','return_21', "SMA", "RSI", "TRIMA",'EMA']].to_numpy()[
            start:end
        ]
        return prices, signal_features

    class MyForexEnv(ForexEnv):
        _process_data = my_process_data

    if state == 'train':
        df_train = df[:'2023-01-02'].copy()
        env = MyForexEnv(df=df_train,window_size=window_size,frame_bound=(window_size,len(df_train)))
    else:
        df_test = df["2023":].copy()
        env = MyForexEnv(df=df_test,window_size=window_size,frame_bound=(window_size,len(df_test)))
    return env


def train(env, name_model: str):
    env = DummyVecEnv([lambda: env])
    if name_model == 'DQN':
        policy_kwargs = dict(net_arch=dict(128,128))
    else:
        policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    model = model_base_agent[name_model](
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00005,
        # batch_size=128,
        # gradient_steps=10,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=5000000, log_interval=5)
    model.save(f"{name_model}_model")


def infer(env, name_model: str, checkpoint: str):
    model = model_base_agent[name_model].load(checkpoint)
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


def performace_metrics(env, start_index: int, end_index: int):
    """
    @start_index = windown_size
    @end_index   = len(df)
    """
    qs.extend_pandas()
    net_worth = pd.Series(
        env.history["total_profit"], index=df.index[start_index + 1 : end_index]
    )
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.full(returns)
    qs.reports.html(returns, output="quantstats.html")


if __name__ == "__main__":
    # RUN python main.py [options: {'train','test}]
    model_state = sys.argv[1]
    window_size = 30
    df = load_data(path_csv="./datasets/eurusd.csv", window_size=window_size)
    env_train = create_envs(df=df,state='train', window_size=window_size)
    env_test = create_envs(df=df,state='test', window_size=window_size) 
    if model_state == 'train':
        train(env=env_train, name_model="A2C")
    elif model_state == 'test':
        infer(env=env_test,name_model='PPO',checkpoint='PPO_model')