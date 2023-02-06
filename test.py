from copy import deepcopy
from sklearn.preprocessing import scale

import talib
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.envs.registration import register

from custom_agent.agents import Agent
from envs import ForexEnv

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
df = load_data(path_csv='./datasets/eurusd.csv',window_size=12)
window_size = 12

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
    signal_features = env.df.loc[:, ["Close",'returns',"SMA", "RSI", "EMA"]].to_numpy()[
        start:end
    ]
    return prices, signal_features


class MyForexEnv(ForexEnv):
    _process_data = my_process_data


env = MyForexEnv(df=df, window_size=12, frame_bound=(12, len(df)))

# env = gym.make('forex-v100')


agent = Agent(state_dim=window_size)
total_loss_a = []
total_loss_c = []
for e in range(1, 10):
    this_state = env.reset()
    print(f"Epoch: {e}")
    states, actions, rewards, predictions, dones, next_states = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for epi in range(10000):
        action, prediction = agent.act(state=this_state)
        next_state, reward, done, info = env.step(action)
        states.append(np.expand_dims(this_state, axis=0))
        next_states.append(np.expand_dims(next_state, axis=0))
        action_onehot = np.zeros(2)
        action_onehot[action] = 1
        actions.append(action_onehot)
        rewards.append(reward)
        dones.append(done)
        predictions.append(prediction)
        if done:
            print("info", info)
            break
        this_state = next_state
    a_loss, c_loss = agent.replay(
        states, actions, rewards, predictions, dones, next_states
    )
    total_loss_a.append(a_loss.history["loss"][0])
    total_loss_c.append(c_loss.history["loss"][0])


plt.figure(figsize=(16, 6))
env.render_all()
plt.show()

print("Loss Actor:", sum(total_loss_a) / len(total_loss_a))
print("Loss Critic:", (sum(total_loss_c) / len(total_loss_c)))

env.close()
