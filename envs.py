import copy
import random
from collections import deque
from typing import List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from models import Actor_Model, Critic_Model
from utils import TradingGraph


class Envs:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: int,
        lookback_windowsize: int,
        render_range: int,
    ):
        self.df = df.dropna().reset_index()
        self.len_df = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_windowsize = lookback_windowsize
        self.reader_range = render_range

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])
        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_windowsize, 10)

        # Orders history contains the balance, net_worth, buy, sold, hold values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_windowsize)
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_windowsize)

        self.actor_model = Actor_Model(
            input_shape=self.state_size, action_shape=self.action_space.shape[0]
        )
        self.critic_model = Critic_Model(input_shape=self.state_size)

        self.normalize_value = 100000
        
    def create_tracking_log(self):
        self.replay_count = 0
        self.writer = SummaryWriter(comment="Trading")

    def reset(self, env_steps_size: int = 0):
        self.visualization = TradingGraph(render_range=self.reader_range)
        # limited orders memory for visualization
        self.trades = deque(maxlen=self.reader_range)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buy = 0
        self.sell = 0
        self.held = 0
        self.episode_orders = 0
        self.env_steps_size = env_steps_size

        if env_steps_size > 0:  # use for training dataset
            self.start_step = random.randint(
                self.lookback_windowsize, (self.len_df - env_steps_size)
            )
            self.end_step = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_windowsize
            self.end_step = self.len_df

        self.current_step = self.start_step
        for i in reversed(range(self.lookback_windowsize)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.buy, self.sell, self.held]
            )
            self.market_history.append(
                [
                    self.df.loc[current_step, "Open"],
                    self.df.loc[current_step, "High"],
                    self.df.loc[current_step, "Low"],
                    self.df.loc[current_step, "Close"],
                    self.df.loc[current_step, "Volume"],
                ]
            )

        state = np.concatenate((self.market_history, self.orders_history), axis=1)

        return state

    def _next_observation(self):
        self.market_history.append(
            [
                self.df.loc[self.current_step, "Open"],
                self.df.loc[self.current_step, "High"],
                self.df.loc[self.current_step, "Low"],
                self.df.loc[self.current_step, "Close"],
                self.df.loc[self.current_step, "Volume"],
            ]
        )
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    def step(self, action):
        self.buy = 0
        self.sell = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "Close"],
        )
        # for visualization
        Date = self.df.loc[self.current_step, "Date"]
        High = self.df.loc[self.current_step, "High"]
        Low = self.df.loc[self.current_step, "Low"]

        if action == 0:
            pass
        elif action == 1 and self.balance > self.initial_balance / 100:
            self.buy = self.balance / current_price
            self.balance -= self.buy * current_price
            self.held += self.buy
            self.trades.append(
                {
                    "Date": Date,
                    "High": High,
                    "Low": Low,
                    "total": self.buy,
                    "type": "buy",
                }
            )
            self.episode_orders += 1
        elif action == 2 and self.held > 0:
            self.sell = self.held
            self.balance += self.sell * current_price
            self.held -= self.sell
            self.trades.append(
                {
                    "Date": Date,
                    "High": High,
                    "Low": Low,
                    "total": self.sell,
                    "type": "sell",
                }
            )
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.buy, self.sell, self.held]
        )
        reward = self.net_worth - self.prev_net_worth
        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self._next_observation() / self.normalize_value

        return obs, reward, done

    def render(self, visualize=False):
        if visualize:
            Date = self.df.loc[self.current_step, "Date"]
            Open = self.df.loc[self.current_step, "Open"]
            Close = self.df.loc[self.current_step, "Close"]
            High = self.df.loc[self.current_step, "High"]
            Low = self.df.loc[self.current_step, "Low"]
            Volume = self.df.loc[self.current_step, "Volume"]

            # Render the environment to the screen
            self.visualization.render(
                Date, Open, High, Low, Close, Volume, self.net_worth, self.trades
            )

    def get_gaes(
        self,
        rewards,
        dones,
        values,
        next_values,
        gamma=0.99,
        lamda=0.95,
        normalize=True,
    ):
        """
        gaes: Generalized Advantage Estimation
        refers: https://arxiv.org/abs/1506.02438
        """
        deltas = [
            r + gamma * (1 - d) * nv - v
            for r, d, nv, v in zip(rewards, dones, next_values, values)
        ]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]
        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(
        self,
        states: List[np.array],
        actions: List[np.array],
        rewards: List[np.array],
        predictions: List[np.array],
        dones,
        next_states: List[np.array],
    ):
        states = np.vstack(states)
        next_states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.critic_model.predict(states)
        next_values = self.critic_model.predict(next_states)

        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values)
        )

        y_true = np.hstack([advantages, predictions, actions])

        a_loss = self.actor_model.fit(states, y_true, epochs=1, verbose=0, shuffle=True)
        c_loss = self.critic_model.fit(
            states, target, epochs=1, verbose=0, shuffle=True
        )

        self.writer.add_scalar(
            "Data/actor_loss_per_replay",
            np.sum(a_loss.history["loss"]),
            self.replay_count,
        )
        self.writer.add_scalar(
            "Data/critic_loss_per_replay",
            np.sum(c_loss.history["loss"]),
            self.replay_count,
        )
        self.replay_count += 1

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.actor_model.save_weights(f"{name}_Actor.h5")
        self.critic_model.save_weights(f"{name}_Critic.h5")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.actor_model.load_weights(f"{name}_Actor.h5")
        self.critic_model.load_weights(f"{name}_Critic.h5")
