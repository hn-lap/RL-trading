import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from envs.utils import Actions, Positions


class TradingEnv(gym.Env):
    def __init__(self, df, window_size) -> None:
        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        self.start_tick = self.window_size
        self.end_tick = len(self.prices) - 1
        self.done = None
        self.current_tick = None
        self.last_trade_tick = None
        self.position = None
        self.position_history = None
        self.total_reward = None
        self.total_profit = None
        self.first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, action):
        self.done = False
        self.current_tick = self.start_tick
        self.last_trade_tick = self.current_tick - 1
        self.position = Positions.Short
        self.position_history = self.window_size * [None] + self.position
        self.total_reward = 0
        self.total_profit = 1
        self.first_rendering = True
        self.history = {}
        return self.get_observation()

    def step(self, action):
        self.done = False
        self.current_tick += 1

        if self.current_tick == self.end_tick:
            self.done = True

        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward

        self.update_profit(action)

        trade = False
        if (action == Actions.Buy.values and self.position == Positions.Short) or (
            action == Actions.Sell.value and self.position == Positions.Long
        ):
            trade = True

        if trade:
            self.position = self.position.opposite()
            self.last_trade_tick = self.current_tick

        self.position_history.append(self.position)
        obs = self.get_observation()
        info = dict(total_reward=self.total_reward, total_profit=self.total_profit, position=self.position.value)
        self.update_history(info)

        return obs, step_reward, self.done, info

    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def get_observation(self):
        return self.signal_features[(self.current_tick - self.window_size + 1) : self.current_tick]

    def _process_data(self):
        raise NotImplementedError

    def update_profit(self, action):
        raise NotImplementedError

    def calculate_reward(self, action):
        raise NotImplementedError
