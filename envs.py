from collections import deque

import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from models import Actor_Model,Critic_Model
from utils import TradingGraph
class Envs:
    def __init__(self,df: pd.DataFrame,initial_balance: int,lookback_windowsize: int,render_range: int):
        self.df = df.dropna().reset_index()
        self.len_df = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_windowsize = lookback_windowsize
        self.reader_range = render_range

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0,1,2])
        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size   = (self.lookback_window_size, 10)

        # Orders history contains the balance, net_worth, buy, sold, hold values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.actor_model  = Actor_Model(input_shape=self.state_size,action_shape=self.action_space.shape[0])
        self.critic_model = Critic_Model(input_shape=self.state_size)
        
    def create_tracking_log(self):
        self.replay_count = 0
        self.writer = SummaryWriter(comment='Trading')
    def reset(self,env_steps_size:int = 0):
        self.visualization = TradingGraph(render_range=self.reader_range)
        # limited orders memory for visualization
        self.trades = deque(maxlen = self.reader_range)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buy  = 0
        self.sell = 0
        self.hold = 0
        self.episode_orders = 0
        self.env_steps_size = env_steps_size

        if env_steps_size > 0:  #use for training dataset
            self.start_step = random.randint(self.lookback_windowsize,(self.len_df - env_steps_size))
            self.end_step   = self.start_step + env_steps_size
        else:
            self.start_step = self.lookback_window_size
            self.end_step = self.len_df
        
        self.current_step = self.start_step
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([
                self.balance,self.net_worth,self.buy,self.sell,self.hold
            ])
            self.market_history.append([
                self.df.loc[current_step,'Open'],
                self.df.loc[current_step, 'High'],
                self.df.loc[current_step, 'Low'],
                self.df.loc[current_step, 'Close'],
                self.df.loc[current_step, 'Volume']
            ])
        
        state = np.concatenate((self.market_history, self.orders_history), axis=1)

        return state


    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs
    def step(self):
        pass
    def reader(self):
        pass
    def get_gaes(self):
        '''
        gaes: Generalized Advantage Estimation
        refers: https://arxiv.org/abs/1506.02438
        '''
        pass
    def replay(self):
        pass
    def act(self):
        pass
    def save(self):
        pass
    def load(self):
        pass