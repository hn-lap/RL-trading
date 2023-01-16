import numpy as np
import pandas as pd


class Envs:
    def __init__(self) -> None:
        
        pass
    def create_tracking_log(self):
        pass
    def reset(self):
        pass
    def _next_observation(self):
        pass
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