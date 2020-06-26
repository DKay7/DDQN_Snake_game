from QNetwork import DeepQNetwork, Agent, Memory
from SnakeEnv import SnakeEnv

import torch.optim as optim
from torch import nn
import torch
from random import randint
import gym

env = SnakeEnv()

agent = Agent(
    env=env,
    file_name='snake_2.0',
    max_epsilon=1,
    min_epsilon=0.0005,
    target_update=1000,
    epochs=2**19
)

agent.load_model()
agent.train()
agent.plotter()

times, mean, unsuccessful, mean_unsuccessful = agent.show_playing(visualize=False,
                                                                  print_=False,
                                                                  epochs=500)

print('Mean: ', mean,
      '\nUnsuccessful: ', unsuccessful,
      '\nMean unsuccessful: ', mean_unsuccessful, '%',
      '\nMax time: ', max(times),
      '\nMin time: ', min(times),
      sep='')
