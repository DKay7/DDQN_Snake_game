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
    file_name='snake_1.1',
    max_epsilon=0.9,
    min_epsilon=0.0005,
    epochs=2**19
)

# agent.load_model()
agent.train(load_hist=False)
agent.plotter()
# c = input('press any key')

times, mean, unsuccessful, mean_unsuccessful = agent.show_playing(visualize=True,
                                                                  print_=True,
                                                                  epochs=50)

print('Mean: ', mean,
      '\nUnsuccessful: ', unsuccessful,
      '\nMean unsuccessful: ', mean_unsuccessful, '%',
      '\nMax time: ', max(times),
      '\nMin time: ', min(times),
      sep='')
