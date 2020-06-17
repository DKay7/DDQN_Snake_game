from Environment import Environment
from QNetwork import DeepQNetwork, Agent, Memory

import torch.optim as optim
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import gym


env = gym.make("MountainCar-v0")

agent = Agent(
    env=env,
    file_name='test5',
    max_epsilon=0.57,
    min_epsilon=0.05,
    epochs=100000
)

# agent.load_model(agent.model, type_='optim')
# agent.load_model(agent.target_model, type_='target')
agent.train()
# agent.plotter()
# c = input('press any key')


times, mean, unsuccessful, mean_unsuccessful = agent.show_playing(visualize=False, print_=False, epochs=20)

print('Mean: ', mean,
      '\nUnsuccessful: ', unsuccessful,
      '\nMean unsuccessful: ', mean_unsuccessful, '%',
      '\nMax time: ', max(times),
      '\nMin time: ', min(times),
      sep='')







