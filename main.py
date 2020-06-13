from Environment import Environment
from QNetwork import DeepQNetwork, Agent, Memory

import torch.optim as optim
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import gym


env = gym.make("MountainCar-v0")

dqn = DeepQNetwork(input_dims=2, fc1_dims=32, fc2_dims=32, n_actions=3)

optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss = nn.MSELoss()

agent = Agent(
    env=env,
    model=dqn,
    optimizer=optimizer,
    criterion=loss,
    scheduler=None,
    file_name='kuku1',
    max_epsilon=0.7,
    min_epsilon=0.1,
    epochs=100000
)

agent.train()
agent.plotter()








