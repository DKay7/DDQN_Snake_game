from QNetwork import DeepQNetwork, Agent, Memory
from SnakeEnv import SnakeEnv

import torch.optim as optim
from torch import nn
import torch
from random import randint
import gym

# создаем объект среды
# и самого агента

env = SnakeEnv(fps=70000)

agent = Agent(
    env=env,
    file_name='snake_7.2',
    max_epsilon=1,
    min_epsilon=0,
    target_update=2000,
    epochs=2**15,
    batch_size=16,
    memory_size=8000
)

# загружаем модель и
# запускаем обучение,
# после коготоро строим график
agent.load_model()
agent.train()
agent.plotter()
