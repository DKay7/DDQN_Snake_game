from QNetwork import Agent
from SnakeEnv import SnakeEnv

import torch.optim as optim
from torch import nn
import torch
from random import randint
import gym


# создаем объект среды
# и самого агента
env = SnakeEnv(field_size=7)

agent = Agent(
    env=env,
    file_name='snake_7.2',
    max_epsilon=0.9,
    min_epsilon=0.0005,
    epochs=7**7
)

# загружаем модель и
# запускаем тест со сбором статистики
agent.load_model()
times, mean = agent.show_playing(visualize=True,
                                 print_=True,
                                 type_='model',
                                 epochs=12)

print('Mean: ', mean,
      '\nMax time: ', max(times),
      '\nMin time: ', min(times),
      sep='')
