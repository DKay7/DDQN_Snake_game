# from SnakeEnv import SnakeEnv
# import os
# from time import sleep
# import gym
#
# en = gym.make('MountainCar-v0')
# env = SnakeEnv()
# print(env)
# print(env.reset())
# env.render()
# done = False
#
# for i in range(10):
#     env.reset()
#     env.render()
#     done = False
#
#     while not done:
#         os.system('cls')
#         env.render()
#         state, reward, done, _ = env.step(int(input()))
#         print(f'\nState: {state},\nReward {reward}')
#         sleep(2)
#
#
# env.render()
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
    max_epsilon=0.9,
    min_epsilon=0.0005,
    epochs=7**7
)

agent.load_model()
# agent.train()
# agent.plotter()
# c = input('press any key')
state = env.reset()
done = False

# while not done:
#     action = agent.action_choice(state=state, epsilon=0, model=agent.model)
#     new_state, reward, done, _ = env.step(action)
#
#     image = env.render(mode='screenshot')
#     image.show()
#
#     state = new_state

times, mean, unsuccessful, mean_unsuccessful = agent.show_playing(visualize=True,
                                                                  print_=True,
                                                                  mode='human',
                                                                  epochs=12)

print('Mean: ', mean,
      '\nUnsuccessful: ', unsuccessful,
      '\nMean unsuccessful: ', mean_unsuccessful, '%',
      '\nMax time: ', max(times),
      '\nMin time: ', min(times),
      sep='')
