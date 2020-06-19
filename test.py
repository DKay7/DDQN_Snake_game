from SnakeEnv import SnakeEnv
import os
from time import sleep

env = SnakeEnv()
env.reset()
env.render()
done = False
for i in range(1000):
    env.reset()
    env.render()
    done = False
    while not done:
        os.system('cls')
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        sleep(1)

env.render()
