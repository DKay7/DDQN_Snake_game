from Environment import Environment
from random import randint


env = Environment()
env.graphic = True

while True:
    env.game_step(randint(0, 4))

