import gym
import numpy as np
from gym import spaces
from random import randint


class Block:
    def __init__(self, x, y, color='green'):
        self.x = x
        self.y = y

        self.box = [self.x,
                    self.y,
                    self.x + 1,
                    self.y + 1]

        self.color = color


class SnakeGame:
    # TODO позвать Кира чтобы объяснил почему ничего не работает
    def __init__(self, field_size, position, cell_size):

        self.field_size = field_size
        self.cell_size = cell_size

        # TODO разобраться со скоростью != 1
        self.speed = 1
        self.score = 0
        self.done = False
        self.reward_for_prize = 10
        self.prize = None

        self.snake = None
        self.last_cords = None

        self.reset(position)

        self.actions = {
            # action_code: (delta x, delta y)
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
        }

    def reset(self, position):
        if not (0 <= position <= self.field_size//self.cell_size - 2 * self.cell_size):
            print('Стартовая позиция некорректна')
            position = field_size // 2

        self.snake = [Block(position, position, color='red'), Block(position+self.cell_size, position)]
        self.speed = 1
        self.score = 0
        self.done = False
        self.prize = self.paste_prize()
        self.last_cords = [self.snake[-1].x,
                           self.snake[-1].y]

    def crash_check(self):

        for block in self.snake[1:]:
            if block.x == self.snake[0].x \
               and block.y == self.snake[0].y:

                self.done = True

        # for i in range(len(self.snake)):
        #     for j in range(i+1, len(self.snake)):
        #         if self.snake[i].x == self.snake[j].x\
        #            and self.snake[i].y == self.snake[j].y:
        #
        #             self.done = True

        if self.snake[0].x < 0 or \
           self.snake[0].x > self.field_size//self.cell_size-1 or \
           self.snake[0].y < 0 or \
           self.snake[0].y > self.field_size//self.cell_size-1:

            self.done = True

        return self.done

    def snake_eat(self):
        if self.snake[0].x == self.prize.x \
           and self.snake[0].y == self.prize.y:

            self.score += self.reward_for_prize

            self.prize = self.paste_prize()

            self.snake.append(Block(self.last_cords[0],
                                    self.last_cords[1]))

            return True

        else:
            return False

    def paste_prize(self):
        prize = [randint(0, self.field_size//self.cell_size)-1,
                 randint(0, self.field_size//self.cell_size)-1]
        return Block(*prize,
                     color='blue')

    def snake_move(self, action):

        if action not in self.actions.keys():
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        dx, dy = self.actions[action]

        for block_id in range(len(self.snake)-1, 0, -1):
            self.snake[block_id].x = self.snake[block_id-1].x
            self.snake[block_id].y = self.snake[block_id-1].y
        # TODO разобраться со скоростью != 1
        # TODO разобраться с ошибкой, когда змея ходит в стену!

        self.snake[0].x += dx * self.speed * self.cell_size
        self.snake[0].y += dy * self.speed * self.cell_size

        self.last_cords = [self.snake[-1].x,
                           self.snake[-1].y]


class SnakeEnv(gym.Env):
    def __init__(self,
                 field_size=7,
                 cell_size=1):
        super(SnakeEnv, self).__init__()
        self.metadata = {'render.modes': ['human', 'console']}

        self.field_size = field_size * cell_size
        self.cell_size = cell_size

        self.rewards = {
            'eat_prize': 5,
            'dead': -10,
            'step': 2
        }

        self.snake_game = SnakeGame(self.field_size,
                                    self.field_size // 2,
                                    self.cell_size)

        self.action_space = spaces.Discrete(len(self.snake_game.actions))
        self.observation_space = spaces.Discrete((self.field_size//self.cell_size)**2)

    def reset(self):
        self.snake_game.reset(randint(0, self.field_size-2*self.cell_size))

        return np.array([self.snake_game.snake[0].x, self.snake_game.snake[0].y,
                         self.snake_game.prize.x, self.snake_game.prize.y])

    def step(self, action):

        self.snake_game.snake_move(action)

        state = np.array([self.snake_game.snake[0].x,
                          self.snake_game.snake[0].y,
                          self.snake_game.prize.x,
                          self.snake_game.prize.y])

        reward = 0
        done = self.snake_game.crash_check()

        if self.snake_game.snake_eat():
            reward += self.rewards['eat_prize']

        if done:
            reward += self.rewards['dead']

        else:
            reward += self.rewards['step']

        info = {'state': state,
                'score': self.snake_game.score}

        return state, reward, done, info

    def render(self, mode='console'):

        if mode not in self.metadata['render.modes']:
            raise NotImplementedError()

        if mode == 'console':

            real_size = self.field_size//self.cell_size

            field = [['=' for _ in range(real_size)]
                     for __ in range(real_size)]

            for block in self.snake_game.snake[1:]:
                field[block.x][block.y] = 's'

            print(self.snake_game.prize.x, self.snake_game.prize.y)
            print(self.snake_game.snake[0].x, self.snake_game.snake[0].y)

            field[self.snake_game.prize.x][self.snake_game.prize.y] = 'P'

            field[self.snake_game.snake[0].x][self.snake_game.snake[0].y] = 'S'

            for row in field:
                print(*row)

    def close(self):
        pass
