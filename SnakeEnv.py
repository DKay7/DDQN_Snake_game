import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


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
    def __init__(self, field_size, position, max_steps=10, seed=None):

        self.field_size = field_size

        if not (0 <= position <= self.field_size - 2):
            print('Стартовая позиция некорректна')
            position = field_size // 2

        self.body = [Block(position, position, color='red'), Block(position + 1, position)]

        self.prize = self.paste_prize()

        self.done = False

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.max_steps = max_steps
        self.step = 0

        self.score = 0
        self.step = 0
        self.reward_for_prize = 10

        self.last_cords = [self.body[-1].x,
                           self.body[-1].y]
        self.last_action = 3

        self.actions = {
            # action_code: (delta y, delta x)
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
        }

    def crash_check(self, delta=(0, 0)):

        for block in self.body[1:]:
            if block.x == self.body[0].x \
               and block.y == self.body[0].y:

                self.done = True

        if self.body[0].x+delta[0] < 0 or \
           self.body[0].x+delta[0] > self.field_size-1 or \
           self.body[0].y+delta[1] < 0 or \
           self.body[0].y+delta[1] > self.field_size-1:

            self.done = True


        return self.done

    def snake_eat(self, delta=(0, 0)):
        if self.body[0].x + delta[0] == self.prize.x \
           and self.body[0].y + delta[1] == self.prize.y:

            self.score += self.reward_for_prize

            self.prize = self.paste_prize()

            self.body.append(Block(self.last_cords[0],
                                   self.last_cords[1]))

            return True

        else:
            return False

    def paste_prize(self):
        coordinates_equality = True
        prize = [0, 0]

        while coordinates_equality:

            prize = list(np.random.randint(0, self.field_size-1, 2))
            prize = Block(*prize,
                          color='blue')

            for block in self.body:
                coordinates_equality = coordinates_equality and \
                                       (prize.x == block.x and prize.y == block.y)

        return prize

    def snake_move(self, action):

        if action not in self.actions.keys():
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        if self.is_move_wrong(action):
            action = self.last_action

        dx, dy = self.actions[action]

        done = self.crash_check((dx, dy))
        did_eat = self.snake_eat((dx, dy))

        if not done:

            for block_id in range(len(self.body) - 1, 0, -1):
                self.body[block_id].x = self.body[block_id - 1].x
                self.body[block_id].y = self.body[block_id - 1].y

            self.body[0].x += dx
            self.body[0].y += dy

            self.last_cords = [self.body[-1].x,
                               self.body[-1].y]

            self.last_action = action
            self.step += 1

            new_state = [self.body[0].x,
                         self.body[0].y,
                         self.prize.x,
                         self.prize.y,
                         len(self.body)]

        else:
            new_state = self.last_cords+[self.prize.x, self.prize.y, len(self.body)]

        return new_state, done, did_eat

    def is_move_wrong(self, action):
        if self.actions[action][0] == -self.actions[self.last_action][0] \
        and self.actions[action][1] == -self.actions[self.last_action][1]:
            return True

        else:
            return False


class SnakeEnv(gym.Env):
    def __init__(self,
                 field_size=7,
                 seed=None):

        super(SnakeEnv, self).__init__()
        self.metadata = {'render.modes': ['human', 'console']}

        self.field_size = field_size
        self.seed(seed)

        if self.seed is not None:
            np.random.seed(seed)

        self.rewards = {
            'eat_prize': 30,
            'dead': -100,
            'step': -2,
            'wrong_step': -10
        }

        self.snake_game = SnakeGame(self.field_size,
                                    np.random.randint(0, self.field_size-2))

        self.action_space = spaces.Discrete(len(self.snake_game.actions))
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 2]).astype(np.float64),
                                            np.array([self.field_size,
                                                      self.field_size,
                                                      self.field_size,
                                                      self.field_size,
                                                      self.field_size**2]).astype(np.float64))

    def reset(self):

        self.snake_game.__init__(self.field_size,
                                 np.random.randint(0, self.field_size-2))

        return [self.snake_game.body[0].x, self.snake_game.body[0].y,
                self.snake_game.prize.x, self.snake_game.prize.y,
                len(self.snake_game.body)]

    def step(self, action):

        state, done, did_eat = self.snake_game.snake_move(action)

        reward = self.get_reward(done, did_eat, action)

        info = {'state': state,
                'score': self.snake_game.score}

        return state, reward, done, info

    def get_reward(self, done, did_eat, action):
        reward = 0

        if self.snake_game.is_move_wrong(action):
            reward += self.rewards['wrong_step']

        if did_eat:
            reward += self.rewards['eat_prize']

        if done:
            reward += self.rewards['dead']

        else:
            reward += self.rewards['step'] + len(self.snake_game.body)

        return reward

    def render(self, mode='console'):

        if mode not in self.metadata['render.modes']:
            raise NotImplementedError()

        if mode == 'console':

            field = [['=' for _ in range(self.field_size)]
                     for __ in range(self.field_size)]

            for block in self.snake_game.body[1:]:
                field[block.x][block.y] = 's'

            field[self.snake_game.prize.x][self.snake_game.prize.y] = 'P'

            field[self.snake_game.body[0].x][self.snake_game.body[0].y] = 'S'

            for row in field[::-1]:
                print(*row)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def close(self):
        pass
