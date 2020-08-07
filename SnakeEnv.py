import gym
import pygame
import numpy as np
from PIL import Image
from gym import spaces
from functools import reduce
from gym.utils import seeding


class Block:
    """
    Класс блока (вся игра состоит из "блоков" -
        блок змейки, блок пустой клетки, блок яблока
        и т.д.)
    """

    def __init__(self, x, y, color='green'):
        self.x = x
        self.y = y

        self.box = [self.x,
                    self.y,
                    self.x + 1,
                    self.y + 1]
    
        self.color = color


class SnakeGame:
    """
    Класс игры змейка. Обработка всех событий, передвижение,
        генерация яблока происходит тут.
    """
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

        self.circle_check = [-1] * 16
        self.circle_index = 0

        self.actions = {
            # action_code: (delta y, delta x)
            0: (0, 1),
            1: (1, 0),
            2: (0, -1),
            3: (-1, 0),
        }

    def crash_check(self, delta=(0, 0)):
        """
        Проверяет, не столкнется ли змейка с чем-нибудь,
            если сделает шаг на delta

        :param delta: (delta x, delta y) координаты,
            на которые переместится змейка.

        :return: True, если змейка ударилась, иначе False
        """

        for block in self.body[1:]:
            if block.x == self.body[0].x + delta[0] \
               and block.y == self.body[0].y + delta[1]:

                self.done = True

        if self.body[0].x+delta[0] < 0 or \
           self.body[0].x+delta[0] > self.field_size-1 or \
           self.body[0].y+delta[1] < 0 or \
           self.body[0].y+delta[1] > self.field_size-1:

            self.done = True

        return self.done

    def snake_eat(self, delta=(0, 0)):
        """
        Проверяет, не съест ли змейка яблоко,
            если сделает шаг на delta

        :param delta: (delta x, delta y) координаты,
            на которые переместится змейка.

        :return: True, если змейка съела яблоко, иначе False
        """
        if self.body[0].x + delta[0] == self.prize.x \
           and self.body[0].y + delta[1] == self.prize.y:

            self.score += self.reward_for_prize

            self.body.append(Block(*self.last_cords))
            self.prize = self.paste_prize(delta)

            return True

        else:
            return False

    def paste_prize(self, delta=(0, 0)):
        """
        Устанавливает яблоко, так, чтобы змейка не съела его
            следующим ходом на delta

        :param delta: (delta x, delta y) координаты,
            на которые переместится змейка.

        :return: Возвращает объект яблока
        """

        while True:
            not_equals = 0
            prize = list(np.random.randint(0, self.field_size, 2))
            prize = Block(*prize)

            if prize.x == self.body[0].x + delta[0]\
                    and prize.y == self.body[0].y + delta[1]:
                not_equals += 1

            for block in self.body[1:]:
                if prize.x == block.x and prize.y == block.y:
                    not_equals += 1

            if not_equals == 0:
                break

        return prize

    def snake_move(self, action):
        """
        Метод, передвигающий змейку

        :param action: код действия

        :return: Новое состояние, завершилась ли игра,
            съела ли змейка яблоко, двигается ли
            змейка по бесконечному кругу.
        """

        if action not in self.actions.keys():
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        if action != self.last_action:
            self.circle_check[self.circle_index % len(self.circle_check)] = action
            self.circle_index += 1

        if self.is_move_wrong(action):
            action = self.last_action

        dx, dy = self.actions[action]

        done = self.crash_check((dx, dy))
        did_eat = self.snake_eat((dx, dy))
        is_snake_moving_circle = self.circle_checker()

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

        return new_state, done, did_eat, is_snake_moving_circle

    def is_move_wrong(self, action):
        """
        Проверяет, что змейка не может ходить вправо,
            пока дживется влево и т.д.

        :param action: код действия

        :return: True если действие запрещено, иначе False
        """
        if self.actions[action][0] == -self.actions[self.last_action][0] \
                and self.actions[action][1] == -self.actions[self.last_action][1]:
            return True

        else:
            return False

    def circle_checker(self):
        """
        Змейка меня переиграла и ходила кругами,
            этот метод нужен, чтобы переиграть ее.

        :return: True, если змейка гуляет бесконечным
            кругом, иначе False
        """
        for i in range(3, len(self.circle_check)):
            if ((self.circle_check[i-3] == 0 and
                 self.circle_check[i-2] == 1 and
                 self.circle_check[i-1] == 2 and
                 self.circle_check[i] == 3) or

                (self.circle_check[i-3] == 3 and
                 self.circle_check[i-2] == 2 and
                 self.circle_check[i-1] == 1 and
                 self.circle_check[i] == 0)):
                return True

            else:
                return False


class SnakeEnv(gym.Env):
    """
    Удобнее было обернуть игру в класс
        среды gym от openAI
    """
    def __init__(self,
                 field_size=7,
                 cell_size=50,
                 fps=30,
                 seed=None):

        super(SnakeEnv, self).__init__()
        self.metadata = {'render.modes': ['human', 'console', 'screenshot']}

        self.field_size = field_size
        self.seed(seed)
        self.cell_size = cell_size
        self.fps = fps

        self.last_eight_acts = [-1 for _ in range(8)]
        self.index = 0

        if self.seed is not None:
            np.random.seed(seed)

        self.snake_game = SnakeGame(self.field_size,
                                    np.random.randint(0, self.field_size-2))

        self.rewards = {
            'eat_prize': 350,
            'dead': -7500,
            'step': 0,
            'wrong_step': -650,
            'circle': -7500
        }

        self.action_space = spaces.Discrete(len(self.snake_game.actions))
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 2]).astype(np.float32),
                                            np.array([self.field_size,
                                                      self.field_size,
                                                      self.field_size,
                                                      self.field_size,
                                                      self.field_size**2]).astype(np.float32))

    def reset(self):
        """
        Возвращает среду к начальному состоянию

        :return: Состояние среды
        """

        self.snake_game.__init__(self.field_size,
                                 np.random.randint(0, self.field_size-2))

        return [self.snake_game.body[0].x, self.snake_game.body[0].y,
                self.snake_game.prize.x, self.snake_game.prize.y,
                len(self.snake_game.body)]

    def step(self, action):
        """
        Делает один шаг агента в среде

        :param action: действие агента

        :return: Состояние среды после шага
        """

        state, done, did_eat, is_circle = self.snake_game.snake_move(action)

        reward = self.get_reward(done, did_eat, is_circle, action)

        info = {'state': state,
                'score': self.snake_game.score}

        return state, reward, done, info

    def get_reward(self, done, did_eat, is_circle, action):
        """
        Просчитывает награду агента

        :param done: Закончилась ли игра

        :param did_eat: Съел ли агент яблоко

        :param is_circle: Движется ли агент по кругу

        :param action: Действие агента

        :return: Возвращает награду
        """
        reward = 0

        if self.snake_game.is_move_wrong(action):
            reward += self.rewards['wrong_step']

        if did_eat:
            reward += self.rewards['eat_prize']

        if done:
            reward += self.rewards['dead']

        if is_circle:
            reward += self.rewards['circle']

        else:
            reward += self.rewards['step'] / len(self.snake_game.body)

        return reward

    def render(self, mode='console'):
        """
        Отрисовывает среду

        :param mode: Режим отрисовки: консоль, отдельное окно
            или погтовка скриншота для обучения сверточной нс.

        """

        render_game = RenderGame(self.snake_game,
                                 self.field_size,
                                 self.cell_size,
                                 self.fps)

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

        elif mode == 'human':
            render_game.update()

        elif mode == 'screenshot':
            render_game.update()
            return render_game.take_screen()

    def seed(self, seed=None):
        """
        Задает сид среды

        :param seed: Сид среды

        """
        if seed is not None:
            np.random.seed(seed)

    def close(self):
        RenderGame.close()


class RenderGame:
    """
    Дополнительный класс-помошник отрисовки среды
    """

    def __init__(self, snake_game, field_size, cell_size, fps=7):
        pygame.init()
        pygame.mixer.init()
        self.all_sprites = pygame.sprite.Group()

        self.colors = {
            'black': (0, 0, 0),
            'blue': (0, 0, 255),
            'snake': (24, 133, 113),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'dark_green': (14, 92, 40),
            'dark_red': (168, 34, 65),
            'yellow': (245, 163, 39),
            'field': (31, 52, 56)
        }

        self.snake_game = snake_game
        self.field_size = field_size
        self.cell_size = cell_size
        self.ingame_size = self.field_size * self.cell_size

        self.fps = fps

        self.all_sprites.add(Sprite(
            *self.recount_cords(
                self.snake_game.body[0].x,
                self.snake_game.body[0].y
            ),
            self.cell_size,
            color='dark_red'))

        for block in self.snake_game.body[1:]:
            self.all_sprites.add(Sprite(
                *self.recount_cords(
                    block.x,
                    block.y
                ),
                self.cell_size))

        self.all_sprites.add(Sprite(
            *self.recount_cords(
                self.snake_game.prize.x,
                self.snake_game.prize.y
            ),
            self.cell_size,
            color='purple_pizza'))

        self.clock = pygame.time.Clock()
        self.clock.tick(self.fps)
        self.screen = pygame.display.set_mode((self.ingame_size,
                                               self.ingame_size))
        pygame.display.set_caption("Snake")
        self.screen.fill(self.colors['field'])

    def recount_cords(self, x, y):
        """
        Пересчет координат так, чтобы змейка не оказалась между клетками

        :param x: х-координата
        :param y: у-координа
        :return: новые координаты
        """
        new_x = x * self.cell_size + self.cell_size / 2
        new_y = y * self.cell_size + self.cell_size / 2

        return new_x, new_y

    def take_screen(self):
        """
        Делает скриншот окна с игрой

        :return: PIL-картинка игры
        """
        byte_image = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (self.ingame_size, self.ingame_size),
                                byte_image)

        return image

    def update(self):
        """
        Обновляет спрайты всех блоков, для обновления картинки игры

        """

        self.all_sprites.add(Sprite(
            *self.recount_cords(
                self.snake_game.body[0].x,
                self.snake_game.body[0].y
            ),
            self.cell_size,
            color='dark_red'))

        for block in self.snake_game.body[1:]:
            self.all_sprites.add(Sprite(
                    *self.recount_cords(
                        block.x,
                        block.y
                    ),
                    self.cell_size))

        self.all_sprites.add(Sprite(
            *self.recount_cords(
                self.snake_game.prize.x,
                self.snake_game.prize.y
            ),
            self.cell_size,
            color='purple_pizza'))

        self.all_sprites.update()

        self.screen.fill(self.colors['field'])
        self.all_sprites.draw(self.screen)
        pygame.display.flip()

    @staticmethod
    def close():
        """
        Выходит из игры
        """
        pygame.quit()


class Sprite(pygame.sprite.Sprite):
    """
    Вспомогательный класс спрайта.
    """
    def __init__(self, x, y, cell_size, color='snake'):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.cell_size = cell_size

        self.colors = {
            'black': (0, 0, 0),
            'purple_pizza': (255, 0, 204),
            'snake': (42, 151, 156),
            'green': (0, 255, 0),
            'red': (255, 0, 0),
            'dark_green': (14, 92, 40),
            'dark_red': (255, 2, 62),
            'yellow': (245, 163, 39),
            'field': (31, 52, 56)
        }

        self.color = self.colors[color]

        self.image = pygame.Surface((self.cell_size, self.cell_size))
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)

    def update(self):
        """
        Обновляет спрайты.
        """
        self.rect.center = (self.x, self.y)
