import pygame
import snake
import random
import prize_object
import Game_Parameters


class Environment:
    """
    Класс среды
    """
    def __init__(self):
        # Creating snake
        self.snakes = [snake.SnakeElement(Game_Parameters.white), snake.SnakeElement(Game_Parameters.white)]
        self.snakes[1].set_snake_element_x(self.snakes[1].get_snake_element_x() + 20)

        # Creating prize
        self.prize = prize_object.Prize()

        # Graphics on/off
        self.graphic = False

        # Reward field
        self.reward = 0

        # Last coordinates
        self.last_x = self.snakes[1].get_snake_element_x()
        self.last_y = self.snakes[1].get_snake_element_y()

        # Game init
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((Game_Parameters.width, Game_Parameters.height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        self.game_running = 1

        # Create sprite group and adding prize and snake to this group
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(i for i in self.snakes)
        self.all_sprites.add(self.prize)

    def crash(self):
        """
        Определяет, погибла змейка или нет ( гибель - столкновение со стенками или самой змейкой)

        :rtype: bool

        :return: Если произошло столкновение - True, иначе - False
        """
        k = 0

        for i in range(1, len(self.snakes)):
            if self.snakes[i].get_snake_element_x() == self.snakes[0].get_snake_element_x() \
                    and \
                    self.snakes[i].get_snake_element_y() == self.snakes[0].get_snake_element_y():
                k += 1
                break

        if self.snakes[0].get_snake_element_x() < 0 \
                or self.snakes[0].get_snake_element_x() > Game_Parameters.width \
                or self.snakes[0].get_snake_element_y() < 0 \
                or self.snakes[0].get_snake_element_y() > Game_Parameters.height:
            k += 1

        if k > 0:
            return True
        else:
            return False

    def eat_prize(self):
        """
        Определяет, было ли сьедено змейкой яблоко ( здесь - "приз" )\n
        Если "приз" был сьеден - генерирует новые координаты приза

        :rtype: bool

        :return: Если приз был сьеден - True, иначе - False
        """
        if self.snakes[0].get_snake_element_x() == self.prize.get_prize_x() and \
                self.snakes[0].get_snake_element_y() == self.prize.get_prize_y():

            cord_equality = True

            while cord_equality:
                k = 0
                for i in self.snakes:
                    if i.get_snake_element_x() == self.prize.get_prize_x() \
                            and \
                            i.get_snake_element_y() == self.prize.get_prize_y():
                        k += 1

                if k > 0:
                    self.prize.set_prize_x(random.randint(10, Game_Parameters.width - 10) // 2 // 10 * 2 * 10 + 10)
                    self.prize.set_prize_y(random.randint(10, Game_Parameters.height - 10) // 2 // 10 * 2 * 10 + 10)
                else:
                    cord_equality = False

            return True
        else:
            return False

    def upgrade(self):
        """
        Увеличивает длину змейки на один элемент
        """
        self.snakes.append(snake.SnakeElement(Game_Parameters.white))
        self.snakes[len(self.snakes) - 1].set_snake_element_x(self.last_x)
        self.snakes[len(self.snakes) - 1].set_snake_element_y(self.last_y)
        self.all_sprites.add(self.snakes[len(self.snakes) - 1])

    def game_step(self, chosen_action):
        """
        Исполняет один шаг игры - сдвигает змейку, проверяет ее на возможное столкновение и поедание приза,\
        при необходимости отрисовывает новый кадр ( отрисовку можно отключить установив поле graphic\
        класса Environment на False)

        :rtype: tuple

        :param chosen_action: Выбранное нейросетью действие для данного шага игры

        :return: Резульат функции get_data()
        """
        self.reward = -1

        if chosen_action == 1 and snake.orientation != 'right':
            snake.orientation = 'left'
        elif chosen_action == 1 and snake.orientation == 'right':
            self.reward += Game_Parameters.incorrect_step_reward

        elif chosen_action == 3 and snake.orientation != 'left':
            snake.orientation = 'right'
        elif chosen_action == 3 and snake.orientation == 'left':
            self.reward += Game_Parameters.incorrect_step_reward

        elif chosen_action == 2 and snake.orientation != 'down':
            snake.orientation = 'up'
        elif chosen_action == 2 and snake.orientation == 'down':
            self.reward += Game_Parameters.incorrect_step_reward

        elif chosen_action == 4 and snake.orientation != 'up':
            snake.orientation = 'down'
        elif chosen_action == 4 and snake.orientation == 'up':
            self.reward += Game_Parameters.incorrect_step_reward

        snake.move(self.snakes)

        self.last_x = self.snakes[len(self.snakes) - 1].get_snake_element_x()
        self.last_y = self.snakes[len(self.snakes) - 1].get_snake_element_y()

        if self.crash():
            self.game_running = 0
            self.reward += Game_Parameters.crash_reward

        if self.eat_prize():
            self.upgrade()
            self.reward += Game_Parameters.eating_reward

        if self.graphic:
            # Update
            self.all_sprites.update()

            # Render
            self.screen.fill(Game_Parameters.black)
            self.all_sprites.draw(self.screen)
            pygame.display.flip()

        self.get_data()

    @staticmethod
    def end_game():
        """
        Заканчивает игру
        """
        pygame.quit()

    def get_data(self):
        """
        Определяет текущее состояние системы: наличие приза в пределах 10 клеток вокруг (если приз есть -\
        возвращает его координаты, иначе (-1, -1) ), координаты головы змеи, награду за текущий шаг и\
        переменную состояние игры (True, если игра продолжается, False, если игра закончилась)

        :rtype: tuple, bool

        :return: Tuple, Bool - такие, что Tuple =  (координата Х приза, координата Y приза, координата Х головы,\
            координата Y головы, награда за текущее действие), Bool = переменная состояния игры
        """
        if self.snakes[0].get_snake_element_x() + 10 * Game_Parameters.cell >= self.prize.get_prize_x() \
                >= self.snakes[0].get_snake_element_x() - 10 * Game_Parameters.cell and \
                self.snakes[0].get_snake_element_y() + 10 * Game_Parameters.cell >= self.prize.get_prize_y() \
                >= self.snakes[0].get_snake_element_y() - 10 * Game_Parameters.cell:
            prize_cord = (self.prize.get_prize_x(), self.prize.get_prize_y())
        else:
            prize_cord = (-1, -1)

        head_cord_and_reward = (self.snakes[0].get_snake_element_x(), self.snakes[0].get_snake_element_y(), self.reward)

        return prize_cord + head_cord_and_reward, self.game_running

    def return_to_start(self):
        """
        Возвращает данные к начальному значению:
        Длину змеи устанавливает на 2, помещает змею в центр
        """
        self.snakes = [snake.SnakeElement(Game_Parameters.white), snake.SnakeElement(Game_Parameters.white)]
        self.snakes[1].set_snake_element_x(self.snakes[1].get_snake_element_x() + 20)
        self.all_sprites.add(i for i in self.snakes)
