import torch
import os
from time import sleep
from tqdm import tqdm
import gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy
from torchvision import transforms
from random import random, randint, sample


class Memory:
    """
    Класс-буфер для сохранения результатов в формате
    (s, a, r, s', done).
    """
    def __init__(self, capacity):
        """
        :param capacity: размер буфера памяти.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """
        Данный метод сохраняет переданный элемент в циклический буфер.

        :param element: Элемент для сохранения.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.position] = element
            self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        """
        Данный метод возвращает случайную выборку из циклического буфера.

        :param batch_size: Размер выборки.

        :return: Выборка вида [(s1, s2, ... s-i), (a1, a2, ... a-i), (r1, r2, ... r-i),
         (s'1, s'2, ... s'-i), (done1,  done2, ..., done-i)],
            где i = batch_size - 1.
        """
        return list(zip(*sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    """
    Класс полносвязной нейронной сети.
    """

    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):

        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.relu = nn.ReLU()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = torch.device('cuda')
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = f.softmax(x, dim=-1)

        return x


class DeepConvQNet(nn.Module):
    """
    Класс сверточной нейронной сети.
    """
    def __init__(self, h, w, outputs):
        super(DeepConvQNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_w * conv_h * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class Agent:
    """
    Класс агента, обучающегося играть в игру.
    """
    def __init__(self,
                 env,
                 file_name,
                 max_epsilon=1,
                 min_epsilon=0.01,
                 target_update=1024,
                 memory_size=4096,
                 epochs=25,
                 batch_size=64):
        """


        :type env: gym.Env

        :param env gym.Env: Среда, в которой играет агент.
        :param file_name: Имя файла для сохранения и загрузки моделей.
        :param max_epsilon: Макимальная эпсилон для e-greedy police.
        :param min_epsilon: Минимальная эпсилон для e-greedy police.
        :param target_update: Частота копирования параметров из model в target_model.
        :param memory_size: Размер буфера памяти.
        :param epochs: Число эпох обучения.
        :param batch_size: Размер батча.
        """

        self.gamma = 0.97
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        self.file_name = file_name
        self.batch_size = batch_size

        self.device = torch.device("cuda")

        self.memory = Memory(capacity=memory_size)

        self.env = env

        # self.model = DeepQNetwork(input_dims=env.observation_space.shape[0],
        #                           fc1_dims=64,
        #                           fc2_dims=64,
        #                           n_actions=self.env.action_space.n).to(self.device)
        #
        # self.target_model = DeepQNetwork(input_dims=env.observation_space.shape[0],
        #                                  fc1_dims=64,
        #                                  fc2_dims=64,
        #                                  n_actions=self.env.action_space.n).to(self.device)

        self.model = DeepConvQNet(h=self.env.field_size * self.env.cell_size,
                                  w=self.env.field_size * self.env.cell_size,
                                  outputs=self.env.action_space.n).to(self.device)

        self.target_model = DeepConvQNet(h=self.env.field_size * self.env.cell_size,
                                         w=self.env.field_size * self.env.cell_size,
                                         outputs=self.env.action_space.n).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.epochs = epochs

        self.history = []

    def fit(self, batch):
        """
        Метод одной эпохи обучения. Скармливает модели данные,
        считает ошибку, берет градиент и делает шаг градиентного спуска.

        :param batch: Батч данных.

        :return: Возвращает ошибку для вывода в лог.
        """
        state, action, reward, next_state, done = batch

        # Распаковываем батч, оборачиваем данные в тензоры,
        # перемещаем их на GPU

        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            # В этой части кода мы предсказываем максимальное
            # значение q-функции для следующего состояния,
            # см. ур-е Беллмана
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target = reward + self.gamma * q_target

            # Если следующее состояние конечное - то начисляем за него
            # награду за смерть, предусмотренную средой
            q_target[done] = self.env.rewards['dead']

        # Предсказываем q-функцию для действий из текущего состояния
        q = self.model(state).gather(1, action.unsqueeze(1))

        # Зануляем градиент, делаем backward, считаем ошибку,
        # делаем шаг оптимизатора
        self.optimizer.zero_grad()

        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()

        for param in self.model.parameters():
            param.data.clamp_(-1, 1)

        self.optimizer.step()
        
        return loss

    def train(self, max_steps=2**10, save_model_freq=100):
        """
        Метод обучения агента.

        :param max_steps: Из-за того, что в некоторых средах
            агент может существовать бесконечно долго,
            необходимо установить максимальное число шагов.

        :param save_model_freq: Частота сохранения параметров модели
        """

        max_steps = max_steps
        loss = 0

        for epoch in tqdm(range(self.epochs)):
            step = 0
            done = False

            # Очищаем кэш GPU
            torch.cuda.empty_cache()

            episode_rewards = []

            # Получаем начальное состояние среды
            state_vector = self.env.reset()
            state = self.get_screen()

            # Играем одну игру до проигрыша, или пока не сделаем
            # максимальное число шагов
            while not done and step < max_steps:
                step += 1

                # Считаем epsilon для e-greedy police
                epsilon = (self.max_epsilon - self.min_epsilon) * (1 - epoch / self.epochs)

                # Выбираем действие с помощью e-greedy police
                action = self.action_choice(state, epsilon, self.model)

                # Получаем новое состояние среды
                next_state_vector, reward, done, _ = self.env.step(action)
                next_state = self.get_screen()

                # new_distance = abs(next_state_vector[0] - next_state_vector[2]) + \
                #     abs(next_state_vector[1] - next_state_vector[3])
                #
                # old_distance = abs(state_vector[0] - state_vector[2]) + \
                #     abs(state_vector[1] - state_vector[3])
                #
                # reward = reward - 5 * (new_distance - old_distance)

                episode_rewards.append(reward)

                if done or step == max_steps:
                    # Если игра закончилась, добавляем опыт в память

                    total_reward = sum(episode_rewards)
                    self.memory.push((state, action, reward, next_state, done))

                    tqdm.write(f'Episode: {epoch},\n' +
                               f'Total reward: {total_reward},\n' +
                               f'Training loss: {loss:.4f},\n' +
                               f'Explore P: {epsilon:.4f},\n' +
                               f'Action: {action}\n')

                else:
                    # Иначе - добавляем опыт в память и переходим в новое состояние
                    self.memory.push((state, action, reward, next_state, done))
                    state = next_state
                    state_vector = next_state_vector

            if epoch % self.target_update == 0:
                # Каждые target_update эпох копируем параметры модели в target_model,
                # согласно алгоритму
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % save_model_freq == 0:
                # Каждые save_model_freq эпох сохраняем модель
                # и играем тестовую игру, чтобы оценить модель
                eval_reward, step, snake_len = self.eval_epoch(max_steps)
                self.history.append((snake_len, step))
                self.save_model()

            if epoch > self.batch_size:
                # Поскольку изначально наш буфер пуст, нам нужно наполнить его,
                # прежде чем учить модель. Если буфер достаточно полон, то учим модель.
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))

        self.save_model()

    def eval_epoch(self, max_steps):
        """
        Метод оценки модели. По сути бесполезен, но я его оставил,
        ради красивых графиков.

        Играет одну игру, собирая с нее разные данные.

        :param max_steps: Из-за того, что в некоторых средах
            агент может существовать бесконечно долго,
            необходимо установить максимальное число шагов.

        :return: Возвращает общую награду, полученную агентом,
            Количество сделанных шагов (т.е. относительное
            время жизни в среде) и длину тела змейки.
        """
        _ = self.env.reset()
        state = self.get_screen()
        total_reward = 0
        done = False
        step = 0

        while not done and step <= max_steps:
            step += 1
            action = self.action_choice(state, 0, self.model)
            _, reward, done, _ = self.env.step(action)
            next_state = self.get_screen()
            total_reward += reward
            state = next_state

        return total_reward, step, len(self.env.snake_game.body)

    def action_choice(self, state, epsilon, model):
        """
        Метод, реализующий e-greedy политику выбора действий.
        С вероятностью, равной e, будет выбрано случайное действие,
        с вероятностью, равной 1-e, будет выбрано действие,
        предсказанное моделью.

        :param state: Состояние, на основе которого модель делает предсказание.
        :param epsilon: Вероятность совершения случайного действия.
        :param model: Модель, которая будет предсказывать действие.

        :return: Возвращает выбранное действие.
        """
        if random() < epsilon:
            # Выбираем случайное действие из возможных,
            # если случайное число меньше epsilon
            action = self.env.action_space.sample()
        else:
            # Иначе предсказываем полезность каждого действия из даного состояния
            action = model(torch.tensor(state.unsqueeze(0)).to(self.device)).view(-1)
            # И берем argmax() от предсказания, чтобы определить, какое действие
            # лучше всего совершить
            action = action.max(0)[1].item()

        return action

    def get_screen(self):
        """
        Метод, получающий скриншот экрана для свертночной
        нейронной сети.

        :return: Возвращает объект torch.Tensor(),
            содержащий скриншот состояния.
        """
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = self.env.render(mode='screenshot')
        image = transformations(image)

        return image

    def plotter(self, file_name=None, visualize=True):
        """
        Строит графики времени жизни и длины змеи от эпохи обучения,
        затем сохраняет их под именем, переданным в конструктор
        класса

        :param visualize: Выводить ли график. Если False,
        то график просто сохраняется в директорию

        :param file_name: Имя файла для сохранеия графика,
            если не передано, будет взято имя,
            переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        if self.history is None:
            print('Nothing to show')
            return
        else:
            history = self.history

        path = 'plots/' + file_name + '.png'

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

        steps, snake_len = zip(*history)
        axes[0].plot(steps, label='snake len per iteration')
        axes[1].plot(snake_len, label="lifetime per iterations")

        axes[0].set_xlabel('iterations')
        axes[1].set_xlabel('iterations')
        axes[0].set_ylabel('lifetime')
        axes[1].set_ylabel('snake len')

        fig.suptitle(self.file_name)

        fig.savefig(path)

        if visualize:
            plt.show()

    def save_model(self, file_name=None):
        """
        Метод, сохранящиюй параметры модели

        :param file_name: Имя файла для сохранеия графика,
            если не передано, будет взято имя,
            переданное в конструктор класса
        """
        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'

        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name=None):
        """
        Загружает параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'

        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

    def show_playing(self, visualize=True, print_=True, type_='model', epochs=10, mode='human'):
        """
        Метод, показывающий, как агент играет.



        :param visualize: Показывать ли игру, или только собрать статистику.
            Обратите внимание, параметр всегда True, если действие предсказывает
            свертночная нейронная сеть, в силу особенность получения изображения
            в библиотеке  pyGame.

        :param print_: Выводить ли статистику по ходу игры.

        :param type_: 'model', если действия должна выбирать модель,
            'random', если действия должны быть случайным
            (что может пригодится, например, для оценки модели)

        :param epochs: Количество игр, которые сыграет модель.

        :param mode: Режим рендера среды. 'human' - отдельное окно pyGame,
            'console' - печать в консоли


        :return: Массив времени жизни в каждой из игр, среднее вермя жизни.
        """
        live_times = []

        for _ in tqdm(range(epochs)):
            done = False
            live_time = 0

            _ = self.env.reset()
            state = self.get_screen()

            while not done and live_time < 2**10:

                if type_ == 'model':
                    action = self.action_choice(state=state, epsilon=0, model=self.model)
                else:
                    action = self.env.action_space.sample()

                _, reward, done, _ = self.env.step(action)
                next_state = self.get_screen()

                if visualize:
                    self.env.fps = 7
                    if mode == 'console':
                        sleep(0.4)
                        os.system('cls')
                        self.env.render(mode='console')

                    elif mode == 'human':
                        self.env.render(mode='human')

                if print_:
                    print(f"Action: {action}\nReward: {reward}")

                live_time += 1

                state = next_state

            self.env.close()
            live_times.append(live_time)

        mean = sum(live_times)/epochs

        return live_times, mean
