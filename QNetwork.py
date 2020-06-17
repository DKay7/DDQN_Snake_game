import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy
from random import random, randint, sample


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.position] = element
            self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):

        return list(zip(*sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):

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

        return x


class Agent:
    def __init__(self,
                 env,
                 optimizer,
                 criterion,
                 file_name,
                 max_epsilon,
                 min_epsilon,
                 model,
                 target_model,
                 target_update=1000,
                 memory_size=4096,
                 epochs=25,
                 batch_size=16):

        self.gamma = 0.97
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        self.file_name = file_name
        self.batch_size = batch_size

        self.device = torch.device("cuda")

        self.memory = Memory(capacity=memory_size)
        self.env = env
        self.env.seed(randint(0, 420000))

        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.criterion = criterion

        self.epochs = epochs

        self.history = []

    def fit(self, batch):
        state, action, reward, next_state, done = batch
        # TODO разобраться в порядке применения .to() и .float()
        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target = reward + self.gamma * q_target

        q = self.model(state).gather(1, action.unsqueeze(1))

        self.optimizer.zero_grad()

        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss

    def train(self):

        done = True
        new_state = None
        eval_reward = None
        best_eval_reward = float('-inf')
        best_params = None

        for epoch in tqdm(range(self.epochs)):

            self.env.seed(randint(0, 42000))

            if done:
                state = self.env.reset()
            else:
                state = new_state

            epsilon = (self.max_epsilon - self.min_epsilon) * epoch / self.epochs
            action = self.action_choice(state, epsilon, self.model)

            new_state, reward, done, _ = self.env.step(action)

            modified_reward = self.modified_reward(reward, state, new_state)

            self.memory.push((state, action, modified_reward, new_state, done))

            if epoch % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                eval_reward = self.eval_epoch()
                self.history.append(eval_reward)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_params = self.target_model.state_dict()

            if epoch > self.batch_size:
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))
                loss = str(round(loss.item(), 7)) + '0' * (7-len(str(round(loss.item(), 7))))

                tqdm.write('epoch: {0},\tloss: {1},\tlast eval reward: {2}'.format(epoch,
                                                                                   loss,
                                                                                   eval_reward).expandtabs())
        self.target_model.load_state_dict(best_params)
        self.save_model(self.model, type_='optim')
        self.save_model(self.target_model, type_='target')

    def modified_reward(self, reward, state, new_state):
        modify_reward = reward + 300 * (abs(new_state[1]) - self.gamma * abs(state[1]))
        return modify_reward

    def eval_epoch(self):
        state = self.env.reset()
        total_reward = 0
        done = False

        self.target_model.eval()

        while not done:
            action = self.action_choice(state, 0, self.target_model)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward

        self.target_model.train()

        return total_reward

    def action_choice(self, state, epsilon, model):
        if random() < epsilon:
            action = self.env.action_space.sample()

        else:
            action = model(torch.tensor(state).to(self.device).float()).max(0)[1].item()

        return action

    def plotter(self, history=None, file_name=None, visualize=True):
        """
        Строит графики точности и ошибки от эпохи обучения,
        затем сохраняет их под именем, переданным в конструктор
        класса

        :param file_name: Имя файла для сохранеия графика,
            если не передано, будет взято имя,
            переданное в конструктор класса

        :param history: история обучения, по которой
            нужно строить график. Если не передано, то
            будет использована история из директории
            /histories с именем файла, переданным в
            конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        if history is None:
            history = self.get_history()

        path = 'plots/' + file_name + '.png'

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

        axes.plot(history, label="score per iterations")

        axes.set_xlabel('iterations')
        axes.set_ylabel('score')
        axes.set_xlim(left=0)

        fig.suptitle(self.file_name)

        fig.savefig(path)
        if visualize:
            plt.show()

    def save_model(self, model, file_name=None, type_='optim'):
        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '_' + type_ + '.pth'

        torch.save(model.state_dict(), path)

    def load_model(self, model, file_name=None, type_='optim'):
        """
        Загружает параметры нейронной сети

        :param type_: тип модели (оптимизитор или таргет)
        :param model: объект модели для записи загруженных параметров
        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '_' + type_ + '.pth'

        model.load_state_dict(torch.load(path))

    def save_history(self, file_name=None):
        """
        Сохраняет историю обучения.

        :param file_name: Имя файла для сохранеия истории,
            если не передано, будет взято имя,
            переданное в конструктор класса
        """
        if file_name is None:
            file_name = self.file_name

        with open('histories/' + file_name + '.pickle', 'wb') as file:
            torch.save(self.history, file)

    def get_history(self, file_name=None):
        """

        :param file_name: Имя файла для загрузки истории,
            если не передано, будет взято имя,
            переданное в конструктор класса
        :return:
        """
        if file_name is None:
            file_name = self.file_name

        if self.history is None or self.history == []:
            with open('histories/' + file_name + '.pickle', 'rb') as file:
                self.history = torch.load(file)

        return self.history

    def show_playing(self, visualize=True, print_=True, epochs=10):
        live_times = []

        for _ in tqdm(range(epochs)):
            done = False
            live_time = 0
            self.env.seed(randint(0, 2**17))
            state = self.env.reset()
            self.target_model.eval()

            while not done:
                action = self.action_choice(state=state, epsilon=0, model=self.target_model)

                new_state, reward, done, _ = self.env.step(action)
                modified_reward = self.modified_reward(reward, state, new_state)

                if visualize:
                    self.env.render(mode='human')

                if print_:
                    tqdm.write(f"Action: {action}\nReward: {reward}\nModified reward: {modified_reward}")

                live_time += 1

                state = new_state

            self.env.close()
            live_times.append(live_time)

        mean = sum(live_times)/epochs
        unsuccessful = sum(map(lambda x: 0 if x < 200 else 1, live_times))
        mean_unsuccessful = unsuccessful / epochs * 100
        return live_times, mean,  unsuccessful, mean_unsuccessful
