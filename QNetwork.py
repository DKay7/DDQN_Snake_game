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
                 model,
                 optimizer,
                 criterion,
                 scheduler,
                 file_name,
                 max_epsilon,
                 min_epsilon,
                 target_update=1000,
                 memory_size=5000,
                 use_scheduler=True,
                 epochs=25,
                 batch_size=16):

        self.gamma = 0.99
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        self.device = torch.device("cuda")

        self.memory = Memory(capacity=memory_size)
        self.env = env

        self.model = model
        self.target_model = deepcopy(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.use_scheduler = use_scheduler
        self.epochs = epochs

        self.file_name = file_name
        self.batch_size = batch_size

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
            q_target = torch.zeros(reward.size()[0]).float().to(self.device)
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target[done] = 0

            q_target = reward + self.gamma * q_target

        q = self.model(state).gather(1, action.unsqueeze(1))

        self.optimizer.zero_grad()

        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()

        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss

    def train(self):

        done = True
        new_state = None

        for epoch in tqdm(range(self.epochs)):

            if done:
                state = self.env.reset()
            else:
                state = new_state

            epsilon = (self.max_epsilon - self.min_epsilon) * epoch / self.epochs
            action = self.action_choice(state, epsilon, self.model)

            new_state, reward, done, _ = self.env.step(action)

            modified_reward = reward + 300 * (self.gamma * abs(new_state[1]) - abs(state[1]))
            self.memory.push((state, action, modified_reward, new_state, done))

            if epoch > self.batch_size:
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))
                tqdm.write('epoch: {0}, loss: {1}'.format(epoch, loss.item()))

            if epoch % self.target_update == 0:
                self.target_model = deepcopy(self.model)
                state = self.env.reset()
                total_reward = 0

                while not done:
                    action = self.action_choice(state, 0, self.target_model)
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward

                self.history.append(total_reward)

    def action_choice(self, state, epsilon, model):
        if random() < epsilon:
            # TODO don't forget
            action = randint(0, 2)

        else:
            action = model(torch.tensor(state).to(self.device).float()).max(0)[1].view(1, 1).item()

        return action

    def plotter(self, history=None, file_name=None):
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

        path = 'plots/' + file_name + '.png'

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))

        axes.plot(self.history, label="score per iterations")

        axes.set_xlabel('iterations')
        axes.set_ylabel('score')
        axes.set_xlim(left=0)

        fig.suptitle(self.file_name)

        fig.savefig(path)
        plt.show()
