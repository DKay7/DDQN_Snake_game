import torch
import os
from time import sleep
from tqdm import tqdm
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
        x = f.softmax(x, dim=-1)

        return x


class DeepConvQNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(DeepConvQNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_w * conv_h * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class Agent:
    def __init__(self,
                 env,
                 file_name,
                 max_epsilon,
                 min_epsilon,
                 target_update=1024,
                 memory_size=4096,
                 epochs=25,
                 batch_size=64):

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
        state, action, reward, next_state, done = batch

        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target = reward + self.gamma * q_target
            q_target[done] = self.env.rewards['dead']

        q = self.model(state).gather(1, action.unsqueeze(1))

        self.optimizer.zero_grad()

        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-5, 5)

        return loss

    def train(self, max_steps=2**10, save_model_freq=100):

        max_steps = max_steps
        loss = 0

        for epoch in tqdm(range(self.epochs)):
            step = 0
            done = False
            torch.cuda.empty_cache()
            self.env.seed(randint(0, self.epochs//2))

            episode_rewards = []
            state_vector = self.env.reset()
            state = self.get_screen()

            while not done and step < max_steps:
                step += 1

                epsilon = (self.max_epsilon - self.min_epsilon) * (1 - epoch / self.epochs)

                action = self.action_choice(state, epsilon, self.model)
                next_state_vector, reward, done, _ = self.env.step(action)
                next_state = self.get_screen()

                new_distance = abs(next_state_vector[0] - next_state_vector[2]) + \
                    abs(next_state_vector[1] - next_state_vector[3])

                old_distance = abs(state_vector[0] - state_vector[2]) + \
                    abs(state_vector[1] - state_vector[3])

                reward = 10 * reward - torch.exp(torch.tensor(new_distance - old_distance, dtype=torch.float64))

                episode_rewards.append(reward)

                if done or step == max_steps:

                    total_reward = sum(episode_rewards)
                    self.memory.push((state, action, reward, next_state, done))

                    tqdm.write(f'Episode: {epoch},\n' +
                               f'Total reward: {total_reward},\n' +
                               f'Training loss: {loss:.4f},\n' +
                               f'Explore P: {epsilon:.4f},\n' +
                               f'Action: {action}\n')

                else:
                    self.memory.push((state, action, reward, next_state, done))
                    state = next_state
                    state_vector = next_state_vector

            if epoch % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % save_model_freq == 0:
                eval_reward, step, snake_len = self.eval_epoch(max_steps)
                self.history.append((snake_len, step))
                self.save_model()

            if epoch > self.batch_size:
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))

        self.save_model()

    def eval_epoch(self, max_steps):
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
        if random() < epsilon:
            action = self.env.action_space.sample()
        else:

            action = model(torch.tensor(state.unsqueeze(0)).to(self.device)).view(-1)
            action = action.max(0)[1].item()

        return action

    def get_screen(self):
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = self.env.render(mode='screenshot')
        image = transformations(image)

        return image

    def plotter(self, file_name=None, visualize=True):
        """
        Строит графики точности и ошибки от эпохи обучения,
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

                # mod_reward = 5 * reward - (abs(next_state[0] - next_state[2]) +
                #                            abs(next_state[1] - next_state[3]) -
                #                            (abs(state[0] - state[2]) +
                #                             abs(state[1] - state[3])))

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
        # unsuccessful = sum(map(lambda x: 0 if x < self.env.snake_game.max_steps else 1, live_times))
        # mean_unsuccessful = unsuccessful / epochs * 100
        return live_times, mean,  0, 0  # unsuccessful, mean_unsuccessful
