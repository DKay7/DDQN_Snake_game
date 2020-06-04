import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class DeepQNetwork(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):

        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.lr = lr
        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), amsgrad=True)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = torch.device('cuda')
        self.to(self.device)

    def forward(self, x):
        x = f.ReLU(self.fc1(x))
        x = f.ReLU(self.fc2(x))
        x = self.fc3(x)

        return x
