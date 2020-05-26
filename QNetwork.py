import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy


class QNetwork:

    def __init__(self):
        self.device = torch.device('cuda')

        self.model = None
        self.target_model = None
        self.optimizer = None

    def create_model(self):

        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal(layer.weight)

        self.model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 5)
        )

        self.target_model = deepcopy(model)
        self.optimizer = optim.Adam(self.model.parameters())

        self.model.apply(init_weights)
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)

        return self.model, self.target_model, self.optimizer


