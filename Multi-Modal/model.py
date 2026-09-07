import torch
import torch.nn as nn
import torch.nn.functional as F
from crop_utils import N_WATER_ACTIONS, N_NITROGEN_ACTIONS, INPUT_DIM


class Actor(nn.Module):
    """
    Policy network: state -> irrigation + nitrogen probabilities
    Two independent heads for joint water and nitrogen decisions.
    Architecture matches Saikai et al. (2023): 400-600-800-600-400
    """
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 400), nn.ReLU(),
            nn.Linear(400, 600),       nn.ReLU(),
            nn.Linear(600, 800),       nn.ReLU(),
            nn.Linear(800, 600),       nn.ReLU(),
            nn.Linear(600, 400),       nn.ReLU(),
        )
        self.water_head    = nn.Linear(400, N_WATER_ACTIONS)
        self.nitrogen_head = nn.Linear(400, N_NITROGEN_ACTIONS)

    def forward(self, x):
        f = self.shared(x)
        return (F.softmax(self.water_head(f),    dim=-1),
                F.softmax(self.nitrogen_head(f), dim=-1))


class Critic(nn.Module):
    """Value network: state -> V(s)"""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400), nn.ReLU(),
            nn.Linear(400, 600),       nn.ReLU(),
            nn.Linear(600, 800),       nn.ReLU(),
            nn.Linear(800, 600),       nn.ReLU(),
            nn.Linear(600, 400),       nn.ReLU(),
            nn.Linear(400, 1),
        )

    def forward(self, x):
        return self.net(x)
