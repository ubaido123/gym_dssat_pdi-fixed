import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        """
        Full-scale Feed-Forward MLP matching Saikai et al. (2023) exactly:
        5 hidden layers: 400 -> 600 -> 800 -> 600 -> 400 nodes 
        Activations: ReLU on hidden layers, Softmax on output 
        """
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 600),
            nn.ReLU(),
            nn.Linear(600, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 400),
            nn.ReLU(),
            nn.Linear(400, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)