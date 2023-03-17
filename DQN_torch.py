import torch
import torch.nn as nn
import numpy as np



class DQN(nn.Module):

    def __init__(self, observations, actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(observations[0], 32, kernel_size=(4, 4))
        self.layer2 = nn.Conv2d(32, 64, kernel_size=(4, 4))
        self.layer3 = nn.Linear(64, actions)

    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))  
        return self.layer3(x)


