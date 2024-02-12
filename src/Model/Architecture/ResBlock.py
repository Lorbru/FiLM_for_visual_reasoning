import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ResBlock(nn.Module):

    def __init__(self, size):
        
        self.size = size

        # Construction d'un ResBlock
        super().__init__()
        self.conv1 = nn.Conv2d(size, size, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(size, size, 3, padding=1)
        self.batch = nn.BatchNorm2d(size)
        self.film = FiLM()

    def forward(self, x):
        # Prediction du modèle
        x = self.conv1(x)
        x = self.relu(x)
        y = self.conv2(x)
        y = self.batch(y)
        
        # y = self.film(y)
        y = self.relu(y)
        return x + y
    
