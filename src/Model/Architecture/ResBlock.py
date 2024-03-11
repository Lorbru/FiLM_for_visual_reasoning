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

    def forward(self, x, beta, gamma):
        # Prediction du modèle
        x = self.conv1(x)
        x = self.relu(x)
        id = x

        x = self.conv2(x)
        x = self.batch(x)

        # ============= FiLM =============

        # redimensionnement pour le produit spatial avec chacun des channels des images
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)

        # print(gamma.shape, beta.shape)

        # FiLM
        x = gamma * x + beta

        # ========================================================================

        x = self.relu(x)
        x = x + id
        return x
    
