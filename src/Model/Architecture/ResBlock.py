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

    def forward(self, x, film_params):
        
        # Prediction du modèle
        x = self.conv1(x)
        x = self.relu(x)
        y = self.conv2(x)
        y = self.batch(y)

        # ============= FiLM =============
        
        # récupération des paramètres gamma, beta
        gamma = film_params[0]
        beta = film_params[1]

        # dim1 --> batch_size
        dim1 = gamma.size(0)

        # dim2 --> nb_channels
        dim2 = gamma.size(1)

        # print(gamma.shape, beta.shape)

        # redimensionnement pour le produit spatial avec chacun des channels des images
        gamma = gamma.view(dim1, dim2, 1, 1)
        beta = beta.view(dim1, dim2, 1, 1)

        # gamma = torch.unsqueeze(torch.unsqueeze(gamma,2),3)
        # beta = torch.unsqueeze(torch.unsqueeze(beta,2),3)

        # print(gamma.shape, beta.shape)

        # FiLM
        y = gamma * x + beta

        # ========================================================================

        y = self.relu(y)
        return x + y
    
