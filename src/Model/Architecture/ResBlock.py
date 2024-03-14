import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ResBlock(nn.Module):
    """
    ============================================================================================
    CLASS RESBLOCK(nn.Module) : a single resblock module with FiLM layer

    METHODS : 
        * __init__(size): convolution output size
        * forward(x, beta, gamma): forward 
    ============================================================================================
    """

    def __init__(self, size):
        """
        -- __init__(size) : constructor

        In >> :
            * size : convolution output size
        """
        self.size = size

        # ResBlock
        super().__init__()
        self.conv1 = nn.Conv2d(size, size, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(size, size, 3, padding=1)
        self.batch = nn.BatchNorm2d(size)

    def forward(self, x, beta, gamma):
        """
        -- _forward(x, beta, gamma) : forward

        In >> :
            * x: features
            * beta: beta parameter for the FiLM
            * gamma: gamma parameter for the FiLM
        """
        # Prediction du modèle
        x = self.conv1(x)
        x = self.relu(x)
        id = x

        x = self.conv2(x)
        x = self.batch(x)

        # ============= FiLM =============
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)
        beta = beta.view(x.size(0), x.size(1), 1, 1)

        # FiLM
        x = gamma * x + beta

        # =================================

        x = self.relu(x)
        x = x + id
        return x
    
