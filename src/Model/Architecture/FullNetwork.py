import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from .ResBlock import ResBlock
from GRUNet import GRUNet
from torchvision import transforms
from PIL import Image

class FullNetwork(nn.Module):

    # forward pour la question --> gamma et beta

    # forward pour l'image --> 

    def forward(self, x, z):

        gb = self.gruNet(z)