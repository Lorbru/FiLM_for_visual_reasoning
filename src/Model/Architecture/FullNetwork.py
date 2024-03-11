import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from .ResBlock import ResBlock
from .GRUNet import GRUNet


class FullNetwork(nn.Module):

    def __init__(self, num_channels, output_size, vocab_size, dcr=128, dps=14):
        
        super().__init__()

        # paramètres du modèle
        self.channels = num_channels
        self.outSize = output_size
        self.dcr = dcr
        self.dps = dps

        # Première étape de convolution
        self.conv1 = nn.Conv2d(self.channels, dcr, 3)
        self.pool1 = nn.AdaptiveMaxPool2d((dps, dps))
        # 128 images 14*14 par défaut

        # Resblocks
        self.resBlock1 = ResBlock(dcr)
        self.resBlock2 = ResBlock(dcr)
        self.resBlock3 = ResBlock(dcr)
        self.resBlock4 = ResBlock(dcr)

        # Couche MLP
        self.classifConv = nn.Conv2d(dcr, 512, 1)
        self.classifPool = nn.MaxPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_l1 = nn.Linear(512 * dps * dps, 1024)
        self.activation = nn.ReLU()
        self.classif_l2 = nn.Linear(1024, 1024)
        self.classif_out = nn.Linear(1024, output_size)
        self.activation_out = nn.Softmax(dim=1)
        self.grunet = GRUNet(vocab_size, 2 * 4 * dcr)

    def forward(self, x, z):
        
        # Première étape de convolution
        x = self.conv1(x)
        x = self.pool1(x)

        # Evaluation de la question sur le GRU et génération de paramètres FiLM
        FiLM_params = self.grunet(z).view(-1, 4, 2, self.dcr)

        beta1 = FiLM_params[:, 0, 0, :]
        gamma1 = FiLM_params[:, 0, 1, :]
        x = self.resBlock1(x, beta1, gamma1)

        beta2 = FiLM_params[:, 1, 0, :]
        gamma2 = FiLM_params[:, 1, 1, :]
        x = self.resBlock2(x, beta2, gamma2)

        beta3 = FiLM_params[:, 2, 0, :]
        gamma3 = FiLM_params[:, 2, 1, :]
        x = self.resBlock3(x, beta3, gamma3)

        beta4 = FiLM_params[:, 3, 0, :]
        gamma4 = FiLM_params[:, 3, 1, :]
        x = self.resBlock4(x, beta4, gamma4)

        # Sortie de convolution
        x = self.classifConv(x)
        x = self.classifPool(x)

        # Multi layer perceptron pour la classification finale
        x = self.flatten(x)
        x = self.classif_l1(x)
        x = self.activation(x)
        x = self.classif_l2(x)
        x = self.activation(x)

        return self.activation_out(self.classif_out(x))
    


