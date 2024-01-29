import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
# from Resblock import Resblock

class CNN(nn.Module):

    def __init__(self, img_size, nb_channels, output_size):

        # Construction du CNN (analyse de l'image avec introduction de couches FiLM)
        self.imgSize = img_size
        self.channels = nb_channels
        self.outSize = output_size

        #Première convolution
        self.conv1 = nn.Conv2d(self.channels, 128, 3)
        self.pool1 = nn.AdaptiveMaxPool2d((14, 14))
        # 128 images 14*14 
        
        # Resblocks
        self.resBlock1 = None
        self.resBlock2 = None
        self.resBlock3 = None
        self.resBlock4 = None

        # Couche MLP
        self.classifConv = nn.Conv2d(128, 512)
        self.classifPool = nn.MaxPool2d(1)
        self.flatten = nn.Flatten()
        self.classif_l1 = nn.Linear(512, 1024)
        self.activation_l1 = nn.ReLU()
        self.classif_l2 = nn.Linear(1024, output_size)
        self.classif_out = nn.Softmax(dim=1)


    def forward(self, x):

        # Prediction du modèle
        x = self.conv1(x)
        x = self.pool1(x) 

        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)

        x = self.classifConv(x)
        x = self.classifPool(x)
        x = self.flatten(x)
        x = self.classif_l1(x)
        x = self.activation(x)
        x = self.classif_l2(x)
        return self.classif_out(x)