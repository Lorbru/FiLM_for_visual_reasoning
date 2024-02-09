import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from .ResBlock import ResBlock
from torchvision import transforms
from PIL import Image




class CNN(nn.Module):
    DEFAULT_CHANNELS_RESBLOCK = 128
    DEFAULT_POOLING_SIZE = 14

    def __init__(self, img_size, nb_channels, output_size, dcr=DEFAULT_CHANNELS_RESBLOCK, dps=DEFAULT_POOLING_SIZE):
        super().__init__()
        # Construction du CNN (analyse de l'image avec introduction de couches FiLM)
        self.imgSize = img_size
        self.channels = nb_channels
        self.outSize = output_size

        # Première convolution
        self.conv1 = nn.Conv2d(self.channels, dcr, 3)
        self.pool1 = nn.AdaptiveMaxPool2d((dps, dps))
        # 128 images 14*14 

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

        # 100352

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
        x = self.activation(x)
        return self.activation_out(self.classif_out(x))


def test():
    # Définissez le chemin de votre image
    image_path = 'src/Data/Img33/img0.png'

    # Définissez les transformations à appliquer à l'image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    # Utilisez io.read_image pour lire l'image
    image = Image.open(image_path)

    # Appliquez les transformations à l'image
    x = transform(image)
    x = x.unsqueeze(0)

    # Vérifiez la taille du tenseur
    print(x.shape)

    model = CNN(180, 3, 10)
    y = model(x)

    print("je suis passé ici")
    print(y)


