from PIL import Image
import torch

# Transforme des images en features

transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def into_feature(image):

    return

def get_colors(image):
    """Donne toute les couleurs d'une image"""
    return 


img = Image.open("data/train/circle/circle-0.jpg")