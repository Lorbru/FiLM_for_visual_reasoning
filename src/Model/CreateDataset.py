import numpy as np
from PIL import Image
import torch
import os
import sys
import json

from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader


class QAimgDataset(Dataset):

    def __init__(self, images, questions, answers, type, transform=None):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.transform = transform
        self.type = type

    def get_image(self, num):
        image = Image.open("src/Data/GeneratedImages/" + self.type + "/img_" + str(num) + ".png")
        return image

    def __len__(self):
        return len(self.images)  # suppose que toutes les listes ont la même longueur

    def __getitem__(self, idx):
        image = self.get_image(idx)
        image = self.transform(image)
        return image, self.questions[idx], self.answers[idx]


def CreateDataset(datagen, n_images, type):
    img_dataset = []
    quest_dataset = []
    ans_dataset = []

    # processing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


    for i in range(n_images):

        # generation d'une donnée
        quest, answer, img = datagen.buildData()
        img.saveToPNG("src/Data/GeneratedImages/" + type + "/img_" + str(i) + ".png")

        # processing (tenseur/normalisation/encodage question/batch dim)
        quest_dataset.append(torch.tensor(datagen.getEncodedSentence(str(quest))))
        ans_dataset.append(torch.tensor(datagen.getAnswerId(answer))) #.to(device))
        img_dataset.append(i) #.unsqueeze(0).to(device))

    # Padding pour une taille uniforme des phrases.
    quest_dataset = pad_sequence(quest_dataset, batch_first=True) #.unsqueeze(0).to(device)

    # Dataset
    dataset = QAimgDataset(img_dataset, quest_dataset, ans_dataset, type, transform)

    return dataset