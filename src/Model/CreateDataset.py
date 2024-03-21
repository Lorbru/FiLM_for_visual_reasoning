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
    """
    ============================================================================================
    CLASS QAimgDataset : class used to create a pytorch dataset

    ATTRIBUTES :
        * images :list[int]   - list of images index
        * questions :list     - list of questions as tensors
        * answers :list[int]  - list of labels as tensors
        * type: str           - train or test, to save images
        * transform :torchvision.Compose  - operation to turn an image into a tensor

    METHODS :
        * __init__(images, questions, answers, type, transform) : constructor
        * get_image(num :int) : get the nth image from the saved images
        * __len__() : constructor
        * __getitem__(idx :int) : get a (image, question, answer)
    ============================================================================================
    """

    def __init__(self, images, questions, answers, type, transform=None):
        """
        -- __init__(images, questions, answers, type, transform) : constructor

        In >> :
            * images :list[int]   - list of images index
            * questions :list     - list of questions as tensors
            * answers :list[int]  - list of labels as tensors
            * type: str           - train or test, to save images
            * transform :torchvision.Compose  - operation to turn an image into a tensor
        """
        self.images = images
        self.questions = questions
        self.answers = answers
        self.transform = transform
        self.type = type

    def get_image(self, n):
        """
        -- get_image(num) : get the nth image from the saved images

        In >> :
            * n :int   - index

        Out << :
            img: An image
        """
        image = Image.open("src/Data/GeneratedImages/" + self.type + "/img_" + str(n) + ".png")
        return image

    def __len__(self):
        """
        -- __len__() : constructor

        Out << :
            int: the length of the dataset
        """
        return len(self.images)  # suppose que toutes les listes ont la même longueur

    def __getitem__(self, idx):
        """
        -- __getitem__(idx) : get a (image, question, answer)

        In >> :
            * idx :int   - index

        Out << :
            img: A (image, question, answer)
        """
        image = self.get_image(idx)
        image = self.transform(image)
        return image, self.questions[idx], self.answers[idx]


def CreateDataset(datagen, n_images, type):
    """
    -- CreateDataset(datagen, n_images, type) : create pytorch dataset

    In >> :
        * datagen: DataGenerator  - Data generator
        * n_images: int           - number of (image, question, answer) in the final dataset
        * type: str               - train or test, to save images

    Out << :
        QAimgDataset: The final dataset
    """

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
        ans_dataset.append(torch.tensor(datagen.getAnswerId(answer)))
        img_dataset.append(i)

    # Padding pour une taille uniforme des phrases.
    quest_dataset = pad_sequence(quest_dataset, batch_first=True)

    # Dataset
    dataset = QAimgDataset(img_dataset, quest_dataset, ans_dataset, type, transform)

    return dataset