import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from Model.Architecture.CNN import CNN
from torchvision import transforms


def first_CNN(archi=CNN, img_size=180, input_shape=3, output_shape=16, device="cpu", n_epochs=10, lr=0.01):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    batch_size = 5
    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labels = json.load(f)

    labelsInv = dict(zip(labels.values(), labels.keys()))

    TrainLoader = []
    question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
    for i in range(20):
        DataGen = DataGenerator()
        _, answer, img = DataGen.buildImageFromQA(question)
        print(answer,int(labelsInv[answer]))
        answer = torch.tensor([int(labelsInv[answer])])
        img.saveToPNG("src/Data/DataGen/"+str(i)+".png")
        img = transform(img.img)
        img = img.unsqueeze(0)
        TrainLoader += [(img, answer)]

    model = archi(img_size, input_shape, output_shape)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(TrainLoader):
            inputs, labels = x.to(device), y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        running_loss = 0.0
    return model

first_CNN()

def main():

    print("====== RUNNING PROJECT ======")

    






    print("======       END      =======")
    return 



if __name__ == "__main__":
    main()
