import numpy as np
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from Model.Architecture.CNN import CNN
from torchvision import transforms


def first_CNN(archi=CNN, img_size=180, input_shape=3, output_shape=16, device="cpu", n_images=500, n_epochs=10, lr=0.01, batch_size = 256, unique=False, model=None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)

    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))

    dataset = []
    DataGen = DataGenerator()

    if unique :
        for i in range(n_images):
            answer, img = DataGen.buildUniqueImageFromFigure()
            answer = torch.tensor([int(labelsInv[answer])])
            img = transform(img.img)
            dataset += [(img.to(device), answer.to(device))]
    else :
        question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
        for i in range(n_images):
            _, answer, img = DataGen.buildImageFromQA(question)
            answer = torch.tensor([int(labelsInv[answer])])
            img = transform(img.img)
            dataset += [(img.to(device), answer.to(device))]

    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("-> Images Générées")

    if model == None:
        model = archi(img_size, input_shape, output_shape).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,n_epochs+1):
        running_loss = 0.0
        for i, (x, y) in enumerate(TrainLoader):
            inputs, labels = x.to(device), y.squeeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        if epoch%10 == 0 :
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
        running_loss = 0.0
    return model