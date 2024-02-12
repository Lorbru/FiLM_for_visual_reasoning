import numpy as np
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from Model.Architecture.GRUNet import GRUNet
from torchvision import transforms


def first_GRU(archi=GRUNet, input_shape=1, hidden_dim=512, output_shape=4, n_layer=8, device="cpu", n_epochs=10, lr=0.01, batch_size=256, unique=False, model=None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    dataset = []
    
    DataGen = DataGenerator()

    # TODO Train Loader

    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("-> Questions Générées")

    if model == None:
        input_shape = next(iter(TrainLoader))[0].shape[2]
        model = archi(input_shape, hidden_dim, output_shape, n_layer).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,n_epochs+1):
        running_loss = 0.0
        h = model.init_hidden(batch_size)
        for i, (x, y) in enumerate(TrainLoader):
            inputs, labels = x.to(device), y.squeeze(1).to(device)
            h = h.data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, h = model(inputs, h)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        if epoch%10 == 0 :
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
        running_loss = 0.0

    return model
