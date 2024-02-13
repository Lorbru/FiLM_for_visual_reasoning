import numpy as np
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from Model.Architecture.FullNetwork import FullNetwork
from torchvision import transforms
from Vocab import BuildVocab


def first_Full(archi=FullNetwork, img_size=180, input_shape=3, output_shape=16, vocab_size=60, embedding_dim=128, hidden_size=4096, num_layers=1, device="cpu", n_images=500, n_epochs=10, lr=0.01, batch_size = 256, model=None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])


    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)

    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))

    dataset = []
    DataGen = DataGenerator()
    Vocab = BuildVocab()

    question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
    for i in range(n_images):
        quest, answer, img = DataGen.buildImageFromQA(question)
        answer = torch.tensor([int(labelsInv[answer])])
        quest = Vocab.encode_sentence(str(quest))
        quest = torch.tensor(quest)
        img = transform(img.img)
        dataset += [(img.to(device), quest.to(device), answer.to(device))]

    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("-> Images Générées")

    if model == None:
        model = archi(img_size, input_shape, output_shape, vocab_size, embedding_dim, hidden_size, num_layers).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,n_epochs+1):
        running_loss = 0.0
        for i, (x, z, y) in enumerate(TrainLoader):
            inputs, questions, labels = x.to(device), z.to(device), y.squeeze(1).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, questions)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        if epoch%10 == 0 :
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
        running_loss = 0.0
    return model