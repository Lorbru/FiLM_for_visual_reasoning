import numpy as np
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from Model.Architecture.CNN import CNN
from torchvision import transforms


def first_CNN(archi=CNN, img_size=180, input_shape=3, output_shape=16, device="cpu", n_images=500, n_epochs=10, lr=0.01, batch_size = 256, unique=False):

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

    model = archi(img_size, input_shape, output_shape).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
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
        if (epoch-1)%10 == 0 :
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
        running_loss = 0.0
    return model

def main():

    print("====== CHECKING GPU ======")

    if torch.cuda.is_available():
        print("Cuda Nvidia available",torch.cuda.get_device_name(0))
        device = 'cuda'
    else :
        print("Cuda Nvidia not available. Go on CPU")
        device = 'cpu'

    print("====== RUNNING PROJECT ======")

    unique = True
    mod = first_CNN(n_epochs=50, n_images=1000, output_shape=4, device=device, unique=unique, lr=0.00001)
    # mod = CNN(180, 3, 4).to(device)
    # mod.load_state_dict(torch.load("src/Data/mod31.pth"))
    # mod.eval()

    print("====== RUNNING TESTS ======")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)

    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))

    DataGen = DataGenerator()
    if unique :
        for i in range(10):
            answer, img = DataGen.buildUniqueImageFromFigure()
            print(answer, int(labelsInv[answer]))
            img = transform(img.img)
            img = img.unsqueeze(0).to(device)
            output = mod(img)
            print(output.argmax())
    else :
        question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
        for i in range(10):
            _, answer, img = DataGen.buildImageFromQA(question)
            print(answer, int(labelsInv[answer]))
            img = transform(img.img)
            img = img.unsqueeze(0).to(device)
            output = mod(img)
            print(output.argmax())

    torch.save(mod.state_dict(), "src/Data/mod_final.pth")

    print("======       END      =======")
    return 



if __name__ == "__main__":
    main()
