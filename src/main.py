import numpy as np
import torch
import sys
import json

from DataGenerator.DataGenerator import DataGenerator
from DataGenerator.QAFactory import QAFactory
from Model.Architecture.CNN import CNN
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset


def first_CNN(archi=CNN, img_size=180, input_shape=3, output_shape=16, device="cpu", n_images=500, n_epochs=10, lr=0.01, batch_size = 256):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)

    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))

    dataset = []
    question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")

    for i in range(n_images):
        
        _, answer, img = DataGen.buildImageFromQA(question)
        answer = torch.tensor([int(labelsInv[answer])])
        img = transform(img.img)
        dataset += [(img.to(device), answer.to(device))]

    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        y.append(int(labelsInv[answer]))
        
        img_transform = transform(img.img)

        X.append(img_transform)

    X = torch.stack(X)
    y = torch.tensor(y)

    # Créez un TensorDataset à partir de X et y :
    dataset = TensorDataset(X, y)

    # Enfin, créez un DataLoader à partir du TensorDataset :
    TrainLoader = DataLoader(dataset, batch_size=batch_size)

    print("====== IMAGES GENEREES ======")

    model = archi(img_size, input_shape, output_shape).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

            if not(i%(n_images/2)) :
                print("-> Semi-Epoch")

        print(f'Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        running_loss = 0.0
    return model

def main():

    print("====== CHECKING GPU ======")

    if torch.cuda.is_available():
        print("Cuda Nvidia available")
    else : 
        print("Cuda Nvidia not available. Go on CPU")

    print("====== RUNNING PROJECT ======")

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    if torch.cuda.is_available() :
        device = 'cuda'
    else :
        device = 'cpu'

    mod = first_CNN(n_epochs=10, n_images=50, output_shape=4, device=device)


    # Data Transform to tensor and normalization 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    print("====== TRAINING TERMINE ======")

    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)
    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))
    
    # Kind of question, datagenerator
    question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
    DataGen = DataGenerator()

    # Building Dataset
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
