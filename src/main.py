import numpy as np
import torch
import sys
import json

from DataGenerator import DataGenerator
from QAFactory import QAFactory
from torchvision import transforms

from Script.Train_CNN import first_CNN

def main():

    print("====== CHECKING GPU ======")

    if torch.cuda.is_available():
        print("Cuda Nvidia available",torch.cuda.get_device_name(0))
        device = 'cuda'
    else :
        print("Cuda Nvidia not available. Go on CPU")
        device = 'cpu'

    print("====== RUNNING PROJECT ======")

    unique = False
    mod = first_CNN(n_epochs=10, n_images=5, output_shape=4, device=device, unique=unique, lr=0.00001)

    # mod = CNN(180, 3, 4).to(device)
    # mod.load_state_dict(torch.load("src/Data/mod31.pth"))
    # mod.eval()
    # mod = first_CNN(n_epochs=50, n_images=5000, output_shape=4, device=device, unique=unique, lr=0.00001, model=mod)

    print("====== RUNNING TESTS ======")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    with open('src/DataGenerator/json/LabelsMaps.json', 'r') as f:
        labelsMaps = json.load(f)

    labelsInv = dict(zip(labelsMaps.values(), labelsMaps.keys()))

    DataGen = DataGenerator()
    n_tests = 10
    if unique :
        sum = 0
        for i in range(n_tests):
            answer, img = DataGen.buildUniqueImageFromFigure()
            img = transform(img.img)
            img = img.unsqueeze(0).to(device)
            output = mod(img)
            sum += int(labelsInv[answer]) == int(output.argmax())
        print("Accuracy : "+str(sum/n_tests*100)+"%")
    else :
        sum = 0
        question = QAFactory.randomQuestion(qtype="position", dirAlea="au centre")
        for i in range(n_tests):
            _, answer, img = DataGen.buildImageFromQA(question)
            img = transform(img.img)
            img = img.unsqueeze(0).to(device)
            output = mod(img)
            sum += int(labelsInv[answer]) == int(output.argmax())
        print("Accuracy : " + str(sum / n_tests * 100) + "%")

    print("======       END      =======")
    return 



if __name__ == "__main__":
    main()
