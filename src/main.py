import numpy as np
from PIL import Image
import torch
import os
import sys
import json

from torchvision import transforms
from src.DataGenerator.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

class QAimgDataset(Dataset):

    def __init__(self, images, questions, answers, transform=None):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.transform = transform

    def get_image(self, num):
        image = Image.open("src/Data/GeneratedImages/img_" + str(num) + ".png")
        return image

    def __len__(self):
        return len(self.images)  # suppose que toutes les listes ont la même longueur

    def __getitem__(self, idx):
        image = self.get_image(idx)
        image = self.transform(image)
        return image, self.questions[idx], self.answers[idx]



def main():
    print("#########################################################")
    print("###############      RUNNING PROJECT      ###############")
    print(f"#########################################################\n")


    



    print("=========== CHECKING IF NVIDIA CUDA AVAILABLE ===========")

    if torch.cuda.is_available() :
        print("  > set device on nvidia-cuda")
        device = 'cuda'
    else : 
        print(f"  > nvidia cuda not available : set device on cpu\n")
        device = 'cpu'

    print("===========        SETTING PARAMETERS         ===========")
    
    # Paramètres par défaut (si non ou mal configurés) :
    batch_size = 4                           # taille de batch
    nb_channels = 3                          # nombre de channels images (3 -> RGB)
    n_images = 10                            # nombre d'images générées pour l'entrainement
    n_epochs = 3                             # nombre d'époques pour l'entrainement
    type_vocab = "completeQA"                # type de jeu de données choisi

    # Lecture de configuration choisie
    with open('src/config.txt', 'r') as config:
        for line in config.readlines():
            line = line.replace('\n', '=')
            wrds = line.split('=')
            if wrds[0] == "QAData":
                type_vocab = wrds[1]
            elif wrds[0] == "nData":
                n_images = int(wrds[1])
            elif wrds[0] == "nEpochs":
                n_epochs = int(wrds[1])
            elif wrds[0] == "batchSize":
                batch_size = int(wrds[1])

    print(f"  > Question/Answer data       : '{type_vocab}'")

    # instanciation générateur de données et vocabulaire
    datagen = DataGenerator(type_vocab)

    output_size = datagen.getAnswerSize()    # taille des sortie (nombre de réponses possibles)
    vocab_size = datagen.getVocabSize() + 1     # taille du vocabulaire


    print(f"  > Number of data             : {n_images}")
    print(f"  > Number of epochs           : {n_epochs}")
    print(f"  > Batch size                 : {batch_size}")
    print(f"  > Vocabulary size            : {vocab_size}")
    print(f"  > Number of possible answers : {output_size}\n")


    print("===========  DATA GENERATION AND PROCESSING   ===========")
    # Generation et processing des données
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
        img.saveToPNG("src/Data/GeneratedImages/img_" + str(i) + ".png")

        # processing (tenseur/normalisation/encodage question/batch dim)
        quest_dataset.append(torch.tensor(datagen.getEncodedSentence(str(quest))))
        ans_dataset.append(torch.tensor(datagen.getAnswerId(answer))) #.to(device))
        img_dataset.append(i) #.unsqueeze(0).to(device))

    # Padding pour une taille uniforme des phrases.
    quest_dataset = pad_sequence(quest_dataset, batch_first=True) #.unsqueeze(0).to(device)

    # Dataset
    dataset = QAimgDataset(img_dataset, quest_dataset, ans_dataset, transform)

    # Train loader
    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  > Done\n")

    print("===========           TRAINING LOOP           ===========")
    # Model
    model = FullNetwork(nb_channels, output_size, vocab_size).to(device)

    # Optimizer/Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,n_epochs+1):

        running_loss = 0.0
        
        for i, (x, z, y) in enumerate(TrainLoader):

            inputs, questions, labels = x.to(device), z.to(device), y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, questions)

            # Compute loss
            loss = criterion(outputs, labels)

            # Gradient back propagation
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Save loss 
            running_loss += loss.item()

        print(f'  > Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)}')
        if epoch%10 == 0 :
            # Saving model each 10 epochs of training
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
        running_loss = 0.0
    
    
    print(f"\n===========            END PROCESS            ===========")
    
    return 0
    



if __name__ == "__main__":
    main()
