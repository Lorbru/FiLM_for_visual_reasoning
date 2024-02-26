import numpy as np
from PIL import Image
import torch
import os
import sys
import json

from torchvision import transforms
from src.DataGenerator.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset
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
    n_images_train = 10                      # nombre d'images générées pour l'entrainement
    n_images_test = 10                       # nombre d'images générées pour le test
    n_epochs = 3                             # nombre d'époques pour l'entrainement
    type_vocab = "completeQA"                # type de jeu de données choisi

    # Lecture de configuration choisie
    with open('src/config.txt', 'r') as config:
        for line in config.readlines():
            line = line.replace('\n', '=')
            wrds = line.split('=')
            if wrds[0] == "QAData":
                type_vocab = wrds[1]
            elif wrds[0] == "nDataTrain":
                n_images_train = int(wrds[1])
            elif wrds[0] == "nDataTest":
                n_images_test = int(wrds[1])
            elif wrds[0] == "nEpochs":
                n_epochs = int(wrds[1])
            elif wrds[0] == "batchSize":
                batch_size = int(wrds[1])

    print(f"  > Question/Answer data       : '{type_vocab}'")

    # instanciation générateur de données et vocabulaire
    datagen = DataGenerator(type_vocab)

    output_size = datagen.getAnswerSize()    # taille des sortie (nombre de réponses possibles)
    vocab_size = datagen.getVocabSize() + 1     # taille du vocabulaire


    print(f"  > Number of data             : {n_images_train}")
    print(f"  > Number of epochs           : {n_epochs}")
    print(f"  > Batch size                 : {batch_size}")
    print(f"  > Vocabulary size            : {vocab_size}")
    print(f"  > Number of possible answers : {output_size}\n")


    print("===========  DATA GENERATION AND PROCESSING   ===========")
    # Generation et processing des données


    # Train loader
    dataset = CreateDataset(datagen, n_images_train, 'train')
    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Test loader
    dataset = CreateDataset(datagen, n_images_test, 'test')
    TestLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

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

        print(f'  > Epoch {epoch} / {n_epochs} | Loss: {running_loss / len(TrainLoader)} ', end='')

        # Bloc test
        res = 0
        for (x, z, y) in TestLoader:
            inputs, questions, labels = x.to(device), z.to(device), y.to(device)
            outputs = model(inputs, questions)
            res += (int(outputs.argmax()) == int(labels))
        print('| Accuracy: ' + str(res / n_images_test * 100) + '%')

        if epoch%10 == 0 :
            # Saving model each 10 epochs of training
            torch.save(model.state_dict(), "src/Data/mod"+str(epoch)+".pth")
    
    print(f"\n===========            END PROCESS            ===========")
    
    return 0
    



if __name__ == "__main__":
    main()
