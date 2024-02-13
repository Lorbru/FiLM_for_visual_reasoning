import numpy as np
import torch
import sys
import json

from torchvision import transforms
from DataGenerator.DataGenerator import DataGenerator
from DataGenerator.Vocab import BuildVocab
from Model.Architecture.FullNetwork import FullNetwork
from DataGenerator.LoadData import Data
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader

class QAimgDataset(Dataset):

    def __init__(self, list1, list2, list3):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3

    def __len__(self):
        return len(self.list1)  # suppose que toutes les listes ont la même longueur

    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx], self.list3[idx]



def main():

    print("===== RUNNING PROJECT =====")

    if torch.cuda.is_available() :
        print("  > set device on nvidia-cuda")
        device = 'cuda'
    else : 
        print("  > set device on cpu")
        device = 'cpu'
    
    # instanciation générateur de données et vocabulaire
    datagen = DataGenerator()
    vocab = BuildVocab()

    # Recupératon de tous les paramètres :
    batch_size = 4
    img_dim = 180      
    nb_channels = 3
    output_size = 16
    vocab_size = len(vocab.vocab) + 1 #padding
    dimension_GRU_embedding = 200
    hidden_GRU_size = 4096
    num_GRU_layers = 1

    # Nombre d'images générées pour l'entrainement / nombre d'epochs
    n_images = 10
    n_epochs = 80

    # Generation et processing des données
    print("  > Data generation and processing")
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
        quest, answer, img = datagen.buildImageFromQA()

        # processing (tenseur/normalisation/encodage question/batch dim)
        quest_dataset.append(torch.tensor(vocab.encode_sentence(str(quest))))
        ans_dataset.append(torch.tensor(Data.getAnsIndex(answer))) #.to(device))
        img_dataset.append(transform(img.img)) #.unsqueeze(0).to(device))

    # Padding pour une taille uniforme des phrases.
    quest_dataset = pad_sequence(quest_dataset, batch_first=True) #.unsqueeze(0).to(device)

    # Dataset
    dataset = QAimgDataset(img_dataset, quest_dataset, ans_dataset)

    # Train loader
    TrainLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = FullNetwork(img_dim, nb_channels, output_size, vocab_size, dimension_GRU_embedding, hidden_GRU_size, num_GRU_layers, batch_size)
    
    # Optimizer/Criterion
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1,n_epochs+1):
        print("hello")

        running_loss = 0.0
        
        for i, (x, z, y) in enumerate(TrainLoader):
            inputs, questions, labels = x.to(device), z.to(device), y.to(device)

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
    



if __name__ == "__main__":
    main()
