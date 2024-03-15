import matplotlib.pyplot as plt 
import numpy as np
import torch
import pandas as pd
import matplotlib

from Model.Architecture.FullNetwork import FullNetwork
from DataGen.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset
from DataGen.Question import Question
from sklearn.decomposition import PCA


def FiLMGeneratorPCA(model_name):

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

    # Chargement du modèle
    datagen = DataGenerator(180, type_vocab)
    output_size = datagen.getAnswerSize()  # taille des sortie (nombre de réponses possibles)
    vocab_size = datagen.getVocabSize()  # taille du vocabulaire

    model = FullNetwork(3, output_size, vocab_size)
    model.load_state_dict(torch.load("src/Data/"+model_name+".pth", map_location={'cuda:0': 'cpu'}))
    model.eval()

    # Initialisation liste des paramètres gamma/beta
    gamma1list = []
    beta1list = []

    gamma2list = []
    beta2list = []

    gamma3list = []
    beta3list = []

    gamma4list = []
    beta4list = []

    # types de questions possibles pour notre modele
    qtypes = datagen.qtypes
    n_points = 100

    for q in qtypes :
        for n_try in range(n_points):
            
            question = Question(q)
            x = torch.tensor(datagen.getEncodedSentence(str(question))).unsqueeze(0)
            FiLM_params = model.grunet(x).view(-1, 4, 2, model.dcr)

            # gamma/beta resblock 1 
            beta1list.append(FiLM_params[:, 0, 0, :].detach().numpy().flatten())
            gamma1list.append(FiLM_params[:, 0, 1, :].detach().numpy().flatten())

            # gamma/beta resblock 2
            beta2list.append(FiLM_params[:, 1, 0, :].detach().numpy().flatten())
            gamma2list.append(FiLM_params[:, 1, 1, :].detach().numpy().flatten())

            # gamma/beta resblock 3
            beta3list.append(FiLM_params[:, 2, 0, :].detach().numpy().flatten())
            gamma3list.append(FiLM_params[:, 2, 1, :].detach().numpy().flatten())

            # gamma/beta resblock 4
            beta4list.append(FiLM_params[:, 3, 0, :].detach().numpy().flatten())
            gamma4list.append(FiLM_params[:, 3, 1, :].detach().numpy().flatten())

    gammaList = [gamma1list, gamma2list, gamma3list, gamma4list]
    betaList = [beta1list, beta2list, beta3list, beta4list]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cmap = matplotlib.colormaps.get_cmap("rainbow")

    pca = PCA(n_components=1)

    for i in range(4):

        axes[i//2, i%2].set_title(f"FiLM PCA - ResBlock {i+1}")
        axes[i//2, i%2].set_xlabel("gamma")
        axes[i//2, i%2].set_ylabel("beta")
        axes[i//2, i%2].grid()

        gamma_df = pd.DataFrame(np.array(gammaList[i]))
        beta_df = pd.DataFrame(np.array(betaList[i]))

        pca_gamma = pca.fit_transform(gamma_df)
        pca_gamma = pd.DataFrame(pca_gamma, columns=['PC1'])

        pca_beta = pca.fit_transform(beta_df)
        pca_beta = pd.DataFrame(pca_beta, columns=['PC1'])
        
        for j, q in enumerate(qtypes) : 
            clr = cmap(j / (len(qtypes) - 1))   
            axes[i//2, i%2].scatter(pca_gamma[j*n_points:(j+1)*n_points], pca_beta[j*n_points:(j+1)*n_points], color=clr, label=q)

        axes[i//2, i%2].legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.savefig(f"src/Graphs/FiLM_PCA_{model_name}.png")
    plt.show()


        
FiLMGeneratorPCA("mod_best_data3")