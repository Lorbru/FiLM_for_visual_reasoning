import torch
import streamlit as st

from torchvision import transforms
from DataGen.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset, QAimgDataset
from DataGen.Question import Question
from DataGen.QuestionElement import QuestionElement

if torch.cuda.is_available():
    print(f"  > set device on nvidia-cuda\n")
    device = 'cuda'
else:
    print(f"  > nvidia cuda not available : set device on cpu\n")
    device = 'cpu'

# Paramètres par défaut (si non ou mal configurés) :
batch_size = 4  # taille de batch
nb_channels = 3  # nombre de channels images (3 -> RGB)
n_images_train = 10  # nombre d'images générées pour l'entrainement
n_images_test = 10  # nombre d'images générées pour le test
n_epochs = 3  # nombre d'époques pour l'entrainement
type_vocab = "data1"  # type de jeu de données choisi

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

datagen = DataGenerator(180, type_vocab)
output_size = datagen.getAnswerSize()  # taille des sortie (nombre de réponses possibles)
vocab_size = datagen.getVocabSize()  # taille du vocabulaire

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# Model
model = FullNetwork(nb_channels, output_size, vocab_size)
model.load_state_dict(torch.load("src/Data/mod_best_data3.pth", map_location=torch.device(device)))
model.eval()

########################################################################################################################

list_type = ["presence", "couleur33", "position33"]


def int_to_type(x):
    liste = ["présence", "couleur", "forme"]
    return "Deviner une " + liste[x]


qtype = st.selectbox(
    'Quel type de question ?',
    [0, 1, 2],
    index=None,
    placeholder="Choisir un type de question...",
    format_func=int_to_type)

if qtype != None:

    direction = None
    shape = '*aucun*'
    color = '*aucun*'

    if qtype == 0:

        col1, col2 = st.columns(2)

        with col1:
            color = st.radio(
                "Choix de couleur :",
                ["*aucun*", "bleu", "rouge", "vert", "blanc"],
            )

        with col2:
            shape = st.radio(
                "Choix de  forme :",
                ["*aucun*", "etoile", "triangle", "rectangle", "ellipse"],
            )

    else:
        direction = st.radio(
            "Quel emplacement ?",
            ["a droite", "a gauche", "en haut", "en bas", "au centre", "en haut a droite", "en haut a gauche",
             "en bas a droite", "en bas a gauche"],
            index=None,
            horizontal=True
        )

    if qtype == 0:

        if shape == '*aucun*' and color != '*aucun*':
            elem = QuestionElement(0, 'figure', color)
            quest = Question(list_type[qtype], mainObject=elem)

        elif shape != '*aucun*':

            if color == '*aucun*':
                color = ''

            elem = QuestionElement(0, shape, color)
            quest = Question(list_type[qtype], mainObject=elem)

    elif direction != None:
        quest = Question(list_type[qtype], direction=direction)

    if direction != None or shape != '*aucun*' or color != '*aucun*':
        def int_to_formulation(x):
            quest.formulation = x
            return str(quest)


        question = st.selectbox(
            'Votre question',
            [int_to_formulation(x) for x in [0, 1, 2]]
        )


        col1, col2 = st.columns(2)


        with col1:

            # if st.button("Nouvelle image aléatoire", type='primary') :
            #     _, _, img = datagen.buildData()


            if st.button("Calculer la réponse sur une image aléatoire", type='primary') :
                _, _, img = datagen.buildData()

                z = torch.tensor(datagen.getEncodedSentence(question)).to(device)
                x = transform(img.img)

                out = model(torch.unsqueeze(x, 0).to(device), torch.unsqueeze(z, 0).to(device))
                y = int(out.argmax())

                st.write("***Réponse*** : ", datagen.answers[y])

                with col2:
                    st.image(
                        img.img,
                        caption="Votre image"
                    )




