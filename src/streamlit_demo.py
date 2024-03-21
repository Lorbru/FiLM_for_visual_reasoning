import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from DataGen.ObjectsData.LoadData import Data
from DataGen.DataGenerator import DataGenerator
from DataGen.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset, QAimgDataset
from DataGen.Question import Question
from DataGen.QuestionElement import QuestionElement
from Analysis import FiLMGeneratorPCA

@st.cache_resource
def loading_datagen():

    return DataGenerator(180, "data3")

datagen = loading_datagen()

@st.cache_resource
def loading_fullNetwork():
    
    if torch.cuda.is_available():
        print(f"  > set device on nvidia-cuda\n")
        device = 'cuda'
    else:
        print(f"  > nvidia cuda not available : set device on cpu\n")
        device = 'cpu'
    
    output_size = datagen.getAnswerSize()  # taille des sorties (nombre de réponses possibles)
    vocab_size = datagen.getVocabSize()  # taille du vocabulaire

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    # Model
    model = FullNetwork(3, output_size, vocab_size)
    model.load_state_dict(torch.load("src/Data/mod_best_data3.pth", map_location=torch.device(device)))
    model.eval()
    return device, transform, model

device, transform, model = loading_fullNetwork()

@st.cache_data
def write_introduction():
    st.title("Feature-wise transformations : application for visual reasonning")

    st.subheader("Introduction")

    st.write("En apprentissage multi-tache, il est très souvent naturel de vouloir interpréter nos données d'entrées (x) sous un contexte (z). Prenons l'exemple d'un modèle d'IA générative qui devrait produire des images de types différents. Si nous n'avons que quelques catégories d'images à produire, il peut être raisonnable de lancer un apprentissage sur chacune de ces catégories. Mais si nous souhaitons une plus grande diversité dans les résultats produits, nous chercherons plutôt à produire une image sous le contexte d'une tache donnée : 'Dessine-moi un chat roux avec des taches blanches ! '.")
    st.write("Nous retrouvons ce problème dans des taches très concrètes de la vie réelle, par exemple lorsque nous posons des questions sur des images.  (ce que l'on appelle des taches de raisonnement visuel). Sur une seule et même image simple et sur un vocabulaire restreint, nous pouvons déjà imaginer de très nombreuses combinaisons de questions : comme indiqué précédemment, il serait très irréaliste de vouloir entraîner notre modèle pour chaque type de question que l'on pourrait poser.")
    st.write("Les feature-wise transformations nous permettent de contourner ce problème en introduisant des couches de transformation dans notre réseau de neurones principal. Les feature-wise linear modulations (FiLM) sont un cas particulier de transformation affine de nos données. Étant donnée des features d'images $x$ et une question $z$, nous générons deux paramètres $\gamma(z)$ et " + r'$\beta(z)$' + " dépendants du contexte $z$ et appliquons la transformation suivante :")
    st.latex(r"FiLM(x) = \gamma(z).x + \beta(z)")
    st.write("Dans notre démonstration, nous avons posé des questions sur des images contenant diverses formes géométriques. Nous avons entraîné un réseau convolutif sur 500000 triplets (image, question, réponse) avec un GRU qui nous sert de générateur FiLM sur les questions posées en langage naturel (français). Nous posons des questions sur la position, la couleur ou encore la présence et le nombre de figures d'un certain type.")
    st.write("L'architecture utilisée ci-dessous est validée par Ethan Perez (Anthropic) : https://arxiv.org/abs/1709.07871")

    imgArch = Image.open("src/Graphs/Network.png")
    st.image(imgArch, caption='Architecture du modèle')

@st.cache_data
def getPCAGammaBeta():
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))

    return [fig1, fig2, fig3, fig4], [ax1, ax2, ax3, ax4], FiLMGeneratorPCA("mod_best_data3")

figures, axes, PCAdata = getPCAGammaBeta()


def results():

    st.subheader("Resultats")

    st.write("Le premier résultat qui nous intéresse particulièrement est la distribution des paramètres générés selon le type de question. Pour cela on réalise une analyse en composante principale sur les paramètres générés pour nos images :")

    numPlot = st.slider("resblock :", min_value=1, max_value=4)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.patch.set_alpha(0) 
    plt.tick_params('x', colors='white')
    plt.tick_params('y', colors='white')
    ax.set_title(f"PCA sur gamma et beta - Resblock {numPlot}", color='white')
    qtypes = datagen.qtypes
    cmap = matplotlib.colormaps.get_cmap("rainbow")
    for j, q in enumerate(qtypes) : 
        clr = cmap(j / (len(qtypes) - 1))   
        ax.scatter(PCAdata.iloc[j*100:(j+1)*100, 2*(numPlot-1)], PCAdata.iloc[j*100:(j+1)*100, 2*numPlot-1], color=clr, label=q)
    ax.grid()
    ax.set_xlabel("$gamma$", color='white')
    ax.set_ylabel("$beta$", color='white')
    ax.set_facecolor("black")
    ax.legend()

    _, _, center, _, _ = st.columns([1, 1, 6, 1, 1])
    with center:
        st.pyplot(fig, )

@st.cache_resource
def resultats_analyse():
    st.write("Les analyses PCA révèlent des clusters formés sur nos points selon chaque type de question posé. Cela permet de mettre en évidence l'apprentissage sur nos questions et l'application de transformation proches pour des questions similaires.")
    st.write("Nous pouvons également visualiser les zones d'attention provoquée par les couches FiLM :")


@st.cache_resource
def demonstration():
    st.subheader("Demonstration")


#=================== MISE EN PAGE ====================

write_introduction()

results()

resultats_analyse()

demonstration()


@st.cache_resource
def getImage():
    q, a, img = datagen.buildData()
    return img

if 'img' not in st.session_state:
    q, a, img = datagen.buildData()
    st.session_state['img'] = img

if st.button("Nouvelle image aléatoire"):
    q, a, img = datagen.buildData()
    st.session_state['img'] = img

if st.session_state['img'] is not None :
    st.image(st.session_state['img'].img)

# Créer une entrée de chat
prompt = st.text_input("Posez votre question sur l'image générée (Ex : Quelle est la couleur de la figure à gauche ?)")

# Utiliser l'entrée de chat
if prompt:
    st.write(f"<span style='color:green'>**Question :**</span> {prompt}", unsafe_allow_html=True)
    z = torch.tensor(datagen.getEncodedSentence(prompt, True)).to(device)
    if z.shape[0] < 5 : 
        answer = "Désolé, j'ai du mal à comprendre votre question."
    else :
        x = transform(st.session_state['img'].img)
        out = model(torch.unsqueeze(x, 0).to(device), torch.unsqueeze(z, 0).to(device))
        word_y = datagen.answers[int(out.argmax())]
        if word_y in datagen.relation["questions"]["couleur33"]:
            acc = Data.ObjData["color"][word_y]["F"]
            answer = f"Cette figure est de couleur {acc} !"
        elif word_y in datagen.relation["questions"]["position33"]:
            if Data.ObjData["shape"][word_y]["genre"] == "M" :
                answer = f"Il s'agit d'un {word_y} !"
            else :
                answer = f"Il s'agit d'une {word_y} !"
        elif word_y in datagen.relation["questions"]["presence"] :
            if word_y == "oui" :
                answer = f"Oui, il y en a."
            else :
                answer = f"Non il n'y en a pas."


    st.write(f"<span style='color:red'>**Reponse :**</span> {answer}", unsafe_allow_html=True)

