import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from unidecode import unidecode

from DataGen.ObjectsData.LoadData import Data
from DataGen.DataGenerator import DataGenerator
from DataGen.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset, QAimgDataset
from DataGen.Question import Question
from DataGen.QuestionElement import QuestionElement
from Analysis import FiLMGeneratorPCA
import os 
import gdown

@st.cache_resource
def loading_datagen():

    if os.path.isfile('src/Data/mod_best_data3.pth') == False :
        print(f"  > Downloading mod_best_data3.pth from https://drive.google.com")
        url = 'https://drive.google.com/file/d/13mYfOEcDRQ_yscapmz4Z3NaNL34qfTkh/view?usp=sharing'
        output = 'src/Data/mod_best_data3.pth'
        gdown.download(url, output, quiet=False, fuzzy=True)

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
    st.title("Feature-wise transformations : application for visual reasoning")
    st.write("*Alexandre Loret, Lorenzo Brucato, Sam Vallet*")     


    st.header("Introduction")

    st.write("En apprentissage multi-tache, il est très souvent naturel de vouloir interpréter nos données d'entrées (x) sous un contexte (z). Prenons l'exemple d'un modèle d'IA générative qui devrait produire des images de types différents. Si nous n'avons que quelques catégories d'images à produire, il peut être raisonnable de lancer un apprentissage sur chacune de ces catégories. Mais si nous souhaitons une plus grande diversité dans les résultats produits, nous chercherons plutôt à produire une image sous le contexte d'une tache donnée : 'Dessine-moi un chat roux avec des taches blanches ! '.")
    st.write("Nous retrouvons ce problème dans des taches très concrètes de la vie réelle, par exemple lorsque nous posons des questions sur des images.  (ce que l'on appelle des taches de raisonnement visuel). Sur une seule et même image simple et sur un vocabulaire restreint, nous pouvons déjà imaginer de très nombreuses combinaisons de questions : comme indiqué précédemment, il serait très irréaliste de vouloir entraîner notre modèle pour chaque type de question que l'on pourrait poser.")
    st.write("Les feature-wise transformations nous permettent de contourner ce problème en introduisant des couches de transformation dans notre réseau de neurones principal. Les feature-wise linear modulations (FiLM) sont un cas particulier de transformation affine de nos données. Étant donnée des features d'images $x$ et une question $z$, nous générons deux paramètres $\gamma(z)$ et " + r'$\beta(z)$' + " dépendants du contexte $z$ et appliquons la transformation suivante :")
    st.latex(r"FiLM(x) = \gamma(z).x + \beta(z)")
    st.write("Dans notre démonstration, nous avons posé des questions sur des images contenant diverses formes géométriques. Nous avons entraîné un réseau convolutif sur 500000 triplets (image, question, réponse) avec un GRU qui nous sert de générateur FiLM sur les questions posées en langage naturel (français). Nous posons des questions sur la position, la couleur ou encore la présence et le nombre de figures d'un certain type.")

    imgArch = Image.open("src/Graphs/Network.png")
    st.image(imgArch, caption='Architecture du modèle. Ethan Perez et al. (Anthropic) : https://arxiv.org/abs/1709.07871')

@st.cache_data
def getPCAGammaBeta():
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 6))

    return [fig1, fig2, fig3, fig4], [ax1, ax2, ax3, ax4], FiLMGeneratorPCA("mod_best_data3")

figures, axes, PCAdata = getPCAGammaBeta()


def results():

    st.header("Résultats")


    st.subheader("Analyse en composantes principales sur les paramètres du générateur FiLM")

    st.write("Le premier résultat qui nous intéresse particulièrement est la distribution des paramètres générés selon le type de question. Pour cela on réalise une analyse en composante principale sur les paramètres générés pour nos images :")

    numPlot = st.slider("resblock :", min_value=1, max_value=4)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.patch.set_alpha(0) 
    plt.tick_params('x', colors="gray")
    plt.tick_params('y', colors="gray")
    ax.set_title(f"PCA sur gamma et beta - Resblock {numPlot}", color="gray")
    qtypes = datagen.qtypes
    cmap = matplotlib.colormaps.get_cmap("rainbow")
    for j, q in enumerate(qtypes) : 
        clr = cmap(j / (len(qtypes) - 1))   
        ax.scatter(PCAdata.iloc[j*100:(j+1)*100, 2*(numPlot-1)], PCAdata.iloc[j*100:(j+1)*100, 2*numPlot-1], color=clr, label=q)
    ax.grid()
    ax.set_xlabel("$gamma$", color="gray")
    ax.set_ylabel("$beta$", color="gray")
    ax.legend()

    fig2, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig2.patch.set_alpha(0) 

    for j, q in enumerate(qtypes) :
        clr = cmap(j / (len(qtypes) - 1))
        subdf = PCAdata.iloc[j*100:(j+1)*100, 2*(numPlot-1)]
        axes[0].hist(subdf, weights=np.ones_like(subdf)/len(subdf), label=q, bins=30, color=clr, edgecolor='black', alpha=0.7)
        subdf = PCAdata.iloc[j*100:(j+1)*100, 2*numPlot-1]
        axes[1].hist(subdf, weights=np.ones_like(subdf)/len(subdf), label=q, bins=30, color=clr, edgecolor='black', alpha=0.7)
        
    for i in range(2):
        ax = axes[i]
        ax.set_facecolor("white")
        ax.tick_params('x', colors="gray")
        ax.tick_params('y', colors="gray")
        ax.legend()

    axes[0].set_title("gamma distribution", color="gray")
    axes[1].set_title("beta distribution", color="gray")

    PCA, Hist = st.columns([6, 12])
    with PCA:
        st.pyplot(fig)

    with Hist:
        st.pyplot(fig2)


    st.write("Les analyses PCA révèlent des clusters formés sur nos points selon chaque type de question posé. Cela permet de mettre en évidence l'apprentissage sur nos questions et l'application de transformation proches pour des questions similaires. La catégorie *presence* par exemple possède elle-même plusieurs clusters, suggérant qu'il y a également de l'apprentissage selon les éléments de question (couleur, forme, nombre)")
    

@st.cache_resource
def demonstration():
    st.header("Demonstration")

    st.subheader("Modèle entrainé :")


#=================== MISE EN PAGE ====================

write_introduction()

results()

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
prompt = unidecode(st.text_input("Posez votre question sur l'image générée (Ex : Quelle est la couleur de la figure à gauche ?)")).capitalize()

# Utiliser l'entrée de chat
if prompt:
    st.write(f"<span style='color:green'>**Question :**</span> {prompt}", unsafe_allow_html=True)
    z = torch.tensor(datagen.getEncodedSentence(prompt, True)).to(device)
    if z.shape[0] < 5 : 
        answer = "Désolé, j'ai du mal à comprendre votre question."
        st.write(f"<span style='color:red'>**Reponse :**</span> {answer}", unsafe_allow_html=True)
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

        st.write("*Zone d'attention relative à la question :*")

        x = torch.unsqueeze(x, 0).to(device)
        z = torch.unsqueeze(z, 0).to(device)
        x = model.conv1(x)
        x = model.pool1(x)

        # FiLM parameters generator using GRU network
        FiLM_params = model.grunet(z).view(-1, 4, 2, model.dcr)

        # FiLM and resblocks layers
        beta1 = FiLM_params[:, 0, 0, :]
        gamma1 = FiLM_params[:, 0, 1, :]
        x = model.resBlock1(x, beta1, gamma1)


        beta2 = FiLM_params[:, 1, 0, :]
        gamma2 = FiLM_params[:, 1, 1, :]
        x = model.resBlock2(x, beta2, gamma2)
        
        beta3 = FiLM_params[:, 2, 0, :]
        gamma3 = FiLM_params[:, 2, 1, :]
        x = model.resBlock3(x, beta3, gamma3)

        beta4 = FiLM_params[:, 3, 0, :]
        gamma4 = FiLM_params[:, 3, 1, :]
        x = model.resBlock4(x, beta4, gamma4)

        x = x.view(128, 14, 14).mean(dim=0).detach().numpy()
        
        xmax = x.max()
        xmin = x.min()
        x = Image.fromarray((x - xmin)/(xmax - xmin).astype(np.float32))

        filter = np.array(x.resize((180, 180), Image.NEAREST))
        real_img = np.array(st.session_state['img'].img, dtype=np.float32)
        for i in range(3):
            real_img[:, :, i] *= filter



        st.image(Image.fromarray(real_img.astype(np.uint8)), caption="zone d'attention avant le classifier")
        

st.write("Après le quatrième bloc résiduel appliqué sur notre image 3 (RGB) x (180 x 180) (pixels), nous récupérons une image 128 (features) x (14 x 14) (pixels). On moyenne ensuite le résultat de la sortie d'activation ReLU() et on normalise en min-max pour obtenir un filtre qui est appliqué sur l'image originale. Cela permet de mettre en évidence quelles zones de l'images sont plus mobilisées par rapport aux autres. Lorsque l'on pose une question ci-dessus, les mots inconnus du vocabulaire sont retirés pour garder la séquence de mots-clefs utilisés pour l'entrainement. Si la question posée diverge un peu trop de l'une des syntaxes ci-dessous, les réponses deviendront moins précises. Nous avons essayé de récupérer les formulations les plus courantes pour le genre de question que l'on voudrait poser. Les fautes de grammaire et de syntaxe peuvent être mal prises en compte par le modèle. Si trop peu de mots-clefs reconnus, la question n'est pas traitée.")

@st.cache_data
def questionDataframe():
    qtypes = datagen.qtypes
    df = pd.DataFrame(np.nan, index=range(len(qtypes)), columns=range(3))
    for i, qtype in enumerate(datagen.qtypes) :
        for j in range(3):
            df.loc[i, j] = str(Question.buildQuestion(qtype, formulation=j))
    df.columns = [f'formulation {i+1}' for i in range(3)]
    df.index = [qtype for qtype in qtypes]
    return df

@st.cache_data
def questionElementDataframe():
    df = pd.DataFrame(np.nan, index=range(len(Data.ClrList)), columns=range(len(Data.FigList)))
    clrlist = Data.ClrList + [""]
    for i, color in enumerate(clrlist):
        for j, figure in enumerate(Data.FigList):
            qe = QuestionElement(1, figure, color)
            df.loc[i, j] = qe.printObject(False)
    df.columns = Data.FigList
    df.index = Data.ClrList + ["NA"]
    return df

@st.cache_data
def direction33dataframe():
    df = pd.DataFrame(np.nan, index=range(3), columns=range(3))
    for i, pos in enumerate(Data.PosList33):
        df.loc[i//3, i%3] = pos
    return df

st.subheader("Formulation des questions dans notre jeu de données :")

st.write("*Questions complètes :*")
qDf = questionDataframe()
st.dataframe(qDf)

st.write("*Elements de question :*")
qeDf = questionElementDataframe()
st.dataframe(qeDf)

st.write("*Elements de position33 :*")
posDf = direction33dataframe()
st.dataframe(posDf)

