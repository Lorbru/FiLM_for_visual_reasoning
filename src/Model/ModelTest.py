import torch

from DataGen.DataGenerator import DataGenerator
from Model.Architecture.FullNetwork import FullNetwork
from Model.CreateDataset import CreateDataset

def ModelTest(model_name):

    print("=========== CHECKING IF NVIDIA CUDA AVAILABLE ===========")

    if torch.cuda.is_available():
        print("  > set device on nvidia-cuda")
        device = 'cuda'
    else:
        print(f"  > nvidia cuda not available : set device on cpu\n")
        device = 'cpu'

    print("===========        SETTING PARAMETERS         ===========")

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

    print(f"  > Question/Answer data       : '{type_vocab}'")

    # instanciation générateur de données et vocabulaire
    datagen = DataGenerator(180, type_vocab)

    output_size = datagen.getAnswerSize()  # taille des sortie (nombre de réponses possibles)
    vocab_size = datagen.getVocabSize()  # taille du vocabulaire

    print(f"  > Number of test data        : {n_images_test}")
    print(f"  > Number of epochs           : {n_epochs}")
    print(f"  > Batch size                 : {batch_size}")
    print(f"  > Vocabulary size            : {vocab_size}")
    print(f"  > Number of possible answers : {output_size}\n")

    print("===========  DATA GENERATION AND PROCESSING   ===========")
    # Generation et processing des données

    # Test loader
    dataset = CreateDataset(datagen, n_images_test, 'test')
    TestLoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"  > Done\n")

    print("===========           Test  LOOP           ===========")
    # Model
    model = FullNetwork(nb_channels, output_size, vocab_size).to(device)
    model.load_state_dict(torch.load("src/Data/"+model_name+".pth", map_location={'cuda:0': 'cpu'}))
    model.eval()

    # Optimizer/Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    resTest = 0
    for (x, z, y) in TestLoader:
        inputs, questions, labels = x.to(device), z.to(device), y.to(device)
        outputs = model(inputs, questions)
        resTest += (int(outputs.argmax()) == int(labels))
    print('    Accuracy Test: ' + str(resTest / n_images_test * 100) + '%')
