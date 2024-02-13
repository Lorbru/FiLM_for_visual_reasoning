from .LoadData import Data
from .QAFactory import QAFactory
import json

class BuildVocab():

    def __init__(self):

        with open('src/DataGenerator/json/vocab.json', 'r') as f:
            self.vocab = json.load(f)

    def encode_sentence(self, sentence):
        sentence = sentence.replace("'", " ")
        sentence = sentence.replace("-", " ")
        words = sentence.split()
        return [self.vocab[word] for word in words]


    @staticmethod
    def createVocabulary():

        words_list = []

        # Type position : 
        positions = Data.PosList
        for pos in positions :
            pos = pos.replace("'", " ")
            words_list = words_list + pos.split(" ")
        
        # Type couleur
        colors = Data.ClrList
        for clr in colors :
            for key in ["M", "F", "MP", "FP"]: 
                words_list = words_list + [Data.ObjData["color"][clr][key]]

        figures = Data.FigList
        for fig in figures : 
            for key in ["singul", "plural", "indet"]:
                gn = Data.ObjData["shape"][fig][key]
                gn = gn.replace("'", " ")
                words_list = words_list + gn.split()

        for type in Data.QType :
            for i in range(3):
                quest = str(QAFactory.randomQuestion(qtype=type, formulation=i))
                quest = quest.replace("-", " ")
                quest = quest.replace("'", " ")
                words_list = words_list + quest.split()


        ensemble = set(words_list)

        dict = {word: i+1 for i, word in enumerate(ensemble)}
        
        with open('src/DataGenerator/json/vocab.json', 'w') as f:
            json.dump(dict, f)

        

        
             
            




def test():

    BuildVocab.createVocabulary()

    monVocab = BuildVocab()

    print(monVocab.encode_sentence("Combien y a t-il d'etoiles bleues ?"))



test()