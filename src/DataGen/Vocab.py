from .ObjectsData.LoadData import Data
from .Question import Question 
import json

class BuildVocab():

    def __init__(self):

        with open(f'src/DataGen/Vocabulary/vocabulary.json', 'r') as f:
            self.vocab = json.load(f)

    def encode_sentence(self, sentence):
        sentence = sentence.replace("'", " ")
        sentence = sentence.replace("-", " ")
        words = sentence.split()
        return [self.vocab[word] for word in words]
    
    def vocabSize(self):
        return len(self.vocab)


    @staticmethod
    def createVocabulary():

        words_list = []

        # Type position : 
        positions = Data.PosList33
        for pos in positions :
            pos = pos.replace("'", " ")
            words_list = words_list + pos.split(" ")

        positions = Data.PosList12
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

        for type in Data.Qlist :
            for i in range(3):
                quest = str(Question(type=type, formulation=i))
                quest = quest.replace("-", " ")
                quest = quest.replace("'", " ")
                words_list = words_list + quest.split()


        ensemble = set(words_list)

        dict = {word: i+1 for i, word in enumerate(ensemble)}
        
        with open('src/DataGen/Vocabulary/vocabulary.json', 'w') as f:
            json.dump(dict, f)
        
        return
