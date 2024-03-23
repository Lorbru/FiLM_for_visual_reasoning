from .ObjectsData.LoadData import Data
from .Question import Question 
import json

class BuildVocab():

    """
    ============================================================================================
    CLASS BUILDVOCAB : class used to generate all the vocabulary and encode sentences for the 
    GRU network.
    WARNING : this class include a static method which build the entire vocabulary and save it in 
    Vocabulary/vocabulary.json file. Reusing this method will erase the previous file and change
    the way to encode our sentences. Model has to be trained again after.
    
    ATTRIBUTES :
        * vocab :dict{str: int} - the vocabulary

    METHODS : 
        * encode_sentence(sentence) - build sentence representation for the GRU network
        * vocabSize() - get vocabulary size

    STATIC METHOD :
        * createVocabulary() : create the vocabulary in vocabulary.json file. 
        Warning : it will change the way to represent our sentences for our models. 
    ============================================================================================
    """

    def __init__(self):
        """
        -- __init__() : constructor, loading the vocabulary
        """
        with open(f'src/DataGen/Vocabulary/vocabulary1.json', 'r') as f:
            self.vocab = json.load(f)

    def encode_sentence(self, sentence, check_words=False):
        """
        -- encode_sentence(sentence, check_words=False) : Encoding a sentence for the GRU network

        In >> :
            * sentence :str - sentence in french natural language
            * check_words: bool - if we check and extract known words of the sentence

        Out << :
            * list[int] - encoded sentence
        """
        sentence = sentence.replace("'", " ")
        sentence = sentence.replace("-", " ")
        words = sentence.split()
        if (check_words):
            knownWords = []
            for word in words :
                if word in self.vocab.keys():
                    knownWords.append(word)
            words = knownWords

        return [self.vocab[word] for word in words]
    
    
    def vocabSize(self):
        """
        -- vocabSize() : get vocabulary size

        Out << :
            * int - vocabulary size
        """
        return len(self.vocab) + 1 # add 1 for padding

    @staticmethod
    def createVocabulary():

        """
        -- createVocabulary() : get all the words from questions and create a vocabulary 
        for the natural language processing. Using this function will erase previous vocabulary file
        and will create a new one depending on the new questions we use.
        """

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
            for i in range(4):
                quest = str(Question(type=type, formulation=i))
                quest = quest.replace("-", " ")
                quest = quest.replace("'", " ")
                words_list = words_list + quest.split()


        ensemble = set(words_list)

        dict = {word: i+1 for i, word in enumerate(ensemble)}
        
        with open('src/DataGen/Vocabulary/vocabulary1.json', 'w') as f:
            json.dump(dict, f)
        
        return
