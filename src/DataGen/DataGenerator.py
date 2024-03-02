from .ImgFactory import ImgFactory
import numpy as np
from .Question import Question
from .Vocab import BuildVocab
import json

class DataGenerator():

    def __init__(self, dim, data_type):

        self.type = data_type

        with open(f"src/DataGen/IQArelations/{data_type}.json", 'r') as f :
            self.relation = json.load(f)

        self.imgType = self.relation["images"]["type"]
        self.gradient = bool(self.relation["images"]["gradient"])
        self.noise = bool(self.relation["images"]["noise"])

        self.qtypes =  [qtype for qtype in self.relation["questions"]]
        self.answers = self.relation["answers"]

        self.imgFactory = ImgFactory(dim, self.imgType, self.gradient, self.noise)
        self.vocab = BuildVocab()

    def buildData(self):

        qtype = np.random.choice(self.qtypes)
        answer = np.random.choice(self.relation["questions"][qtype])
        question = Question.buildQuestion(qtype)
        img = self.imgFactory.buildData(question, answer)

        return (question, answer, img)
        
    def getAnswerSize(self):
        return len(self.answers)
    
    def getAnswerId(self, answer):
        return self.answers.index(answer)
    
    def getVocabSize(self):
        return len(self.vocab.vocab) + 1
    
    def getEncodedSentence(self, sentence):
        return self.vocab.encode_sentence(sentence)
    
