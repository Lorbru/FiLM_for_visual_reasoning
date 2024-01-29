import numpy as np
import json
from ImgFactory import ImgFactory
from QAFactory import QAFactory

class DataGenerator():

    def __init__(self):
        
        self.imgFactory = ImgFactory()
        self.qaFactory = QAFactory()

    @staticmethod
    def randomAnswer():
        answers = Answer()
        return np.random.choice(answers.Qlist)
    
    @staticmethod
    def buildQuestionAndImage(question=None, returnQuestion=True):

        answers = Answer()

        if question == None :
            qtype = np.random.choice(answers.Qlist)
        else :
            qtype = question.type
        alea_answer = np.random.choice(answers.QAjson[qtype])

        return 




def test():
    print(Answer().Qlist)

test()
