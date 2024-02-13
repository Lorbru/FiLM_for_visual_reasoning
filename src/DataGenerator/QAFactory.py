import numpy as np
import json
from .QuestionElement import QuestionElement
from .Question import Question
from .LoadData import Data

class QAFactory():

    @staticmethod
    def randomQuestion(qtype=None, dirAlea=None, q1=None, q2=None, formulation=None):

        if qtype == None :
            qtype = Data.randomQType()
        if dirAlea == None :
            dirAlea = Data.randomPosition()
        if q1 == None :
            q1 = QuestionElement.randomElement()
        if q2 == None :
            q2 = QuestionElement.randomElement()
        return Question(qtype, q1, q2, dirAlea, formulation)
    
    @staticmethod
    def randomQuestionAndAnswer(qtype=None, dirAlea=None, q1=None, q2=None):

        if (qtype==None):      
            # on laisse comparaison de côté pour le moment
            qtype = Data.randomQType(without=['comparaison'])
        # print(f"-------------- {qtype}")
        answer = np.random.choice(Data.QAjson[qtype])
        question = QAFactory.randomQuestion(qtype=qtype, dirAlea=dirAlea, q1=q1, q2=q2)

        # NB : on a traité dans la génération de question le problème d'incompatibilité entre
        # questions triviales/interdites et certaines réponses

        return (question, answer)

