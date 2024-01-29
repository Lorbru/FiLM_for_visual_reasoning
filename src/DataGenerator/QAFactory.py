import numpy as np
import json
from QuestionElement import QuestionElement
from Question import Question

class QAFactory():
        
    with open('src/DataGenerator/json/QA.json', 'r') as f:
        QAjson = json.load(f)
        Qlist = [q for q in QAjson]

    Dlist = ["en haut", "en bas", "à droite", "à gauche", "au centre", "en haut à droite", "en haut à gauche",
                "en bas à droite", "en bas à gauche" ]
    
    def __init__(self):
        return

    @staticmethod
    def randomQuestion(qtype=None, dirAlea=None, q1=None, q2=None):

        if qtype == None :
            qtype = np.random.choice(QAFactory.Qlist)
        if dirAlea == None :
            dirAlea = QAFactory.Dlist[np.random.randint(0, len(QAFactory.Dlist))]
        if q1 == None :
            q1 = QuestionElement.randomElement()
        if q2 == None :
            q2 = QuestionElement.randomElement()
        return Question(qtype, q1, q2, dirAlea)
    
    @staticmethod
    def randomQuestionAndAnswer(qtype=None):

        if (qtype==None):      
            qtype = np.random.choice(QAFactory.Qlist)

        answer = np.random.choice(QAFactory.QAjson[qtype])

        return (QAFactory.randomQuestion(qtype=qtype), answer)





def test():
    L = []
    for k in range(100):
        q = QAFactory.randomQuestionAndAnswer()
        print(q)
        

test()