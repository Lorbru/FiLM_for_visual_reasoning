import numpy as np
from QuestionElement import QuestionElement

class Question():

    QUESTION_TYPES = ["presence", "comptage", "comparaison", "direction"]
    DIRECTIONS = ["en haut", "en bas", "à droite", "à gauche", "au centre", "en haut à droite", "en haut à gauche",
                  "en bas à droite", "en bas à gauche" ]
        
    def __init__(self, type, mainObject:QuestionElement, secondObject:QuestionElement=None, direction=None):

        self.type = type # type de question
        # type 1 : (Présence) Y a t-il [...] sur l'image => (Oui/Non)
        # type 2 : (Comptage) Combien y a t-il [...] sur l'image => (int)
        # type 3 : (Comparaison) Y a t-il plus [...] que [...] sur l'image ? => (Oui/Non)
        # type 4 : (Direction) Quelle est la figure la plus à (droite/gauche/haut/bas) => (Ellipse, Rectangle, Cercle)

        # [...] représente une liste de propriétés sur un objet de la scène (classe SceneObject) : on en conserve deux pour les questions de type 3 :
        self.direction = direction
        self.mainObject = mainObject
        self.secondObject = secondObject

    def __str__(self):
        
        if self.type == "presence" : 
            return "Y a t-il " + self.mainObject.printObject(False) + " ?"
        
        elif self.type == "comptage" :
            return "Combien y a t-il " + self.mainObject.printObject(True) + " ?"

        elif self.type == "comparaison" : 
            return "Y a t-il plus " + self.mainObject.printObject(True) + " que " + self.secondObject.printObject(True) + " ?"

        else : 
            return "Quelle figure se trouve " + self.direction + " ?"
    
        return None
    
    @staticmethod
    def randomQuestion():

        qtypeAlea = Question.QUESTION_TYPES[np.random.randint(0, len(Question.QUESTION_TYPES))]
        dirAlea = Question.DIRECTIONS[np.random.randint(0, len(Question.DIRECTIONS))]
        q1 = QuestionElement.randomElement()
        q2 = QuestionElement.randomElement()
        return Question(qtypeAlea, q1, q2, dirAlea)


def test():
    L = []
    for k in range(1000):
        q = Question.randomQuestion()
        print(q.__str__())
        

test()