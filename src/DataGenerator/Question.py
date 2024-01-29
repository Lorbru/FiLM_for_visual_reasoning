from QuestionElement import QuestionElement
import json

class Question():
        
    def __init__(self, type, mainObject:QuestionElement, secondObject:QuestionElement=None, direction=None):
        
        self.type = type # type de question
        # type 1 : (Présence) Y a t-il [...] sur l'image => (Oui/Non)
        # type 2 : (Comptage) Combien y a t-il [...] sur l'image => (int)
        # type 3 : (Comparaison) Y a t-il plus [...] que [...] sur l'image ? => (Oui/Non)
        # type 4 : (Direction) Quelle est la figure la plus à (droite/gauche/haut/bas) => (Ellipse, Rectangle, Etoile, Triangle)

        # [...] représente une liste de propriétés sur un objet de la scène (classe SceneObject) : on en conserve deux pour les questions de type 3 :
        
        self.direction = direction
        self.mainObject = mainObject
        self.secondObject = secondObject

    def __str__(self):
        
        if self.type == 'presence' : 
            return "Y a t-il " + self.mainObject.printObject(False) + " ?"
        
        elif self.type == 'comptage' :
            return "Combien y a t-il " + self.mainObject.printObject(True) + " ?"

        elif self.type == 'comparaison' : 
            return "Y a t-il plus " + self.mainObject.printObject(True) + " que " + self.secondObject.printObject(True) + " ?"

        elif self.type == 'position' : 
            return "Quelle figure se trouve " + self.direction + " ?"
    
        return None
    
    def __repr__(self):
        return self.__str__()