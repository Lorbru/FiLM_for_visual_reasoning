from .QuestionElement import QuestionElement
from .ObjectsData.LoadData import Data
import json
import numpy as np



class Question():
        
    def __init__(self, type, mainObject=None, secondObject:QuestionElement=None, direction=None, formulation=None):

        self.type = type


        if direction == None :
            direction = 'a droite'
        self.direction = direction
        
        if mainObject == None :
            mainObject = QuestionElement.randomElement()
        self.mainObject = mainObject

        if secondObject == None :
            secondObject = QuestionElement.randomElement()
        self.secondObject = secondObject

        if formulation == None : 
            formulation = np.random.randint(0, 3)
        self.formulation = formulation

    
    @staticmethod
    def buildQuestion(qtype, formulation=None):

        if (qtype == "position12") :

            direction = np.random.choice(["a droite", "a gauche"])

            return Question(type=qtype, direction=direction, formulation=formulation)

        elif (qtype == "position33") : 

            direction = np.random.choice(["a droite", "a gauche", "en haut", "en bas", "au centre",
                                          "en haut a droite", "en haut a gauche", "en bas a droite", "en bas a gauche"])

            return Question(type=qtype, direction=direction, formulation=formulation)
        
        elif (qtype == "couleur12"):

            direction = np.random.choice(["a droite", "a gauche"])

            return Question(type=qtype, direction=direction, formulation=formulation)

        elif (qtype == "couleur33") :

            direction = np.random.choice(["a droite", "a gauche", "en haut", "en bas", "au centre",
                                          "en haut a droite", "en haut a gauche", "en bas a droite", "en bas a gauche"])

            return Question(type=qtype, direction=direction, formulation=formulation)

        elif (qtype == "presence") : 

            mainObject = QuestionElement.randomElement()

            if (mainObject.shape == 'figure' and mainObject.color == ''):
                mainObject.color = Data.randomColor()

            return Question(type=qtype, mainObject=mainObject, formulation=formulation)
        
        elif (qtype == "comptage") :

            mainObject = QuestionElement.randomElement()

            if (mainObject.shape == 'figure' and mainObject.color == ''):
                mainObject.color = Data.randomColor()

            formulation = np.random.randint(0, 3)

            return Question(type=qtype, mainObject=mainObject, formulation=formulation)
        

    def __str__(self):
        
        if self.type == 'presence' : 
            if self.formulation == 0 :
                return "Y a t-il " + self.mainObject.printObject(False) + " ?"
            elif self.formulation == 1 :
                return "Peut-on voir " + self.mainObject.printObject(False) + " ?"
            elif self.formulation == 2 :
                return "Est-ce qu'il y a " + self.mainObject.printObject(False) + " ?"
        
        elif self.type == 'comptage' :
            if (self.formulation == 0):
                return "Combien y a t-il " + self.mainObject.printObject(True) + " ?"
            elif (self.formulation == 1):
                return "Quel est le nombre " + self.mainObject.printObject(True) + " ?"
            elif (self.formulation == 2):
                return "Combien " + self.mainObject.printObject(True) + " peut-on observer ?"

        elif self.type == 'position12' or self.type == 'position33': 
            if (self.formulation == 0):
                return "Quelle figure se trouve " + self.direction + " ?"
            elif (self.formulation == 1):
                return "Que peut-on voir " + self.direction + " ?"
            elif (self.formulation == 2):
                return "Qu'y a t-il " + self.direction + " ?"
            
        elif self.type == "couleur12" or self.type == 'couleur33' :
            if (self.formulation == 0):
                return "De quelle couleur est la figure " + self.direction + " ?"
            elif (self.formulation == 1):
                return "Quelle est la couleur de la figure " + self.direction + " ?"
            elif (self.formulation == 2):
                return "De quelle couleur est la figure " + self.direction + " ?"

    
        return None
    
    def __repr__(self):
        return self.__str__()
    
