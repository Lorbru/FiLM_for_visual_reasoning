from .QuestionElement import QuestionElement
from .ObjectsData.LoadData import Data
import json
import numpy as np



class Question():
        
    """
    ============================================================================================
    CLASS QUESTION : class used to build question
    
    ATTRIBUTES : 
        * type :str                     - question type (comptage, presence, position, ...)
        * mainObject :QuestionElement   - first object
        * secondObject :QuestionElement - second object (if needed)
        * direction :str                - direction (if needed)
        * formulation :int              - formulation type

    METHODS : 
        * __init__(dim, type, mainObject, secondObject, direction, formulation) : constructor
        * __str__() : str of question

    STATIC METHOD :
        * buildQuestion(qtype, formulation) : build a new question depending on type and formulation
    ============================================================================================
    """

    def __init__(self, type, mainObject=None, secondObject:QuestionElement=None, direction=None, formulation=None):

        """
        -- __init__(type, mainObject=None, secondObject=None, direction=None, formulation=None) : constructor.
        The question is built depending on the parameters we set. If some are 'None', the corresponding 
        attribute is randomly selected.

        In >> :
            * type :str      - type of question
            * mainObject :QuestionElement - first object
            * secondObject :QuestionElement - second object
            * direction :str - position 
            * formulation :int - formulation type (between 0 and 2 inclusive)
        """
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
            formulation = np.random.randint(0, 4)
        self.formulation = formulation

    
    @staticmethod
    def buildQuestion(qtype, formulation=None):
        """
        -- buildQuestion(qtype :str, formulation=None :int) : build a random question depending on qtype

        In >> :
            * qtype :str - question type
            * formulation :int - formulation type (between 0 and 2 inclusive)

        Out << :
            Question - generated question
        """
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

            formulation = np.random.randint(0, 4)

            return Question(type=qtype, mainObject=mainObject, formulation=formulation)
        

    def __str__(self):
        """
        -- __str__ : str of the question 

        Out << :
            str: str of the question
        """
        
        if self.type == 'presence' : 
            if self.formulation == 0 :
                return "y a t-il " + self.mainObject.printObject(False) + " ?"
            elif self.formulation == 1 :
                return "peut-on voir " + self.mainObject.printObject(False) + " ?"
            elif self.formulation == 2 :
                return "est-ce qu'il y a " + self.mainObject.printObject(False) + " ?"
            elif self.formulation == 3 :
                return "il y a " + self.mainObject.printObject(False) + " ?"
        
        elif self.type == 'comptage' :
            if (self.formulation == 0):
                return "combien y a t il " + self.mainObject.printObject(True) + " ?"
            elif (self.formulation == 1):
                return "quel est le nombre " + self.mainObject.printObject(True) + " ?"
            elif (self.formulation == 2):
                return "combien " + self.mainObject.printObject(True) + " peut-on observer ?"
            elif (self.formulation == 3):
                return "il y a combien " + self.mainObject.printObject(True) + " ?"

        elif self.type == 'position12' or self.type == 'position33': 
            if (self.formulation == 0):
                return "quelle figure se trouve " + self.direction + " ?"
            elif (self.formulation == 1):
                return "que peut-on voir " + self.direction + " ?"
            elif (self.formulation == 2):
                return "qu'y a t-il " + self.direction + " ?"
            elif (self.formulation == 3):
                return "quelle est la figure " + self.direction + " ?"
            
        elif self.type == "couleur12" or self.type == 'couleur33' :
            if (self.formulation == 0):
                return "de quelle couleur est la figure " + self.direction + " ?"
            elif (self.formulation == 1):
                return "quelle est la couleur de la figure " + self.direction + " ?"
            elif (self.formulation == 2):
                return "la figure " + self.direction + " est de quelle couleur ?"
            elif (self.formulation == 3):
                return "donne moi la couleur de la figure " + self.direction + " ?"

    
        return None
    
    
