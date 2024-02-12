from QuestionElement import QuestionElement
import json
import numpy as np
from LoadData import Data


class Question():
        
    def __init__(self, type, mainObject:QuestionElement, secondObject:QuestionElement=None, direction=None, formulation=None):
        
        self.type = type # type de question
        # type 1 : (Présence) Y a t-il [...] sur l'image => (Oui/Non)
        # type 2 : (Comptage) Combien y a t-il [...] sur l'image => (int)
        # type 3 : (Comparaison) Y a t-il plus [...] que [...] sur l'image ? => (Oui/Non)
        # type 4 : (Position) Quelle est la figure la plus à (droite/gauche/haut/bas) => (Ellipse, Rectangle, Etoile, Triangle)

        # [...] représente une liste de propriétés sur un objet de la scène (classe SceneObject) : on en conserve deux pour les questions de type 3 :
        
        self.direction = direction
        self.mainObject = mainObject
        self.secondObject = secondObject
        if (formulation==None or not(formulation in [0, 1, 2])) : 
            formulation = np.random.randint(0, 3)
        self.formulation = formulation

        # On transforme ici les questions non pertinentes ("interdites") : 
        # - Combien de figures ? (toujours 9)
        # - Y a t-il 9 figures ? (touours oui)
        # - Y a t-il 3 figures ? (toujours non)
        # - Y a t-il plus de figures que de figures/Y a t-il plus d'étoiles que d'étoiles rouges ?/Y a t-il plus d'étoiles rouges que d'étoiles ?

        # regle sur les questions : 
        # * presence + comptage : 
        #       - la couleur d'une figure quelconque est touours précisée
        # * comparaison :
        #       - la figure d'un élément est toujours précise (étoile, ellipse, rectangle ou triangle)
        #       - si les deux éléments désignent la même figure, les deux couleurs sont obligatoirement précisées (et différentes)
        #       - si les deux éléments désignent des figures différentes, couleurs précisées ou non...
        
        if (self.type == 'comptage' or self.type == 'presence'):
            if (self.mainObject.shape == 'figure' and self.mainObject.color == ''):
                self.mainObject.color = Data.randomColor()
        if (self.type == 'comparaison'):
            if (self.mainObject.shape == 'figure'):
                self.mainObject.shape = Data.randomFigure(without=['figure'])
            if (self.secondObject.shape == 'figure'):
                self.secondObject.shape = Data.randomFigure(without=['figure'])
            if (self.mainObject.shape == self.secondObject.shape):
                if (self.mainObject.color == ''):
                    self.mainObject.color = Data.randomColor()
                if (self.secondObject.color == ''):
                    self.secondObject.color = Data.randomColor()
                if (self.mainObject.color == self.secondObject.color):
                    self.secondObject.color = Data.randomColor(without=[self.mainObject.color])

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

        elif self.type == 'comparaison' : 
            return "Y a t-il plus " + self.mainObject.printObject(True) + " que " + self.secondObject.printObject(True) + " ?"

        elif self.type == 'position' : 
            if (self.formulation == 0):
                return "Quelle figure se trouve " + self.direction + " ?"
            elif (self.formulation == 1):
                return "Que peut-on voir " + self.direction + " ?"
            elif (self.formulation == 2):
                return "Qu'y a t-il " + self.direction + " ?"
    
        return None
    
    def __repr__(self):
        return self.__str__()
    
