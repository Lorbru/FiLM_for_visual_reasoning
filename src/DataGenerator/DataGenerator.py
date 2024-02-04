import random
from ImgFactory import ImgFactory
from Question import Question
from QAFactory import QAFactory
import numpy as np
from PIL import Image
from LoadData import Data

class DataGenerator():

    def __init__(self):
        
        self.imgFactory = ImgFactory()
        self.qaFactory = QAFactory()
    

    def buildImageFromQA(self, question=None):

        if question == None :
            (question, answer) = self.qaFactory.randomQuestionAndAnswer()
            qtype = question.type
        else :
            qtype = question.type
            (question, answer) = self.qaFactory.randomQuestionAndAnswer(qtype, question.direction, question.mainObject, question.secondObject)
        
        shapes = np.zeros(9, dtype=int) -1
        clrs = np.zeros(9, dtype=int) -1

        if (qtype == 'presence'):

            figure = question.mainObject.shape
            color = question.mainObject.color
        
            if (answer == "oui"):
                idalea = np.random.choice(list(range(9)), np.random.randint(1, 10), replace=False)
            else : 
                idalea = []
            
            for i in range(9):
                if i in idalea :
                    clrs[i] = Data.getColorId(color)
                    shapes[i] = Data.getFigId(figure)

                else :

                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        clrs[i] = Data.getColorId(Data.randomColor(without = [color]))
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                        shapes[i] = Data.getFigId(Data.randomFigure(without = [figure, 'figure']))
                
        elif (qtype == 'comptage'):
            figure = question.mainObject.shape
            color = question.mainObject.color
            count = int(answer)
            idalea = np.random.choice(list(range(9)), count, replace=False)
            for i in range(9):

                if i in idalea :
                    if (color != ''):
                        clrs[i] = Data.getColorId(color)
                    else : 
                        clrs[i] = Data.getColorId(Data.randomColor())
                    if (figure != 'figure'):
                        shapes[i] = Data.getFigId(figure)
                    else :
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                else :

                    # 1/2 de différer au moins par la couleur : obliger de différer par la couleur si la figure n'est pas définie
                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        clrs[i] = Data.getColorId(Data.randomColor(without = [color]))
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                    # 1/2 de différer au moins pas la forme : obliger de différer par la fforme si la couleur n'est pas définie
                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                        shapes[i] = Data.getFigId(Data.randomFigure(without = [figure, 'figure']))
        
        elif (qtype == 'position'):
            idPos = Data.getPositionId(question.direction)
            for i in range(9):
                if i == idPos:
                    shapes[i] = Data.getFigId(answer)
                else : 
                    shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))
                clrs[i] = Data.getColorId(Data.randomColor())

        # on laisse comparaison de côté pour l'instant : trop complexe
        """
        elif (qtype == 'comparaison'):

            # tirage aléatoire d'une comparaison de quantités possible : 
            sup = np.random.randint(0, 10)
            inf = np.random.randint(0, min(9 - sup, sup) + 1)

            # tirage des positions aléatoires des figrues :
            aleaid_sup_inf = np.random.choice(list(range(10)), sup + inf, replace=False)
            aleaid_inf = np.random.choice(aleaid_sup_inf, inf, replace=False)

            c2 = question.mainObject.color
            c1 = question.secondObject.color
            f2 = question.mainObject.shape
            f1 = question.secondObject.shape

            if (answer == "non"):
                c3, f3 = c1, f1
                c1, f1 = c2, f2
                c2, f2 = c1, f3

            print(aleaid_sup_inf)
            print(aleaid_inf)
            for i in range(9):
                if i in aleaid_inf :
                    if (c1 != ''):
                        clrs[i] = Data.getColorId(c1)
                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                    shapes[i] = Data.getFigId(f1)
                elif i in aleaid_sup_inf :
                    if (c2 != ''):
                        clrs[i] = Data.getColorId(c2)
                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                    shapes[i] = Data.getFigId(f2)
                else :
                    withoutFig = ['figure']
                    withoutClr = []

                    # Y a t-il plus d'étoiles ... => ne pas mettre d'autres étoiles 
                    if (c1 == ''):
                        withoutFig.append(f1)

                    # Y a t-il plus de ... que d'étoiles => ne pas mettre d'autres étoiles 
                    if (c2 == ''):
                        withoutFig.append(f2)
                    
                    # Choix de la figure aléatoire autorisée
                    aleafig = Data.randomFigure(without=withoutFig)
                    
                    # Si la figure aéatoire choisie reste malgré tout parmi les objets de comparaison, vérifier la couleur :
                    if (aleafig == f1):
                        withoutClr.append(c1)
                    if (aleafig == f2):
                        withoutClr.append(c2)

                    aleaclr = Data.randomColor(without=withoutClr)
                        
                    clrs[i] = Data.getColorId(aleaclr)
                    shapes[i] = Data.getFigId(aleafig)

        """
        return (question, answer, ImgFactory.draw33Figure(shapes, clrs, 180))
        




def test():

    Generator = DataGenerator()

    for k in range(10):
        
        q, a, img = Generator.buildImageFromQA()
        print(q)
        print(a)
        img.saveToPNG(f"src/Data/DataGen/test{k}.png")

# test()
