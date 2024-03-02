from .ObjectsData.LoadData import Data
import numpy as np
from .Question import Question
from .Shapes import Shapes

class ImgFactory():

    def __init__(self, dim, imgType, gradient, noise):
        self.dim = dim
        self.type = imgType
        self.gradient = gradient
        self.noise = noise

    def buildData(self, question, answer):

        if (self.type == "33"):
            return self.dataImg33(question, answer)
        elif (self.type == "12"):
            return self.dataImg12(question, answer)
        
    def dataImg33(self, question, answer):
        
        
        shapes = []
        colors = []
        qtype = question.type

        if (qtype == 'presence'):

            figure = question.mainObject.shape
            color = question.mainObject.color
        
            if (answer == "oui"):
                idalea = np.random.choice(list(range(9)), np.random.randint(1, 10), replace=False)
            else : 
                idalea = []
            
            for i in range(9):
                if i in idalea :
                    if (color != ''):
                        colors.append(color)
                    else :
                        colors.append(Data.randomColor())
                    if (figure == 'figure'):
                        figure = Data.randomFigure(without=['figure'])
                    shapes.append(figure)

                else :

                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        colors.append(Data.randomColor(without = [color]))
                        shapes.append(Data.randomFigure(without = ['figure']))

                    else :
                        colors.append(Data.randomColor())
                        shapes.append(Data.randomFigure(without = [figure, 'figure']))
                
        elif (qtype == 'comptage'):
            
            figure = question.mainObject.shape
            color = question.mainObject.color
            count = int(answer)
            idalea = np.random.choice(list(range(9)), count, replace=False)
            for i in range(9):

                if i in idalea :
                    if (color != ''):
                        colors.append(color)
                    else : 
                        colors.append(Data.randomColor())
                    if (figure != 'figure'):
                        shapes.append(figure)
                    else :
                        shapes.append(Data.randomFigure(without = ['figure']))

                else :

                    # 1/2 de différer au moins par la couleur : obliger de différer par la couleur si la figure n'est pas définie
                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        colors.append(Data.randomColor(without = [color]))
                        shapes.append(Data.randomFigure(without = ['figure']))

                    # 1/2 de différer au moins pas la forme : obliger de différer par la fforme si la couleur n'est pas définie
                    else :
                        colors.append(Data.randomColor())
                        shapes.append(Data.randomFigure(without = [figure, 'figure']))
        
        elif (qtype == 'position33'):
            idPos = Data.getPosition33Id(question.direction)
            for i in range(9):
                if i == idPos:
                    shapes.append(answer)
                else : 
                    shapes.append(Data.randomFigure(without = ['figure']))
                colors.append(Data.randomColor())

        elif (qtype == 'couleur33'):
            idPos = Data.getPosition33Id(question.direction)
            for i in range(9):
                if i == idPos:
                    colors.append(answer)
                else : 
                    colors.append(Data.randomColor())
                shapes.append(Data.randomFigure(without = ['figure']))

        return self.buildImg(colors, shapes)

    def dataImg12(self, question, answer):
        
        shapes = []
        colors = []
        qtype = question.type

        
        if (qtype == 'presence'):

            figure = question.mainObject.shape
            color = question.mainObject.color
        
            if (answer == "oui"):
                idalea = np.random.choice(list(range(2)), np.random.randint(1, 3), replace=False)
            else : 
                idalea = []
            
            for i in range(2):
                if i in idalea :
                    if (color != ''):
                        colors.append(color)
                    else :
                        colors.append(Data.randomColor())
                    if (figure == 'figure'):
                        figure = Data.randomFigure(without=['figure'])
                    shapes.append(figure)

                else :

                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        colors.append(Data.randomColor(without = [color]))
                        shapes.append(Data.randomFigure(without = ['figure']))

                    else :
                        colors.append(Data.randomColor())
                        shapes.append(Data.randomFigure(without = [figure, 'figure']))
                
        elif (qtype == 'comptage'):
            
            figure = question.mainObject.shape
            color = question.mainObject.color
            count = int(answer)
            idalea = np.random.choice(list(range(2)), count, replace=False)
            for i in range(2):

                if i in idalea :
                    if (color != ''):
                        colors.append(color)
                    else : 
                        colors.append(Data.randomColor())
                    if (figure != 'figure'):
                        shapes.append(figure)
                    else :
                        shapes.append(Data.randomFigure(without = ['figure']))

                else :

                    # 1/2 de différer au moins par la couleur : obliger de différer par la couleur si la figure n'est pas définie
                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        colors.append(Data.randomColor(without = [color]))
                        shapes.append(Data.randomFigure(without = ['figure']))

                    # 1/2 de différer au moins pas la forme : obliger de différer par la fforme si la couleur n'est pas définie
                    else :
                        colors.append(Data.randomColor())
                        shapes.append(Data.randomFigure(without = [figure, 'figure']))
        
        elif (qtype == 'position12'):
            idPos = Data.getPosition12Id(question.direction)
            for i in range(2):
                if i == idPos:
                    shapes.append(answer)
                else : 
                    shapes.append(Data.randomFigure(without = ['figure']))
                colors.append(Data.randomColor())

        elif (qtype == 'couleur12'):
            idPos = Data.getPosition12Id(question.direction)
            for i in range(2):
                if i == idPos:
                    colors.append(answer)
                else : 
                    colors.append(Data.randomColor())
                shapes.append(Data.randomFigure(without = ['figure']))

        return self.buildImg(colors, shapes)

    
    
    def buildImg(self, colors, figures):

        img = Shapes(self.dim)

        if (self.gradient):
            img.randomGradient()

        if (self.type == "12"):

            for i in range(2):
                
                color = Data.getRGB(colors[i])
                fig = Data.getFigure(figures[i])
                unit = self.dim//2
                pos = np.array([(i + .25)*unit, 3*unit//4])
                dx, dy = np.array([unit/2, unit/2])
                rot = 360
                
                # bruit aléatoire
                if (self.noise) :
                    pos = pos + (unit/2) * .25 * (np.random.rand(2) - np.array([.5, .5]))
                    dx, dy = np.array([dx, dy]) + (unit/2) * .25 * (np.random.rand(2) - np.array([.5, .5]))
                    color = color + np.random.randint(-min(20, min(color)), min(20, 255-max(color)), 3)
                    rot *= np.random.rand()

                color = tuple(color)
                x, y = pos

                if (fig == "ellipse"):
                    img.drawEllipse(x, y, .75*dx, .75*dy, rot, color)
                elif(fig == "rectangle"):
                    img.drawRect(x, y, .75*dx, .75*dy, rot, color)
                elif(fig == "etoile"):
                    img.drawStar(x, y, dx, dy, 5, rot, color, self.noise)
                elif(fig == "triangle"):
                    img.drawTriangle(x, y, dx, dy, rotation=rot, random_noise=self.noise, clr=color)

        if (self.type == "33"):

            for i in range(3):
                for j in range(3):
                
                    color = Data.getRGB(colors[3*j + i])
                    fig = Data.getFigure(figures[3*j + i])
                    unit = self.dim//3
                    pos = np.array([(i + .25)*unit, (j + .25)*unit])
                    dx, dy = np.array([unit/2, unit/2])
                    rot = 360
                    
                    # bruit aléatoire
                    if (self.noise) :
                        pos = pos + (unit/2) * .25 * (np.random.rand(2) - np.array([.5, .5]))
                        dx, dy = np.array([dx, dy]) + (unit/2) * .25 * (np.random.rand(2) - np.array([.5, .5]))
                        color = color + np.random.randint(-min(20, min(color)), min(20, 255-max(color)), 3)
                        rot *= np.random.rand()

                    color = tuple(color)
                    x, y = pos

                    if (fig == "ellipse"):
                        img.drawEllipse(x, y, .75*dx, .75*dy, rot, color)
                    elif(fig == "rectangle"):
                        img.drawRect(x, y, .75*dx, .75*dy, rot, color)
                    elif(fig == "etoile"):
                        img.drawStar(x, y, dx, dy, 5, rot, color, self.noise)
                    elif(fig == "triangle"):
                        img.drawTriangle(x, y, dx, dy, rotation=rot, random_noise=self.noise, clr=color)

        return img





    