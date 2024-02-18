from .LoadData import Data
from .Shapes import Shapes
import numpy as np

class ImgFactory():

    @staticmethod
    def draw12RandomFigure(dim, type_gauche=None, type_droit=None):

        randomImg = Shapes(dim)
        types = [type_gauche, type_droit]
        unit = dim/2
        for i in range(2):
            color = tuple(np.array(Data.ObjData["color"][Data.randomColor()]["RGB"]))
            alea_fig = types[i]
            if (alea_fig == None):
                alea_fig = Data.randomFigure(without = ["figure"])
            if (alea_fig == "ellipse"):
                randomImg.drawEllipse((i*unit)+dim/8, dim/3, .5*unit, .5*unit, clr=color)
            elif (alea_fig == "rectangle") :
                randomImg.drawRect((i*unit)+dim/8, dim/3, .5*unit, .5*unit, clr=color)
            elif (alea_fig == "triangle") :
                randomImg.drawTriangle((i*unit)+dim/8, dim/3, .5*unit, .5*unit, clr=color)
            elif (alea_fig == "etoile") :
                randomImg.drawStar((i*unit)+dim/8, dim/3, .5*unit, .5*unit, clr=color)
        return randomImg
            
            
    @staticmethod
    def draw33RandomFigure(dim):

        randomImg = Shapes(dim)
        # randomImg.randomGradient()
        unit = dim/3
        tile_shape = np.array([unit, unit])

        for i in range(3):
            for j in range(3):
                alea_d = unit/2 * np.random.rand(2)
                x, y = np.array([i * unit, j * unit]) + alea_d
                dx, dy = (tile_shape - alea_d) * (0.5 + np.random.rand(2)/2)
                alea_rot = 360 * np.random.rand()
                alea_fig = np.random.choice(["ellipse", "etoile", "rectangle", "triangle"])
                alea_color = np.array(Data.ObjData["color"][Data.randomColor()]["RGB"])
                alea_noise = np.random.randint(-min(20, min(alea_color)), min(20, 255-max(alea_color)), 3)
                alea_color = tuple(alea_color + alea_noise)
                if (alea_fig == 0):
                    randomImg.drawEllipse(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(alea_fig == 1):
                    randomImg.drawRect(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(alea_fig == 2):
                    randomImg.drawStar(x, y, dx, dy, np.random.randint(5, 7), alea_rot, alea_color, False)
                elif(alea_fig == 3):
                    randomImg.drawTriangle(x, y, dx, dy, rotation=alea_rot, random_noise=True, clr=alea_color)

        return randomImg
    
    @staticmethod
    def draw33Figure(figures, colors, dim):
        
        randomImg = Shapes(dim)
        # randomImg.randomGradient()
        unit = dim/3
        tile_shape = np.array([unit, unit])

        for i in range(3):
            for j in range(3):
                    
                alea_d = unit/2 * np.random.rand(2)
                x, y = np.array([j * unit, i * unit]) + alea_d
                dx, dy = (tile_shape - alea_d) * (0.5 + np.random.rand(2)/2)
                alea_rot = 360 * np.random.rand()
                fig = figures[3*i + j]
                alea_color = Data.ObjData["color"][Data.ClrList[colors[3*i + j]]]["RGB"]
                alea_noise = np.random.randint(-min(20, min(alea_color)), min(20, 255-max(alea_color)), 3)
                alea_color = tuple(alea_color + alea_noise)
                if (fig == 3):
                    randomImg.drawEllipse(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(fig == 4):
                    randomImg.drawRect(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(fig == 2):
                    randomImg.drawStar(x, y, dx, dy, np.random.randint(5, 7), alea_rot, alea_color, False)
                elif(fig == 1):
                    randomImg.drawTriangle(x, y, dx, dy, rotation=alea_rot, random_noise=True, clr=alea_color)

        return randomImg

    @staticmethod
    def drawUniqueRandomFigure(dim):
        fig = Data.randomFigure(without=["figure"])
        alea_color = Data.randomColor()
        return ImgFactory.drawUniqueFigure(fig, alea_color, dim)
        
    @staticmethod
    def drawUniqueFigure(fig, color, dim):

        randomImg = Shapes(dim) 
        alea_rot = 360 * np.random.rand()
        dx, dy = np.array([dim//2, dim//2]) * (0.5 + np.random.rand(2)/2)
        x, y = dx//2, dy//2
        alea_color = Data.ObjData["color"][color]["RGB"]
        if (fig == "ellipse"):
            randomImg.drawEllipse(x, y, .75*dx, .75*dy, alea_rot, color)
        elif(fig == "rectangle"):
            randomImg.drawRect(x, y, .75*dx, .75*dy, alea_rot, color)
        elif(fig == "etoile"):
            randomImg.drawStar(x, y, dx, dy, np.random.randint(5, 7), alea_rot, color, False)
        elif(fig == "triangle"):
            randomImg.drawTriangle(x, y, dx, dy, rotation=alea_rot, random_noise=True, clr=color)

        return randomImg

def test():

    Img = ImgFactory.draw12RandomFigure(120, "etoile", "ellipse")
    Img.saveToPNG('src/Data/img_12.png')

test()