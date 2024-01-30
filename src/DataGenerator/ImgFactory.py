from Shapes import Shapes
from LoadData import Data
import numpy as np

class ImgFactory():

    @staticmethod
    def draw33RandomFigure(dim):

        randomImg = Shapes(dim)
        randomImg.randomGradient()
        unit = dim/3
        tile_shape = np.array([unit, unit])

        for i in range(3):
            for j in range(3):
                alea_d = unit/2 * np.random.rand(2)
                x, y = np.array([i * unit, j * unit]) + alea_d
                dx, dy = (tile_shape - alea_d) * (0.5 + np.random.rand(2)/2)
                alea_rot = 360 * np.random.rand()
                alea_fig = np.random.randint(0, 4)
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
        randomImg.randomGradient()
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