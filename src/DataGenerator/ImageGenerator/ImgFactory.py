from Shapes import Shapes
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
                alea_color = tuple(Shapes.RGB_COLOR_LIST[np.random.choice(["red", "green", "blue", "white"])] + np.random.randint(-20, 20, 3))
                if (alea_fig == 0):
                    randomImg.drawEllipse(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(alea_fig == 1):
                    randomImg.drawRect(x, y, .75*dx, .75*dy, alea_rot, alea_color)
                elif(alea_fig == 2):
                    randomImg.drawStar(x, y, dx, dy, np.random.randint(5, 7), alea_rot, alea_color, False)
                else:
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
                if figures[i, j] != None :
                    
                    fig = figures[i, j]
                    alea_d = unit/2 * np.random.rand(2)
                    x, y = np.array([i * unit, j * unit]) + alea_d
                    dx, dy = (tile_shape - alea_d) * (0.5 + np.random.rand(2)/2)
                    alea_rot = 360 * np.random.rand()
                    color = tuple(np.array(Shapes.RGB_COLOR_LIST[colors[i, j]]) + np.random.randint(-20, 20, 3))
                    
                    if (fig == "ellipse"):
                        randomImg.drawEllipse(x, y, .75*dx, .75*dy, alea_rot, color)
                    elif(fig == "rectangle"):
                        randomImg.drawRect(x, y, .75*dx, .75*dy, alea_rot, color)
                    elif(fig == "etoile"):
                        randomImg.drawStar(x, y, dx, dy, np.random.randint(5, 8), alea_rot, True, color)
                    elif(fig == "triangle"):
                        randomImg.drawTriangle(x, y, dx, dy, rotation=alea_rot, random_noise=True, clr=color)

        return randomImg

def test():

    for k in range(50):
        ImgFactory.draw33RandomFigure(180).saveToPNG(f"src/Data/Img33/img{k}.png")

test()