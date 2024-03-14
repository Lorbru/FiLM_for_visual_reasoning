import PIL 
from PIL import Image, ImageDraw
import numpy as np
import random

class Shapes():

    """
    ============================================================================================
    CLASS SHAPES : class used to draw images using the PIL library
    
    STATIC ATTRIBUTES : 
        * DIM :int - default image dimension size 
    
    ATTRIBUTES :
        * img :Image - the image
        * dim :int - image dimension

    METHODS : 
        * drawEllipse(x, y, dx, dy, rotation, clr) : draw an ellipse on the image
        * drawRect(x, y, dx, dy, rotation, clr) : draw a rectangle on the image
        * drawStar(x, y, dx, dy, n, rotation, clr, random_noise) : draw a star on the image
        * drawTriangle(x, y, dx, dy, rotation, clr, random_noise) : draw a triangle on the image
        * randomGradient(): set a random background gradient on the image
        * saveToPNG(directory) : save the image to a PNG file 
    ============================================================================================
    """

    DIM = 180

    def __init__(self, dim=DIM):
        """
        -- __init__(dim=DIM) : constructor

        In >> :
            * dim :int       - img dimension (pixels height and width) 
        """
        self.img = Image.new('RGB', (dim, dim), 'black')
        self.dim  = dim
        
    def drawEllipse(self, x, y, dx, dy, rotation=0, clr='blue'):
        """
        -- drawEllipse(x, y, dx, dy, rotation=0, clr='blue') : draw an ellipse on the image

        In >> :
            * x: int   - x North West pixel position
            * y: int   - y North West pixel position
            * dx: int   - size x
            * dy: int   - size y
            * rotation: int  - rotation of the figure (degree)
            * clr: str  - color of the figure (if available in ObjectsData.json)
        """
        # matrice de rotation
        phi = (rotation/180) * np.pi
        rot_matrix = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])

        # angles pour l'équation de l'ellipse
        thetas = np.linspace(0, 2*np.pi, 200)

        # rayons de l'ellipse
        rx, ry = dx//2, dy//2

        # centre de l'ellipse
        cx, cy = x + dx//2, y + dy//2
        center = np.array([cx, cy])
        
        # calcul des points
        pts = []
        for th in thetas:

            # équation de l'ellipse
            p_centered = np.array([rx*np.cos(th), ry*np.sin(th)])

            # transformation (translation + rotation)
            p_transform = center + np.dot(rot_matrix, p_centered)
            px, py = int(p_transform[0]), int(p_transform[1])
            pts.append((px, py))
        
        # dessin
        draw = ImageDraw.Draw(self.img)
        draw.polygon(pts, fill=clr)

    def drawRect(self, x, y, dx, dy, rotation=0, clr='blue'):
        """
        -- drawRect(x, y, dx, dy, rotation=0, clr='blue') : draw a rectangle on the image

        In >> :
            * x: int   - x North West pixel position
            * y: int   - y North West pixel position
            * dx: int   - size x
            * dy: int   - size y
            * rotation: int  - rotation of the figure (degree)
            * clr: str  - color of the figure (if available in ObjectsData.json)
        """
        # points standards
        pts_std = np.array([
            [x, y],
            [x+dx, y], 
            [x+dx, y+dy], 
            [x, y+dy]
        ])

        # matrice de rotation
        phi = (rotation/180) * np.pi
        rot_matrix = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)]
        ])

        # centre
        cx, cy = x + dx//2, y + dy//2
        center = np.array([cx, cy])
        pts = []

        # translation et rotation
        for pt_std in pts_std:
            p_centered = np.array(pt_std - center)
            p_transform = center + np.dot(rot_matrix, p_centered)
            px, py = int(p_transform[0]), int(p_transform[1])
            pts.append((px, py))

        # dessin
        draw = ImageDraw.Draw(self.img)
        draw.polygon(pts, fill=clr)

    def drawStar(self, x, y, dx, dy, n=5, rotation=-np.pi/2, clr='blue', random_noise=False):
        """
        -- drawStar(x, y, dx, dy, n=5, rotation=-np.pi/2, clr='blue', random_noise) : draw a star on the image

        In >> :
            * x: int   - x North West pixel position
            * y: int   - y North West pixel position
            * dx: int   - size x
            * dy: int   - size y
            * n: int    - number of branches
            * rotation: int  - rotation of the figure (radian)
            * clr: str  - color of the figure (if available in ObjectsData.json)
            * random_noise :bool - noise on the star 
        """
        draw = ImageDraw.Draw(self.img)
        rx, ry = dx/2, dy/2
        thetas = np.linspace(0 + rotation, 2*np.pi + rotation, n+1)
        thetas = np.delete(thetas, -1)
        thetas = thetas + random_noise * 0.25 * ((2*np.pi)/(n+1)) * np.random.random(n)
        for k in range(0, n):
            pts = []
            #dessin par triangle élémentaire : deux branches modulo reliées au centre de l'étoile
            for subt in [k, k+2]:
                pts.append( (x + rx + rx*np.cos(thetas[subt%n]), y + ry + ry*np.sin(thetas[subt%n])) )
            pts.append((x+rx, y+ry))
            draw.polygon(pts, fill=clr)
            
    def drawTriangle(self, x, y, dx, dy, rotation=-np.pi/2, clr='blue', random_noise=False):
        """
        -- drawStar(x, y, dx, dy, rotation=-np.pi/2, clr='blue', random_noise) : draw a triangle on the image

        In >> :
            * x: int   - x North West pixel position
            * y: int   - y North West pixel position
            * dx: int   - size x
            * dy: int   - size y
            * rotation: int  - rotation of the figure (radian)
            * clr: str  - color of the figure (if available in ObjectsData.json)
            * random_noise :bool - noise on the triangle
        """
        draw = ImageDraw.Draw(self.img)
        rx, ry = dx/2, dy/2
        thetas = np.linspace(0 + rotation, 2*np.pi + rotation, 4)
        thetas = np.delete(thetas, -1)
        thetas = thetas + random_noise * 0.25 * ((2*np.pi)/4) * np.random.random(3)

        pts = []
        for th in thetas:
            pts.append( (x + rx + rx*np.cos(th), y + ry + ry*np.sin(th)) )

        draw.polygon(pts, fill=clr)
    
    def randomGradient(self):
        """
        -- randomGradient() : add random gradient on the image
        """
        gradient = np.zeros((self.dim, self.dim, 3), dtype=np.uint8)
        color1 = np.random.randint(0, 256, size=3)
        color2 = np.random.randint(0, 256, size=3)
        for x in range(self.dim):
            gradient[x, :, :] = (color1 * (1 - x/(self.dim-1)) + color2 * (x/(self.dim-1))).astype(np.uint8)
        self.img.paste(Image.fromarray(gradient))

    def saveToPNG(self, directory):
        """
        -- saveToPNG(directory) : save image to a PNG file

        In >> :
            * directory :str - filename for saving the file
        """
        self.img.save(directory)
