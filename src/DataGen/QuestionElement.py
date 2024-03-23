import numpy as np
from .ObjectsData.LoadData import Data

class QuestionElement():
    
    """
    ============================================================================================
    CLASS QUESTION ELEMENT : class used to get natural french language syntax on a question
    element
    
    ATTRIBUTES : 
        * count :int - count of figures (un, deux, trois, ...)
        * shape :str - type of figure (etoile, ellipse, rectangle, ...)
        * color :str - color (rouge, vert, bleu, ...)

    METHOD : 
        * printObject(indet) : str of the object element (ex : 'trois etoiles rouges')

    STATIC METHOD :
        * buildRandomElement() : build a random question element 
    ============================================================================================
    """

    def __init__(self, count:int, shape:str, color:str):
        """
        -- __init__(count, shape, color) : constructor.

        In >> :
            * count :int - count of figures (un, deux, trois, ...)
            * shape :str - type of figure (etoile, ellipse, rectangle, ...)
            * color :str - color (rouge, vert, bleu, ...)
        """
        self.count = count                       
        self.shape = shape    
        self.color = color  

    def printObject(self, indet=False):
        """
        -- printObject(indet=False) : str of object.

        In >> :
            * indet: bool -  if False, precise count of elements, else indetermined form : "des figures"

        Out << :
            str : print of question element 
        """
        clr = ""
        genre = Data.ObjData["shape"][self.shape]["genre"]
        if indet : 
            shape = Data.ObjData["shape"][self.shape]["indet"]
            if self.color != "":
                clr = " " + Data.ObjData["color"][self.color][genre + "P"]
            return shape + clr
        else :
            
            if self.count > 1 : 
                shape = Data.ObjData["shape"][self.shape]["plural"]
                if self.color != "":
                    clr = " " + Data.ObjData["color"][self.color][genre + "P"]
                num = "des"
            else :
                shape = Data.ObjData["shape"][self.shape]["singul"]
                if self.color != "":
                    clr = " " + Data.ObjData["color"][self.color][genre]
                if genre == "M":
                    num = "un"
                else : 
                    num = "une"
            return num + " " + shape + clr

    @staticmethod
    def randomElement() :
        """
        -- randomElement()) : generate a random question element

        Out << :
            QuestionElement : random generated element
        """
        rd_count = np.random.randint(0, 10)
        rd_shape = np.random.choice(Data.FigList)
        rd_clr = np.random.choice(Data.ClrList + [""])
        
        return QuestionElement(rd_count, rd_shape, rd_clr)