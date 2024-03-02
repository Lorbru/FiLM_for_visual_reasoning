import numpy as np
from .ObjectsData.LoadData import Data

class QuestionElement():
    
    def __init__(self, count:int, shape:str, color:str):
        
        self.count = count                       
        self.shape = shape    
        self.color = color  


    def printObject(self, indet=False):

        clr = ""
        genre = Data.ObjData["shape"][self.shape]["genre"]
        if indet : 
            shape = Data.ObjData["shape"][self.shape]["indet"]
            if self.color != "":
                clr = " " + Data.ObjData["color"][self.color][genre + "P"]
            return shape + clr
        else : 
            num = "des"
            if self.count != 1 : 
                shape = Data.ObjData["shape"][self.shape]["plural"]
                if self.color != "":
                    clr = " " + Data.ObjData["color"][self.color][genre + "P"]
            else :
                shape = Data.ObjData["shape"][self.shape]["singul"]
                if self.color != "":
                    clr = " " + Data.ObjData["color"][self.color][genre]
            return num + " " + shape + clr

    @staticmethod
    def randomElement() :

        rd_count = np.random.randint(0, 10)
        rd_shape = np.random.choice(Data.FigList)
        rd_clr = np.random.choice(Data.ClrList + [""])
        
        return QuestionElement(rd_count, rd_shape, rd_clr)