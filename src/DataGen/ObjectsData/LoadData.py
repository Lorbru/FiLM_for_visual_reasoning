import json 
import numpy as np

class Data():

    with open('src/DataGen/ObjectsData/ObjectsData.json', 'r') as f:
        ObjData = json.load(f)
    
    ClrList = [clr for clr in ObjData["color"]]
    FigList = [fig for fig in ObjData["shape"]]
    PosList33 = ObjData["position33"]
    PosList12 = ObjData["position12"]
    Qlist = [qtype for qtype in ObjData["qtypes"]]

    @staticmethod
    def getRGB(color):
        if color == None : 
            color = Data.randomColor()
        return Data.ObjData["color"][color]["RGB"]
    
    @staticmethod
    def getFigure(figure):
        if figure == None or figure == 'figure' :
            figure = Data.randomFigure(without=["figure"])
        return figure

    @staticmethod
    def randomColor(without=None):
        liste = Data.ClrList.copy()
        if without != None :
            for i in without :
                liste.remove(i)
        return np.random.choice(liste)
    
    @staticmethod
    def randomFigure(without=None):
        liste = Data.FigList.copy()
        if without != None :
            for i in without :
                liste.remove(i)
        return np.random.choice(liste)
    
    def getPosition33Id(position):
        return Data.ObjData["position33"].index(position)
    
    def getPosition12Id(position):
        return Data.ObjData["position12"].index(position)

    

    
    
    
    
    
