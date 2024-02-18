import json 
import numpy as np

class Data():

    with open('src/DataGenerator/json/ObjectsData.json', 'r') as f:
        ObjData = json.load(f)

    with open('src/DataGenerator/json/QA.json', 'r') as f:
        QAjson = json.load(f)
    
    ClrList = [clr for clr in ObjData["color"]]
    FigList = [fig for fig in ObjData["shape"]]
    QType = [qtype for qtype in QAjson]
    PosList = ObjData["position"]

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
    
    @staticmethod
    def randomQType(without=None):
        liste = Data.QType.copy()
        if without != None :
            for i in without :
                liste.remove(i)
        return np.random.choice(liste)
    
    @staticmethod
    def randomPosition():
        return np.random.choice(Data.PosList)
    
    @staticmethod
    def getColorId(clr):
        return Data.ClrList.index(clr)
    
    @staticmethod
    def getPositionId(pos):
        return Data.PosList.index(pos)
    
    @staticmethod
    def getFigId(fig):
        return Data.FigList.index(fig)
    
    @staticmethod
    def getQTypeId(type):
        return Data.QType.index(type)
    
    
class AnswerData():

    def __init__(self, type_vocab):

        with open(f'src/DataGenerator/AVocabulary/{type_vocab}.json', 'r') as f:
            self.answerList = json.load(f)
        
    def getAnswerIndex(self, answer):
        return self.answerList.index(answer)
    
    def getAnswer(self, index):
        return self.answerList[index]
    
    def randomAnswer(self):
        return np.random.choice(self.answerList)
    
