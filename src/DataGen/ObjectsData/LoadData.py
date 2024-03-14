import json 
import numpy as np

class Data():

    """
    ============================================================================================
    CLASS DATA : class used to load and get the grammar and all the keywords, labels and index 
    to build questions (color, shapes, position)
    This class is used in parallel of the ObjectsData.json file

    STATIC ATTRIBUTES : 
        * ObjData : json file with vocabulary, grammar, syntax (see ObjectsData.json)
        * ClrList : color list labels used for our project
        * FigList : figure list labels used for our project
        * PosList33 : positions list labels used for our project (3 * 3 images)
        * PosList12 : positions list labels used for our project (1 * 2 images)
        * Qlist : question type list labels used for our prroject

    STATIC METHODS : 
        * getRGB(color :str) : convert to RGB from a color label
        * getFigure(figure :str) : convert to figure from a figure label
        * randomColor(without :list) : get random color from all possible colors except 'without'
        * randomFigure(without :list) : get random figure from all possible figures except 'without'
        * getPosition33Id(position :str) : get position index from a position label (3 * 3 image)
        * getPosition12Id(poition :str) : get position index from a position label (1 * 2 image)
    ============================================================================================
    """

    with open('src/DataGen/ObjectsData/ObjectsData.json', 'r') as f:
        ObjData = json.load(f)
    
    ClrList = [clr for clr in ObjData["color"]]
    FigList = [fig for fig in ObjData["shape"]]
    PosList33 = ObjData["position33"]
    PosList12 = ObjData["position12"]
    Qlist = [qtype for qtype in ObjData["qtypes"]]

    @staticmethod
    def getRGB(color):
        """
        -- getRGB(color :str) : convert to RGB from a color label
        
        In >> : 
            * color : a color label which is available in ObjectsData.json
    
        Out << : 
            ndarray : RGB value   
        """
        if color == None : 
            color = Data.randomColor()
        return Data.ObjData["color"][color]["RGB"]
    
    @staticmethod
    def getFigure(figure):
        """
        -- getFigure(figure :str) : get a precise figure from a general figure label
        
        In >> : 
            * figure : a figure label which is available in ObjectsData.json (maybe not defined : 'figure')
        
        Out << : 
            str : precise figure label
        """
        if figure == None or figure == 'figure' :
            figure = Data.randomFigure(without=["figure"])
        return figure

    @staticmethod
    def randomColor(without=None):
        """
        -- randomColor(without=None :list) : get random color label

        In >> :
            * without=None :list. List of colors we don't want to get
        
        Out << :
            str : random color label
        """
        liste = Data.ClrList.copy()
        if without != None :
            for i in without :
                liste.remove(i)
        return np.random.choice(liste)
    
    @staticmethod
    def randomFigure(without=None):
        """
        -- randomFigure(without=None :list) : get random figure label

        In >> :
            * without=None :list. List of figures we don't want to get
        
        Out << :
            str : random figure label
        """
        liste = Data.FigList.copy()
        if without != None :
            for i in without :
                liste.remove(i)
        return np.random.choice(liste)
    
    def getPosition33Id(position):
        """
        -- getPosition33Id(position :str) : get 3*3 position index

        In >> :
            * position :str. 3*3 position label available on ObjectsData.json
        
        Out << :
            int : position index 
        """
        return Data.ObjData["position33"].index(position)
    
    def getPosition12Id(position):
        """
        -- getPosition12Id(position :str) : get 1*2 position index

        In >> :
            * position :str. 1*2 position label available on ObjectsData.json
        
        Out << :
            int : position index 
        """
        return Data.ObjData["position12"].index(position)

    

    
    
    
    
    
