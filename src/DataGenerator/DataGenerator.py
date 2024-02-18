from src.DataGenerator.ImgFactory import ImgFactory
from src.DataGenerator.QAFactory import QAFactory
import numpy as np
from src.DataGenerator.LoadData import Data, AnswerData
from src.DataGenerator.Vocab import BuildVocab

class DataGenerator():

    def __init__(self, data_type):
        
        self.imgFactory = ImgFactory()
        self.vocab = BuildVocab()
        self.type = data_type
        self.answers = AnswerData(data_type)
        self.qaFactory = QAFactory()

    def buildData(self):
        if self.type == "completeQA":
            return self.buildImageFromQA()
        elif self.type == "rightLeftQA":
            return self.buildRightLeftQA()
        
    def getAnswerSize(self):
        return len(self.answers.answerList)
    
    def getAnswerId(self, answer):
        return self.answers.getAnswerIndex(answer)
    
    def getVocabSize(self):
        return len(self.vocab.vocab)
    
    def getEncodedSentence(self, sentence):
        return self.vocab.encode_sentence(sentence)
    

    # ==========================================================================================================

    def buildRightLeftQA(self):
        quest = QAFactory.randomRightLeftQuestion()
        ans = self.answers.randomAnswer()
        if quest == "gauche":
            return (quest, ans, ImgFactory.draw12RandomFigure(120, type_gauche = ans, type_droit = None))
        else :
            return (quest, ans, ImgFactory.draw12RandomFigure(120, type_gauche = None, type_droit = ans))

    def buildImageFromQA(self, question=None):

        if question == None :
            (question, answer) = self.qaFactory.randomQuestionAndAnswer()
            qtype = question.type
        else :
            qtype = question.type
            (question, answer) = self.qaFactory.randomQuestionAndAnswer(qtype, question.direction, question.mainObject, question.secondObject)
        
        shapes = np.zeros(9, dtype=int) -1
        clrs = np.zeros(9, dtype=int) -1

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
                        clrs[i] = Data.getColorId(color)
                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                    shapes[i] = Data.getFigId(figure)

                else :

                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        clrs[i] = Data.getColorId(Data.randomColor(without = [color]))
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                        shapes[i] = Data.getFigId(Data.randomFigure(without = [figure, 'figure']))
                
        elif (qtype == 'comptage'):
            figure = question.mainObject.shape
            color = question.mainObject.color
            count = int(answer)
            idalea = np.random.choice(list(range(9)), count, replace=False)
            for i in range(9):

                if i in idalea :
                    if (color != ''):
                        clrs[i] = Data.getColorId(color)
                    else : 
                        clrs[i] = Data.getColorId(Data.randomColor())
                    if (figure != 'figure'):
                        shapes[i] = Data.getFigId(figure)
                    else :
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                else :

                    # 1/2 de différer au moins par la couleur : obliger de différer par la couleur si la figure n'est pas définie
                    if (color != "" and (figure == 'figure' or np.random.rand() > .5)):
                        clrs[i] = Data.getColorId(Data.randomColor(without = [color]))
                        shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))

                    # 1/2 de différer au moins pas la forme : obliger de différer par la fforme si la couleur n'est pas définie
                    else :
                        clrs[i] = Data.getColorId(Data.randomColor())
                        shapes[i] = Data.getFigId(Data.randomFigure(without = [figure, 'figure']))
        
        elif (qtype == 'position'):
            idPos = Data.getPositionId(question.direction)
            for i in range(9):
                if i == idPos:
                    shapes[i] = Data.getFigId(answer)
                else : 
                    shapes[i] = Data.getFigId(Data.randomFigure(without = ['figure']))
                clrs[i] = Data.getColorId(Data.randomColor())

        return (question, answer, ImgFactory.draw33Figure(shapes, clrs, 180))
        
    def  buildUniqueImageFromFigure(self):

        answer = Data.randomFigure(without=["figure"])
        alea_color = Data.randomColor()
        image = ImgFactory.drawUniqueFigure(answer, alea_color, 180)

        return (answer, image)

    # ==========================================================================================================



