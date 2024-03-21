from .ImgFactory import ImgFactory
import numpy as np
from .Question import Question
from .Vocab import BuildVocab
import json

class DataGenerator():

    """
    ============================================================================================
    CLASS DATAGENERATOR : class used to generate triplets (images, questions, answer) we use 
    for training and testing our model. It is the main class of the DataGen folder.
    
    ATTRIBUTES : 
        * type :str      - name of the data we want to use (a namefile from the folder IQArelations)
        * relation :json - json loading of the data type we want to use
        * imgType :str   - imageType of our data
        * gradient :bool - if we use a background random gradient
        * noise :bool    - if we add random noise to our images
        * qtypes :list[str]      - question type list
        * answers :list[str]     - possible answers list
        * imgFactory :ImgFactory - image generator
        * vocab : BuildVocab     - all the vocabulary used

    METHODS : 
        * __init__(dim, data_type) : constructor
        * buildData() : build a random (question, answer, img)
        * getAnswerSize() : number of possible answers
        * getAnswerId(answer :str) : answer label index 
        * getVocabSize() : vocabulary size
        * getEncodedSentence(sentence :str) : encoded sentence for the GRU network
    ============================================================================================
    """

    def __init__(self, dim, data_type):
        """
        -- __init__(dim, data_type) : constructor

        In >> :
            * dim :int       - img dimension (pixels height and width) 
            * data_type :str - data structure we use (IQArelations namefile)
        """
        self.type = data_type

        with open(f"src/DataGen/IQArelations/{data_type}.json", 'r') as f :
            self.relation = json.load(f)

        self.imgType = self.relation["images"]["type"]
        self.gradient = bool(self.relation["images"]["gradient"])
        self.noise = bool(self.relation["images"]["noise"])

        self.qtypes =  [qtype for qtype in self.relation["questions"]]
        self.answers = self.relation["answers"]

        self.imgFactory = ImgFactory(dim, self.imgType, self.gradient, self.noise)
        self.vocab = BuildVocab()

    def buildData(self):
        """
        -- buildData() : build a random data by using the actual configuration

        Out << :
            * (question: Question, answer: str, img: Shapes) : a single random data
        """
        qtype = np.random.choice(self.qtypes)
        answer = np.random.choice(self.relation["questions"][qtype])
        question = Question.buildQuestion(qtype)
        img = self.imgFactory.buildData(question, answer)

        return (question, answer, img)
        
    def getAnswerSize(self):
        """
        -- getAnswerSize() : number of possible answers in the actual configuration

        Out << :
            * int : number of possible answers
        """
        return len(self.answers)
    
    def getAnswerId(self, answer):
        """
        -- getAnswerId(answer :str) : index of the answer in the actual configuration

        In >> :
            * answer :str - answer label

        Out << :
            int - answer index
        """
        return self.answers.index(answer)
    
    def getVocabSize(self):
        """
        -- getVocabSize() : size of the vocabulary

        Out << :
            int - vocabulary size
        """
        return self.vocab.vocabSize()
    
    def getEncodedSentence(self, sentence, check_words=False):
        """
        -- getEncodedSentence(sentence :str) : get encoded sentence for the GRU network

        In >> :
            * sentence :str - sentence in natural language
        
        Out << :
            ndarray(int) - encoded sentence
        """
        return self.vocab.encode_sentence(sentence, check_words)
    
