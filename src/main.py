from Model.ModelTrain import ModelTrain
from Model.ModelTest import ModelTest
from Analysis import FiLMGeneratorPCA
from DataGen.Vocab import BuildVocab
import random
import numpy as np
from DataGen.DataGenerator import DataGenerator
import os
import gdown

def main():
    print("#########################################################")
    print("###############      RUNNING PROJECT      ###############")
    print(f"#########################################################\n")


    
    # Seed for reproduction
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"  > Setting project SEED : {SEED}")
    
    # if os.path.isfile('src/Data/mod_best_data3.pth') == False :
    #     print(f"  > Downloading mod_best_data3.pth from https://drive.google.com")
    #     url = 'https://drive.google.com/file/d/13mYfOEcDRQ_yscapmz4Z3NaNL34qfTkh/view?usp=sharing'
    #     output = 'src/Data/mod_best_data3.pth'
    #     gdown.download(url, output, quiet=False, fuzzy=True)

    ModelTrain()
    
    print(f"\n===========            END PROCESS            ===========")
    
    return 0

if __name__ == "__main__":
    main()