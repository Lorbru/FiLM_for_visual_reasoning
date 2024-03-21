from Model.ModelTrain import ModelTrain
from Model.ModelTest import ModelTest
from Analysis import FiLMGeneratorPCA
import random
import numpy as np
from DataGen.DataGenerator import DataGenerator

def main():
    print("#########################################################")
    print("###############      RUNNING PROJECT      ###############")
    print(f"#########################################################\n")


    
    # Seed for reproduction
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"  > Setting project SEED : {SEED}")

    # ModelTrain(model_name="mod_best_data7")
    
    # ModelTest("mod_best_data7")
    
    # FiLMGeneratorPCA("mod_best_data3")

    datagen = DataGenerator(180, "data3")

    print(datagen.getEncodedSentence("De quelle couleur troll figure a droite ?", True))

    print(f"\n===========            END PROCESS            ===========")
    
    return 0

if __name__ == "__main__":
    main()