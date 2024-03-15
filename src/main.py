from Model.ModelTrain import ModelTrain
from Model.ModelTest import ModelTest
import random

def main():
    print("#########################################################")
    print("###############      RUNNING PROJECT      ###############")
    print(f"#########################################################\n")

    # Seed for data generation
    GENERATOR_SEED = 21227063002
    random.seed(GENERATOR_SEED)

    # ModelTrain()
    # ModelTest("mod_best_data3")
    
    print(f"\n===========            END PROCESS            ===========")
    
    return 0

if __name__ == "__main__":
    main()