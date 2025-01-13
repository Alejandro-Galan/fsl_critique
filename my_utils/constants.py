### Constants. Read constants from a csv file, and change its content
import json, os
import pandas as pd

class Const_c():

    json_file = "my_utils/constants/constants_"

    def __init__(self, exp_num):
        # Read csv with pandas
        j_file = Const_c.json_file + "exp" + str(exp_num) + ".json"
        if os.path.exists(Const_c.json_file + "exp" + str(exp_num) + ".json"):
            with open(j_file, 'r') as f:
                self.Constants = json.load(f)
        else:
            self.Constants = Const_c.BASE_PARAMETERS

    def modify_constants(self, group_exp_name=None, **kwargs):
        params = Const_c.BASE_PARAMETERS.copy()
        
        assert group_exp_name is not None, "group_exp_name must be provided"

        for key, value in kwargs.items():
            params[key] = value

        with open(Const_c.json_file + group_exp_name + ".json", 'w') as f:
            json.dump(params, f)



    BASE_PARAMETERS = {}
    ##### Main parameters
    #### Use Source dataset/s?
    BASE_PARAMETERS["ALL_DATASETS"] = True
    BASE_PARAMETERS["NoSrcDataset"] = False
    # Use all classes or only a fixed limit
    BASE_PARAMETERS["LIMIT_N_WAY_TRAIN"] = 5 #5 #20 # None
    BASE_PARAMETERS["LIMIT_N_WAY_TEST"]  = 5 #5 #20 # None
    
    ## DATASETS NAMES imp, must be provided
    # BASE_PARAMETERS["DATASETS_NAMES"] = ["b-59-850", "Egyptian", "TKH", "Greek"]
    # BASE_PARAMETERS["TGT_DATASETS"] = {"Greek": {}, "b-59-850": {}, "Egyptian": {}, "TKH": {}}

    BASE_PARAMETERS["INPUT_SIZE"] = (40, 40) #(28, 28) #(84, 84) #(40, 40) # Size the symbol images are resized
    BASE_PARAMETERS["MODEL_TYPE"] = [ "PrototypicalNetwork", "MatchingNetwork"] #["PrototypicalNetwork"] #["MatchingNetwork"]

    BASE_PARAMETERS["EPOCHS"] = 1 #500 #20 #500 #150
    BASE_PARAMETERS["EPISODES"] = 4000 #1000 #1000 #1000
    BASE_PARAMETERS["TEST_EPISODES"] = 4000 # In Prototypical they use only 1000, but I do not think it is consistent
    BASE_PARAMETERS["FINE_TUNING_EPISODES"] = 1000 #1000 #1000 #1000    
    BASE_PARAMETERS["BATCH_SIZE"] = 16 #1 #16
    BASE_PARAMETERS["SAMPLES_PER_CLASS"] = [1, 5, 10] #[10] #, 15, 20, 25, 30]
    
    BASE_PARAMETERS["BOOTSTRAP_ITERS"] = 5 #2 # TODO DEBUG, put 10 
    BASE_PARAMETERS["lr"] = 1e-3 #1e-4 #1e-3
    BASE_PARAMETERS["weight_decay"] = 1e-6 #1e-4 #1e-6
    BASE_PARAMETERS["step_size"] = 2000 # Established by Prototypical Network paper

    BASE_PARAMETERS["FineTuning"] = True #True
    BASE_PARAMETERS["epochsFineTuning"] = 3 #20

    BASE_PARAMETERS["GROUP_EXPERIMENT"] = "testing_experiments" # "all_experiments" 

    BASE_PARAMETERS["VALIDATION_SRC_SRC_DATA"] = False #False # Src domain for the validation
    BASE_PARAMETERS["VALIDATION_PERC"] = 0.0
    if BASE_PARAMETERS["VALIDATION_SRC_SRC_DATA"]:
        BASE_PARAMETERS["VALIDATION_PERC"] = 0.2
    BASE_PARAMETERS["TEST_SET_PERC"] = 0.2 # Only when same dataset is used for training and testing


    ## TODO the fixed supp set is from tgt dataset (allow using from src also at training)
    ### Could be useful that fixed n*k supp from tgt dataset 
    BASE_PARAMETERS["USE_ORIGINAL_FIXED_SUPP_SET"] = False #True
    BASE_PARAMETERS["change_classes_alias"] = True

    BASE_PARAMETERS["SHUFFLE_SUPP_SET"] = True #False
    BASE_PARAMETERS["ReusePretrained"] = True #True ## If stored weights, do not use them
    #####

    BASE_PARAMETERS["ALL_DATASETS_BASE"] = ["TKH", "b-59-850", "Egyptian", "Greek"]

    BASE_PARAMETERS["DEACTIVATE_WANDB"] = False #False


    BASE_PARAMETERS["SOTA_DATASETS"] = ["omniglot", "miniImageNet", "omniglot_SOTA_testSet", "miniImageNet_SOTA_testSet",
                    "omniglot_SOTA_trainvalSet", "miniImageNet_SOTA_trainSet"]
                    

    if not BASE_PARAMETERS["FineTuning"]:
        BASE_PARAMETERS["epochsFineTuning"] = 0

    #### Debug faster settings
    BASE_PARAMETERS["SMALL_TEST_SET"] = False # TODO Debug/Test porposes, faster training
    BASE_PARAMETERS["VALIDATION_SRC_TGT_DATA"] = True # Just metrics of the training without affecting the training
    BASE_PARAMETERS["LIMIT_VALIDATION_SRC_TGT"] = 1 # Only one validation in each episode
    BASE_PARAMETERS["LIMIT_VALIDATION_SRC_SRC"] = BASE_PARAMETERS["EPISODES"] / 20 # 10 evaluations 

    #### Training conf
    BASE_PARAMETERS["PATIENCE"] = 5




    BASE_PARAMETERS["lrFineTuning"] = 1e-4



    #### Experiments:
    BASE_PARAMETERS["Experiment"] = "FineTuning" if BASE_PARAMETERS["FineTuning"] else "NoFT"

    if BASE_PARAMETERS["LIMIT_N_WAY_TEST"]:
        if BASE_PARAMETERS["LIMIT_N_WAY_TRAIN"]:
            BASE_PARAMETERS["Experiment"] += "_" + str(BASE_PARAMETERS["LIMIT_N_WAY_TEST"]) + "wayTest--" + str(BASE_PARAMETERS["LIMIT_N_WAY_TRAIN"]) + "wayTrain"
        else:
            BASE_PARAMETERS["Experiment"] += "_" + str(BASE_PARAMETERS["LIMIT_N_WAY_TEST"]) + "way"
    else:
        BASE_PARAMETERS["Experiment"] += "_AllWay"

    BASE_PARAMETERS["Experiment"] += "_InputS_" + str(BASE_PARAMETERS["INPUT_SIZE"][0]) + "x" + str(BASE_PARAMETERS["INPUT_SIZE"][1])


    #### Prototypical
    BASE_PARAMETERS["n_query"] = 1

    # greek_comparative = False
    # if greek_comparative:
    #     Experiment = "GreekComp"

    #     DATASETS_NAMES = ["TKH"]
    #     TGT_DATASETS = {"Greek": {}}

    #     MODEL_TYPE = ["PrototypicalNetwork", "MatchingNetwork"]
    #     SAMPLES_PER_CLASS = [1, 5, 10]
    #     EPISODES = 100 #250 #1000
    #     epochsFineTuning = 5 #5   

    #     VALIDATION_SRC = True #True
    #     LIMIT_VALIDATION_SRC = 1 #5
    #     SMALL_TEST_SET = True

    #     FineTuning = True
    #     BATCH_SIZE = 4


