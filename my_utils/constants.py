### Constants. Read constants from a csv file, and change its content
import json, os
import pandas as pd
from enum import Enum


class Const_c():

    class DATASETS(Enum): 
        CAPITAN  = 'b-59-850'
        TKH      = 'TKH'
        BREAKHIS = 'BreaKHis_formatted'
        EGYPTIAN = 'Egyptian'
        GREEK    = 'Greek'

    json_folder = "my_utils/constants/"



    def __init__(self, exp_num, full_name, **subparams):
        # Read csv with pandas
        self.folder = Const_c.json_folder + exp_num + "/"

        params = Const_c.BASE_PARAMETERS.copy()
        for key, value in subparams.items():
            params[key] = value
        self.Constants = params

        os.makedirs(self.folder, exist_ok=True)
        self.json_file = self.folder + "constants_" + str(full_name) + ".json"
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                self.Constants = json.load(f)

    def adjust_parameters(params):
        if not params['ALL_DATASETS']: # Over the same dataset train/test
            if Const_c.DATASETS.BREAKHIS.value in params["TGT_DATASETS"]:
                params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
        else: 
            if Const_c.DATASETS.BREAKHIS.value in params["DATASETS_NAMES"]:
                params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
        if Const_c.DATASETS.BREAKHIS.value in params["TGT_DATASETS"]:
            params["LIMIT_N_WAY_TEST"] = 3 if params["LIMIT_N_WAY_TEST"] > 3 else params["LIMIT_N_WAY_TEST"]

        return params

    def modify_constants(self, **kwargs):
        params = Const_c.BASE_PARAMETERS.copy()
        
        for key, value in kwargs.items():
            params[key] = value

        #############################################################################
        ## Adjust n-way
        ## This dataset only has max 8 classes, not possible to 5-way
        params = Const_c.adjust_parameters(params)
        #############################################################################

        os.makedirs(os.path.dirname(self.json_file), exist_ok=True)
        with open(self.json_file, 'w') as f:
            json.dump(params, f)
        self.Constants = params

    def remove_constants_file(self):
        os.remove(self.json_file)

    def exists_file_constants(self):
        return os.path.exists(self.json_file)

    def get_experiment_id(PARAMS, boots_iter, return_extra_params=False):
        PARAMS = Const_c.adjust_parameters(PARAMS)
        src_datasets = "_".join([i for i in PARAMS["DATASETS_NAMES"]])
        src_input_size = str(PARAMS["INPUT_SIZE"][list(PARAMS['TGT_DATASETS'].keys())[0]][0]) + "x" + str(PARAMS["INPUT_SIZE"][list(PARAMS['TGT_DATASETS'].keys())[0]][1]) 

        string_id_base = PARAMS['MODEL_TYPE'] + "--" + str(PARAMS["EPISODES"]) + "_Episodes--" \
                + src_input_size + "_INPUT_SIZE/" \
                + str(PARAMS["USE_ORIGINAL_FIXED_SUPP_SET"]) + "_FIXED_SUPP_SET--" \
                + src_datasets 
        if boots_iter > 0 and not PARAMS['ALL_DATASETS']: # To reuse the previous ones
            string_id_base += "--boots_iter_" + str(boots_iter)
        string_id_base += "--" \
                + str(PARAMS["LIMIT_N_WAY_TRAIN"]) + "_NWAY_TRAIN--" \
                + str(PARAMS["BATCH_SIZE"]) + "_Batch_Size--" \
                + str(PARAMS['SAMPLES_PER_CLASS']) + "_KSamples_per_Class--" \
                + str(PARAMS["lr"]) + "_lr--" \
                + str(PARAMS["VALIDATION_PERC"]) + "_ValSrc--" 
        string_id_ft = string_id_base + "--GROUP_EXP_" + PARAMS["GROUP_EXPERIMENT"] + "--NWAY-test_" + str(PARAMS["LIMIT_N_WAY_TEST"]) + "--tgt_dataset_" + sorted(PARAMS["TGT_DATASETS"].keys() )[0] 

        if return_extra_params:
            return string_id_base, string_id_ft, src_datasets, src_input_size
        return string_id_base, string_id_ft

    def update_filter_search(df, Constants):
        input_size = Constants['INPUT_SIZE'][list(Constants['TGT_DATASETS'].keys())[0]]
        params_filter = {'samples_per_class': Constants['SAMPLES_PER_CLASS'], 
                         'model_type': Constants['MODEL_TYPE'],
                         'batch_size': Constants['BATCH_SIZE'],
                         'input_size': str(input_size[0]) + 'x' + str(input_size[1]),
                         'lr': Constants['lr'],
                         'n_way_train': Constants['LIMIT_N_WAY_TRAIN'],
                         'n_way_test': Constants['LIMIT_N_WAY_TEST'],
                         'tgt_dataset': sorted(Constants['TGT_DATASETS'].keys())[0],
                         'src_datasets': "_".join([i for i in Constants['DATASETS_NAMES']]),
                         }

        for key, value in params_filter.items():
            df = df[df[key] == value]
        return df

    # If False, must be executed again
    def all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter):
        exp_name_sufix = "_tgt_ds--" + ds_name + "--spc_" + str(Constants['SAMPLES_PER_CLASS']) + "--model--" + Constants['MODEL_TYPE']  
        path_logs = "logs_csv/last_exec_" + exp + "/" + exp_name + exp_name_sufix + "_.csv"
        
        if os.path.exists(path_logs):
            df = pd.read_csv(path_logs)
            df = Const_c.update_filter_search(df, Constants)
            if boots_iter in df['bootstrap_iter'].values:
                print("Execution not finished but boots iter", boots_iter, "Already done")
                return True

        return False

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

    # BASE_PARAMETERS["INPUT_SIZE"] = (40, 40) #(28, 28) #(84, 84) #(40, 40) # Size the symbol images are resized
    ## Fixed Input_size depending on dataset:
    BASE_PARAMETERS["INPUT_SIZE"] = {
        DATASETS.CAPITAN.value       : (40,40),
        DATASETS.TKH.value           : (40,40),
        DATASETS.EGYPTIAN.value      : (40,40),
        DATASETS.GREEK.value         : (40,40),
        DATASETS.BREAKHIS.value      : (84,84),
        
        "omniglot"                   : (28,28),
        "omniglot_SOTA_testSet"      : (28,28),
        "omniglot_SOTA_trainvalSet"  : (28,28),

        "miniImageNet"               : (84,84),
        "miniImageNet_SOTA_testSet"  : (84,84),
        "miniImageNet_SOTA_trainSet" : (84,84),
    }


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
                    "omniglot_SOTA_trainvalSet", "miniImageNet_SOTA_trainSet", "BreaKHis_formatted"]
                    

    if not BASE_PARAMETERS["FineTuning"]:
        BASE_PARAMETERS["epochsFineTuning"] = 0

    #### DEBUG_FASTer settings
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

    # BASE_PARAMETERS["Experiment"] += "_InputS_" + str(BASE_PARAMETERS["INPUT_SIZE"][0]) + "x" + str(BASE_PARAMETERS["INPUT_SIZE"][1])


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


