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

    json_folder = "utils/constants/"



    def __init__(self, exp_num, full_name, **subparams):
        # Read csv with pandas
        self.folder = Const_c.json_folder + exp_num + "/"

        params = Const_c.BASE_PARAMETERS.copy()
        splt_tgt_ds = full_name.split("tgt_ds_")
        
        if len(splt_tgt_ds) > 1:
            params['TGT_DATASETS'] = {splt_tgt_ds[1]: {} }
        
        for key, value in subparams.items():
            params[key] = value

        params = Const_c.adjust_parameters(params)
        self.Constants = params


        # python3 scripts/small_run_matching_networks.py exp6_permuted_src_datasets--src_ds_Egyptian exp6 RelationNetwork-4000_E-40x40_I_S/False_FSS-Egyptian-5_NWAY_TRAIN-16_Batch_Size-5_KSpC-0.001_lr-0.0_ValSrc-GROUP_EXP_source_permutation-NWAY-test_5-tgt_ds_b-59-850 

        if "6" in exp_num: 
            params['SAMPLES_PER_CLASS'] = 1 #1 #5
            params['MODEL_TYPE'] = 'PrototypicalNetwork' #'RelationNetwork'
            params['DATASETS_NAMES'] =  {'NO_SRC_DATASET': {}} #{"omniglot_SOTA_trainvalSet": {}} #{Const_c.DATASETS.CAPITAN.value: {}} # {"miniImageNet_SOTA_trainSet": {}}
            params['TGT_DATASETS']   =  {Const_c.DATASETS.CAPITAN.value: {}} #{"omniglot_SOTA_testSet": {}} #{Const_c.DATASETS.CAPITAN.value: {}} # {"miniImageNet_SOTA_trainSet": {}}
            params['OVERWRITE_LOGS'] = False
            params["EPISODES"] = 4000 #20000
            params["BOOTSTRAP_ITERS"] = 5
            params["epochsFineTuning"] = 1 #3
            params["BATCH_SIZE"] = 16 #16
            params["LIMIT_N_WAY_TRAIN"] = 5
            params["LIMIT_N_WAY_TEST"] = 5
            
            
            params["NoSrcDataset"] = True

            # params["CLUSTERING"] = "CONSTRAINED-KMEANS"
            # params["M-labels-SRC"] = 1000
            # params["SMALL_TEST_SET"] = True


        os.makedirs(self.folder, exist_ok=True)
        self.json_file = self.folder + "constants_" + str(full_name) + ".json"
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                self.Constants = json.load(f)

    def adjust_parameters(params):
        ######### AT THIS MOMENT SAME AS TEST #############
        if "DATASETS_NAMES" in params:
            if Const_c.DATASETS.BREAKHIS.value in params["DATASETS_NAMES"]: 
                # if not params["CLUSTERING"]:
                params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
                params["LIMIT_N_WAY_TEST"] = 3 if params["LIMIT_N_WAY_TEST"] > 3 else params["LIMIT_N_WAY_TEST"]
        if Const_c.DATASETS.BREAKHIS.value in params["TGT_DATASETS"]:
            # if not params["CLUSTERING"]:
            params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
            params["LIMIT_N_WAY_TEST"] = 3 if params["LIMIT_N_WAY_TEST"] > 3 else params["LIMIT_N_WAY_TEST"]
        ###################################################
        
        # if not params['ALL_DATASETS']: # Over the same dataset train/test
        #     if Const_c.DATASETS.BREAKHIS.value in params["TGT_DATASETS"]:
        #         params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
        # else: 
        #     if Const_c.DATASETS.BREAKHIS.value in params["DATASETS_NAMES"]:
        #         params["LIMIT_N_WAY_TRAIN"] = 3 if params["LIMIT_N_WAY_TRAIN"] > 3 else params["LIMIT_N_WAY_TRAIN"] 
        # if Const_c.DATASETS.BREAKHIS.value in params["TGT_DATASETS"]:
        #     params["LIMIT_N_WAY_TEST"] = 3 if params["LIMIT_N_WAY_TEST"] > 3 else params["LIMIT_N_WAY_TEST"]
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

    def get_id_extensions(PARAMS, prev_str):
        extended_filename = prev_str + "GROUP_EXP_" + PARAMS["GROUP_EXPERIMENT"] + \
                "-NWAY-test_" + str(PARAMS['LIMIT_N_WAY_TEST']) + \
                "-tgt_ds_" + list(PARAMS["TGT_DATASETS"].keys())[0] + "-_tr_ftm.pt"
        # In case it is encoded
        return Const_c.read_dictionaly_of_files(extended_filename)
    
    def get_experiment_id(PARAMS, boots_iter, return_extra_params=False):
        PARAMS = Const_c.adjust_parameters(PARAMS)
        src_datasets = "_".join([i for i in PARAMS["DATASETS_NAMES"]])

        src_input_size = str(PARAMS["INPUT_SIZE"][list(PARAMS['TGT_DATASETS'].keys())[0]][0]) + "x" + str(PARAMS["INPUT_SIZE"][list(PARAMS['TGT_DATASETS'].keys())[0]][1]) 

        string_id_base = PARAMS['MODEL_TYPE'] + "-" + str(PARAMS["EPISODES"]) + "_E-" \
                + src_input_size + "_I_S/" \
                + str(PARAMS["USE_ORIGINAL_FIXED_SUPP_SET"]) + "_FSS-"
        
        if PARAMS["CLUSTERING"]:
            string_id_base += PARAMS["CLUSTERING"] + "_clst_m_" + str(PARAMS["M-labels-SRC"]) + "-"

        if PARAMS["ALL_DATASETS"]:
            string_id_base += src_datasets 
        else:
            string_id_base += sorted(PARAMS["TGT_DATASETS"].keys())[0] + "_baseline_over_own_ds"

        if boots_iter > 0 and not PARAMS['ALL_DATASETS']: # To reuse the previous ones
            string_id_base += "-boots_iter_" + str(boots_iter)
        string_id_base += "-" \
                + str(PARAMS["LIMIT_N_WAY_TRAIN"]) + "_NWAY_TRAIN-" \
                + str(PARAMS["BATCH_SIZE"]) + "_Batch_Size-" \
                + str(PARAMS['SAMPLES_PER_CLASS']) + "_KSpC-" \
                + str(PARAMS["lr"]) + "_lr-" \
                + str(PARAMS["VALIDATION_PERC"]) + "_ValSrc-" 
        string_id_ft = string_id_base + "GROUP_EXP_" + PARAMS["GROUP_EXPERIMENT"] + "-NWAY-test_" + str(PARAMS["LIMIT_N_WAY_TEST"]) + "-tgt_ds_" + sorted(PARAMS["TGT_DATASETS"].keys() )[0] 

        if return_extra_params:
            return Const_c.read_dictionaly_of_files(string_id_base), string_id_ft, src_datasets, src_input_size
        return Const_c.read_dictionaly_of_files(string_id_base), string_id_ft

    def update_filter_search(df, Constants):
        Constants = Const_c.adjust_parameters(Constants)

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

    def get_logs_csv_path(exp, exp_name, ds_name, Constants):
        exp_name_sufix = "_tgt_ds-" + ds_name + "-spc_" + str(Constants['SAMPLES_PER_CLASS']) + "-model-" + Constants['MODEL_TYPE']  
        # if Constants['LIMIT_N_WAY_TRAIN'] > 5:
        exp_name_sufix += "-" + str(Constants['LIMIT_N_WAY_TRAIN']) + "-nwayTrain"
        # else:
        # exp_name_sufix += "--" + str(5) + "-nwayTrain"
        return "logs_csv/last_exec_" + exp + "/" + exp_name + exp_name_sufix + "_.csv"




    # If False, must be executed again
    def all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter):
        path_logs = Const_c.get_logs_csv_path(exp, exp_name, ds_name, Constants)


        if os.path.exists(path_logs):
            if boots_iter == Constants["BOOTSTRAP_ITERS"]:
                print("Path logs already exists", path_logs)
            df = pd.read_csv(path_logs)
            df = Const_c.update_filter_search(df, Constants)
            # Check until that iteration
            for b_it in range(boots_iter):
                if not b_it in df['bootstrap_iter'].values:
                    print("Execution not finished in boots iter", boots_iter, "boots iteration")
                    return False
                df_boots = df[df['bootstrap_iter'] == b_it]
                if not "before_ft_eval_acc" in df_boots.columns:
                    return False
                # Check if all cells of column 'tr_eval_acc' are not empty
                if not (df_boots['before_ft_eval_acc'].notna()).any():
                    print("Empty cells of before_ft_eval_acc", boots_iter, "boots iteration")
                    breakpoint()
                    return False
                
            return True
        else: 
            print("Path logs does not exist", path_logs)
            return False
    
    def read_dictionaly_of_files(filename):
        dict_files_path = "utils/dictionary_of_files.csv"

        if os.path.exists(dict_files_path):
            df = pd.read_csv(dict_files_path)
            if filename in df['filename'].values:
                new_path = df[df['filename'] == filename]['path'].values[0]
                return new_path
            
        return filename   

    def add_to_dictionary_of_files(filename):
        dict_files_path = "utils/dictionary_of_files.csv"

        if os.path.exists(dict_files_path):
            df = pd.read_csv(dict_files_path)
            if filename in df['filename'].values:
                new_path = df[df['filename'] == filename]['path'].values[0]
            else:
                last_id = len(df['filename'])
                new_path = "encoded_files/" + str(last_id)
                df = df.append({'filename': filename, 'path': new_path}, ignore_index=True)
        else:
            new_path = "encoded_files/0"  
            df = pd.DataFrame(columns=['filename', 'path'])
            df = df.append({'filename': filename, 'path': new_path}, ignore_index=True)
        df.to_csv(dict_files_path, index=False)

        return new_path

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
    


    BASE_PARAMETERS["AllowedModels"] = [ "PrototypicalNetwork", "MatchingNetwork", "RelationNetwork" ]
    BASE_PARAMETERS["MODEL_TYPE"] = [ "PrototypicalNetwork", "MatchingNetwork"] 

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

    #### Clustering experiment / proposal. Expecting type
    BASE_PARAMETERS["CLUSTERING"]   = ""
    BASE_PARAMETERS["M-labels-SRC"] = -1 # If -1, m = n
    


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


