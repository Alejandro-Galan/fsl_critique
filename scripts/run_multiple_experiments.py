
import fire, sys, wandb, itertools, os, importlib

sys.path.append("./")

import scripts.small_run_matching_networks as run



from my_utils.constants import Const_c
# Initialize reading the json constants file for each experiment

exp = int(sys.argv[1])
Constants_c = Const_c(exp)
Constants = Constants_c.Constants


print("Type num. experiment to test:")
if len(sys.argv) < 2:
    print("No experiment number provided")
    exit()


#############################################################
OVERWRITE_LOGS = False #False # Else, continue on the next one
if OVERWRITE_LOGS:
    input("OVERWRITING LOGS, are you sure?. Press Enter to continue or Ctrl+C to cancel...")
#############################################################




# Important to import the functions after the constants are modified
def debug_FAST_CONFIG(sub_params):
    miniep = 10
    sub_params["EPISODES"] = miniep
    sub_params["BOOTSTRAP_ITERS"] = 1
    sub_params["epochsFineTuning"] = 1
    sub_params["SMALL_TEST_SET"] = True
    sub_params["LIMIT_N_WAY_TRAIN"] = 2
    sub_params["LIMIT_N_WAY_TEST"] = 2
    sub_params["BATCH_SIZE"] = 8
    sub_params["LIMIT_VALIDATION_SRC_SRC"] = miniep - 1
    sub_params["INPUT_SIZE"] = (28, 28)
    sub_params["ReusePretrained"] = True
    sub_params["SAMPLES_PER_CLASS"] = [1]
    sub_params["MODEL_TYPE"] = ["PrototypicalNetwork"]
    sub_params["GROUP_EXPERIMENT"] = ["DEBUG FAST"]
    
    return sub_params


############################################################
###### Exp1. Set of source permutation experiments #########
############################################################

def exp1_2__run_source_permutation_experiments(moreParams=[], exp_name="", group_exp_name=""):
    path_exp = "logs_csv/last_exec_" + group_exp_name 
    if OVERWRITE_LOGS:
        os.system("rm -f " + path_exp + "/*")
    else:
        print("TRAINING CONTINUES EXPERIMENT, NOT OVERWRITING")
    os.makedirs(path_exp, exist_ok=True)

    source_datasets = ["b-59-850", "Egyptian", "TKH", "Greek", 
                    "omniglot_SOTA_trainvalSet","miniImageNet_SOTA_trainSet"]

    target_datasets = ["b-59-850", "Egyptian", "TKH", "Greek", 
                       "omniglot_SOTA_testSet", "miniImageNet_SOTA_testSet"]
    PARAMS_TO_MODIFY = {}
    for p in moreParams:
        PARAMS_TO_MODIFY[[*p.keys()][0]] = [*p.values()][0]
    PARAMS_TO_MODIFY["GROUP_EXPERIMENT"] = "source_permutation"


    PARAMS_TO_MODIFY["LIMIT_N_WAY_TRAIN"] = 5
    PARAMS_TO_MODIFY["LIMIT_N_WAY_TEST"] = 5   

    PARAMS_TO_MODIFY["BOOTSTRAP_ITERS"] = 2
    PARAMS_TO_MODIFY["epochsFineTuning"] = 10
    PARAMS_TO_MODIFY["OVERWRITE_LOGS"] = OVERWRITE_LOGS


    executed_permutations = []
    # Each src dataset combination
    for r in range(1, len(source_datasets) + 1):
        all_permutations = itertools.combinations(source_datasets, r)
        for perm in all_permutations:   

            src_datasets = [p for p in perm]
            for tgt_dataset in target_datasets:
                ## Skip if the source and target are the same
                if len(src_datasets) == 1 and src_datasets[0] == tgt_dataset:
                    continue
                
                real_src = [s for s in src_datasets if s != tgt_dataset]
                
                ################# DEBUG #################################
                ### I do not want to test multiple source datasets at this moment
                if len(real_src) > 1: 
                    continue
                ################# DEBUG #################################

                comb = (real_src, tgt_dataset)
                if comb in executed_permutations:
                    continue

                executed_permutations.append(comb)


                print("\nTesting on source:", real_src, "\n")
                sub_params = PARAMS_TO_MODIFY.copy() 
                sub_params["TGT_DATASETS"] = {tgt_dataset:{}}
                sub_params["DATASETS_NAMES"] = real_src
                
                ################# DEBUG ##################
                # sub_params = debug_FAST_CONFIG(sub_params)
                ##########################################    
                             
                Constants_c.modify_constants(**sub_params, group_exp_name=group_exp_name)
                exp_name_sufix = "--src_ds_" + "__".join(real_src)

                try:
                    execute_experiment(exp_name + exp_name_sufix, group_exp_name=group_exp_name)
                except Exception as e:
                    print("Error in experiment:", exp_name + exp_name_sufix)
                    print(e)
                    breakpoint()

def exp3_4__run_source_permutation_experiments(moreParams=[], exp_name="", group_exp_name=""):
    path_exp = "logs_csv/last_exec_" + group_exp_name 
    
    if OVERWRITE_LOGS:
        os.system("rm -f " + path_exp + "/*")
    else:
        print("TRAINING CONTINUES EXPERIMENT, NOT OVERWRITING")
    os.makedirs(path_exp, exist_ok=True)

    target_datasets = ["b-59-850", "Egyptian", "TKH", "Greek", 
                       "BreakHis"]
    PARAMS_TO_MODIFY = {}
    for p in moreParams:
        PARAMS_TO_MODIFY[[*p.keys()][0]] = [*p.values()][0]
    PARAMS_TO_MODIFY["GROUP_EXPERIMENT"] = "source_permutation"


    PARAMS_TO_MODIFY["LIMIT_N_WAY_TRAIN"] = 5
    PARAMS_TO_MODIFY["LIMIT_N_WAY_TEST"] = 5   

    PARAMS_TO_MODIFY["BOOTSTRAP_ITERS"] = 2
    PARAMS_TO_MODIFY["OVERWRITE_LOGS"] = OVERWRITE_LOGS


    
    for tgt_dataset in target_datasets:
                

        print("\nTesting on tgt:", tgt_dataset, "\n")
        sub_params = PARAMS_TO_MODIFY.copy() 
        sub_params["TGT_DATASETS"] = {tgt_dataset:{}}
        sub_params["DATASETS_NAMES"] = {"NO_SRC_DATASET":{}}
        
        ################# DEBUG ##################
        # sub_params = debug_FAST_CONFIG(sub_params)
        ##########################################    
                        
        Constants_c.modify_constants(**sub_params, group_exp_name=group_exp_name)
        if group_exp_name == "exp3":
            exp_name_sufix = "--src_ds_" + "__NO_SRC_DATASET"
        else:    
            exp_name_sufix = "--src_ds_" + tgt_dataset

        try:
            execute_experiment(exp_name + exp_name_sufix, group_exp_name=group_exp_name)
        except Exception as e:
            print("Error in experiment:", exp_name + exp_name_sufix)
            print(e)
            breakpoint()


# It will depend on the value of the constants.json
def execute_experiment(exp_name,group_exp_name=""):
    importlib.reload(run)

    run.main(exp_name, group_exp_name=group_exp_name)

if __name__ == "__main__":
    if exp == 1:
        ## Exp 1. permutation src datasets
        print("Running experiment 1: Permuted source datasets w/o validation")
        exp1_2__run_source_permutation_experiments(exp_name="exp1_permuted_src_datasets", group_exp_name="exp1")
    
    if exp == 2:
        ## Exp 2. permutation src datasets with validation
        print("Running experiment 2: Permuted source datasets with validation")
        use_validation_params = [{"VALIDATION_SRC_SRC_DATA": True}, {"VALIDATION_PERC": 0.2}]
        exp1_2__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp2_permuted_src_datasets_WithValidation", group_exp_name="exp2")

    if exp == 3:
        ## Exp 3. No source training  only Fine tuning with validation
        print("Running experiment 3: No source training only ft")
        use_validation_params = [{"NoSrcDataset": True}, {"epochsFineTuning": 15}]
        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp3_no_src_datasets", group_exp_name="exp3")

    if exp == 4:
        ## Exp 4. Division of classes among the same dataset with validation
        print("Running experiment 4: All training over the same dataset")
        use_validation_params = [{"ALL_DATASETS": False}, {"VALIDATION_SRC_SRC_DATA": True}, {"VALIDATION_PERC": 0.2}, ]
        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp4_same_dataset", group_exp_name="exp4")
