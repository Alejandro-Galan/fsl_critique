
import fire, sys, itertools, os, json

sys.path.append("./")

# import scripts.one_run_network as run



import importlib
from utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp_num = str(sys.argv[1])


print("Type num. experiment to test:")
if len(sys.argv) < 2:
    print("No experiment number provided")
    exit()


#############################################################
global OVERWRITE_LOGS 
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
    # sub_params["INPUT_SIZE"] = (28, 28)
    sub_params["ReusePretrained"] = True
    sub_params["SAMPLES_PER_CLASS"] = [1]
    sub_params["MODEL_TYPE"] = ["PrototypicalNetwork"]
    sub_params["GROUP_EXPERIMENT"] = "DEBUG_FAST"
    
    return sub_params


############################################################
###### Exp1. Set of source permutation experiments #########
############################################################


def filter_n_way_datasets(allMoreParams, source_datasets, target_datasets):
    to_cluster_kmeans = False
    for moreParams in allMoreParams:
        if "CLUSTERING" in moreParams and moreParams["CLUSTERING"] == "KMEANS":
            to_cluster_kmeans = True
    for moreParams in allMoreParams:
        if "LIMIT_N_WAY_TRAIN" in moreParams and moreParams["LIMIT_N_WAY_TRAIN"] > 18:
            banned = [Const_c.DATASETS.EGYPTIAN.value, "miniImageNet_SOTA_trainSet", 
                      Const_c.DATASETS.BREAKHIS.value, "cifar100_SOTA_trainSet",
                      Const_c.DATASETS.ORGANAMNIST.value]
            if to_cluster_kmeans:
                banned = ["miniImageNet_SOTA_trainSet", "cifar100_SOTA_trainSet"]
            for b in banned:
                if b in source_datasets:
                    source_datasets.remove(b)
        if "LIMIT_N_WAY_TEST" in moreParams and moreParams["LIMIT_N_WAY_TEST"] > 18:
            banned = [Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.EGYPTIAN.value, 
                    "miniImageNet_SOTA_testSet", Const_c.DATASETS.BREAKHIS.value,
                    "cifar100_SOTA_testSet", Const_c.DATASETS.ORGANAMNIST.value]
            for b in banned:
                if b in target_datasets:
                    target_datasets.remove(b)

        # Not all datasets for 10 samples per class
        if "SAMPLES_PER_CLASS" in moreParams and 10 in moreParams["SAMPLES_PER_CLASS"]:
            if len(moreParams["SAMPLES_PER_CLASS"]) > 1:
                print("WARNING: Removing some datasets because 10 spc is in the list")
            banned = ["omniglot_SOTA_trainSet", "omniglot_SOTA_testSet",
                      "miniImageNet_SOTA_trainSet", "miniImageNet_SOTA_testSet", 
                      Const_c.DATASETS.BREAKHIS.value, 
                      "cifar100_SOTA_trainSet", "cifar100_SOTA_testSet"]
            for b in banned:
                if b in source_datasets:
                    source_datasets.remove(b)
                if b in target_datasets:
                    target_datasets.remove(b)

    return source_datasets, target_datasets


def  common_execute_experiment(exp_name, exp_name_sufix, group_exp_name, sub_params, PARAMS):

    for model_type in PARAMS['MODEL_TYPE']:   
        for spc in PARAMS['SAMPLES_PER_CLASS']:   
            sub_params['MODEL_TYPE'] = model_type
            sub_params['SAMPLES_PER_CLASS'] = spc
            
            string_id_base, string_id_ft = Const_c.get_experiment_id(PARAMS=sub_params, boots_iter=sub_params["BOOTSTRAP_ITERS"] - 1) ## Assure the last iteration planned is done               
            full_exp_name = exp_name + exp_name_sufix     

            Constants_c = Const_c(exp_num="exp" + exp_num, full_name=string_id_ft, immutable=False, subparams=sub_params)

            # Concurrency:
            if Constants_c.exists_file_constants():
                print("Skipping " + full_exp_name + " as it already exists and it is supposed to be executing right now.")
                return
            
            Constants_c.modify_constants(**sub_params)

            try:
                execute_experiment(full_exp_name, Constants=Constants_c.Constants, group_exp_name=group_exp_name)
                # Free the file (log is supposed to be created, so no repetition)
                Constants_c.remove_constants_file()
            except Exception as e:
                # Free the file (log is supposed to be created, so no repetition)
                Constants_c.remove_constants_file()
                print("Error in experiment:", full_exp_name)
                print(e)
                breakpoint()


def core_exp_1_2(target_datasets, executed_permutations, PARAMS_TO_MODIFY, exp_name, group_exp_name, source_datasets):
        
    for r in range(1, len(source_datasets) + 1):
        all_permutations = itertools.combinations(source_datasets, r)    
        for perm in all_permutations:   
            src_datasets = [p for p in perm]
            for tgt_dataset in target_datasets:
                ## Skip if the source and target are the same
                if len(src_datasets) == 1: 
                    if src_datasets[0] == tgt_dataset:
                        continue

                    if group_exp_name != "exp4":
                        # In case of omniglot, miniImageNet and cifar100
                        if src_datasets[0][:7] == tgt_dataset[:7]:
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

                exp_name_sufix = "-src_ds_" + "__".join(real_src)

                common_execute_experiment(exp_name, exp_name_sufix, group_exp_name, sub_params, PARAMS=PARAMS_TO_MODIFY)



def exp1_2_5_run_source_permutation_experiments(moreParams=[], exp_name="", group_exp_name="", custom_dsts=None):
    path_exp = "logs_csv/last_exec_" + group_exp_name 
    if OVERWRITE_LOGS:
        os.system("rm -f " + path_exp + "/*")
    else:
        print("TRAINING CONTINUES EXPERIMENT, NOT OVERWRITING")
    os.makedirs(path_exp, exist_ok=True)


    source_datasets = ["omniglot_SOTA_trainSet", Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.EGYPTIAN.value, Const_c.DATASETS.TKH.value, Const_c.DATASETS.GREEK.value, 
                        "miniImageNet_SOTA_trainSet", "cifar100_SOTA_trainSet", Const_c.DATASETS.ORGANAMNIST.value] #, Const_c.DATASETS.BREAKHIS.value]


    target_datasets = [Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.EGYPTIAN.value, Const_c.DATASETS.TKH.value, Const_c.DATASETS.GREEK.value, 
                        "omniglot_SOTA_testSet","miniImageNet_SOTA_testSet", "cifar100_SOTA_testSet", Const_c.DATASETS.ORGANAMNIST.value] #, Const_c.DATASETS.BREAKHIS.value]



    #################################################
    ## DEBUG ##
    #################################################
    # if group_exp_name == "exp1":
    #     # if True:
    #     source_datasets = [Const_c.DATASETS.ORGANAMNIST.value]

        # target_datasets = [Const_c.DATASETS.CAPITAN.value]
    #################################################




    # Egypt and BreaKHis are too small to have validation
    if group_exp_name == "exp2":
        source_datasets = [Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.TKH.value, Const_c.DATASETS.GREEK.value, 
                            "omniglot_SOTA_trainSet","miniImageNet_SOTA_trainSet"]

        target_datasets = [Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.TKH.value, Const_c.DATASETS.GREEK.value, 
                            "omniglot_SOTA_testSet","miniImageNet_SOTA_testSet"]

    source_datasets, target_datasets = filter_n_way_datasets(moreParams, source_datasets, target_datasets)


    PARAMS_TO_MODIFY = Const_c.BASE_PARAMETERS.copy()
    PARAMS_TO_MODIFY["GROUP_EXPERIMENT"] = "source_permutation"


    PARAMS_TO_MODIFY["LIMIT_N_WAY_TRAIN"] = 5
    PARAMS_TO_MODIFY["LIMIT_N_WAY_TEST"] = 5    

    PARAMS_TO_MODIFY["BOOTSTRAP_ITERS"] = 2
    # PARAMS_TO_MODIFY["epochsFineTuning"] = 10
    PARAMS_TO_MODIFY["OVERWRITE_LOGS"] = OVERWRITE_LOGS
    
    for p in moreParams:
        PARAMS_TO_MODIFY[[*p.keys()][0]] = [*p.values()][0]


    # Each src dataset combination
    if not custom_dsts:
        executed_permutations = []
        core_exp_1_2(target_datasets, executed_permutations, PARAMS_TO_MODIFY, exp_name, group_exp_name, source_datasets)
    # Exp 5: Hyperparameters
    # else:
    #     for spc, dict_src_tgt in custom_dsts.items():
    #         for src_dataset, tgt_dataset in dict_src_tgt.items():
    #             src_d = [src_dataset]
    #             print("\nTesting on source:", src_dataset, "\n")
    #             sub_params = PARAMS_TO_MODIFY.copy() 
    #             sub_params["TGT_DATASETS"] = {tgt_dataset:{}}
    #             sub_params["DATASETS_NAMES"] = src_d

    #             sub_params["SAMPLES_PER_CLASS"] = [spc]
                    
    #             ################# DEBUG ##################
    #             # sub_params = debug_FAST_CONFIG(sub_params)
    #             ##########################################    

    #             exp_name_sufix = "-src_ds_" + "__".join(src_d)
    #             common_execute_experiment(exp_name, exp_name_sufix, group_exp_name, sub_params)


def exp3_4__run_source_permutation_experiments(moreParams=[], exp_name="", group_exp_name=""):
    path_exp = "logs_csv/last_exec_" + group_exp_name 
    
    if OVERWRITE_LOGS:
        os.system("rm -f " + path_exp + "/*")
    else:
        print("TRAINING CONTINUES EXPERIMENT, NOT OVERWRITING")
    os.makedirs(path_exp, exist_ok=True)


    target_datasets = ["omniglot_SOTA_testSet", Const_c.DATASETS.CAPITAN.value, Const_c.DATASETS.EGYPTIAN.value, Const_c.DATASETS.TKH.value, Const_c.DATASETS.GREEK.value, 
                       "miniImageNet_SOTA_testSet", "cifar100_SOTA_testSet", Const_c.DATASETS.ORGANAMNIST.value] # Const_c.DATASETS.BREAKHIS.value,

    ################## DEBUG ##################
    # if group_exp_name == "exp3":
    #     target_datasets = [ Const_c.DATASETS.TKH.value ] 
    ###########################################

    source_datasets, target_datasets = filter_n_way_datasets(moreParams, {}, target_datasets)
    print("TARGET DATASETS:", target_datasets)



    PARAMS_TO_MODIFY = Const_c.BASE_PARAMETERS.copy()
    PARAMS_TO_MODIFY["GROUP_EXPERIMENT"] = "source_permutation"


    PARAMS_TO_MODIFY["LIMIT_N_WAY_TRAIN"] = 5
    PARAMS_TO_MODIFY["LIMIT_N_WAY_TEST"] = 5   
    PARAMS_TO_MODIFY["BOOTSTRAP_ITERS"] = 2
    PARAMS_TO_MODIFY["OVERWRITE_LOGS"] = OVERWRITE_LOGS

    for p in moreParams:
        PARAMS_TO_MODIFY[[*p.keys()][0]] = [*p.values()][0]

    
    for tgt_dataset in target_datasets:
                

        print("\nTesting on tgt:", tgt_dataset, "\n")
        sub_params = PARAMS_TO_MODIFY.copy() 
        sub_params["TGT_DATASETS"] = {tgt_dataset:{}}
        sub_params["DATASETS_NAMES"] = {"NO_SRC_DATASET":{}}
        
        # Exception as they use that prefixed supp_sets
        if group_exp_name in ["exp4", "exp7"]:
            if tgt_dataset == "miniImageNet_SOTA_testSet":
                sub_params["DATASETS_NAMES"] = {"miniImageNet_SOTA_trainSet":{}}
            if tgt_dataset == "omniglot_SOTA_testSet":
                sub_params["DATASETS_NAMES"] = {"omniglot_SOTA_trainSet":{}}
            if tgt_dataset == "cifar100_SOTA_testSet":
                sub_params["DATASETS_NAMES"] = {"cifar100_SOTA_trainSet":{}}
                

        ################# DEBUG ##################
        # sub_params = debug_FAST_CONFIG(sub_params)
        ##########################################    
                        
        if group_exp_name == "exp3":
            exp_name_sufix = "-src_ds_" + "__NO_SRC_DATASET"
        else:    
            exp_name_sufix = "-src_ds_" + tgt_dataset


        common_execute_experiment(exp_name, exp_name_sufix, group_exp_name, sub_params, PARAMS=PARAMS_TO_MODIFY)


# It will depend on the value of the constants.json
def execute_experiment(exp_name, Constants, group_exp_name=""):
    # importlib.reload(run)
    # run.main(exp_name, group_exp_name=group_exp_name)
    string_id_base, string_id_ft = Const_c.get_experiment_id(PARAMS=Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1) ## Assure the last iteration planned is done

    print("python3 scripts/one_run_network.py " + exp_name + " " + group_exp_name + " " + string_id_ft)
    os.system("python3 scripts/one_run_network.py " + exp_name + " " + group_exp_name + " " + string_id_ft)

# machine = socket.gethostname()
# if machine == "bilbo":
#     NWAY = 5 
# elif machine == "multiscore":
#     NWAY = 20
# else:
#     print("NOT CONFIGURED IN THIS MACHINE YET", machine)
#     exit()

basic_params_file = "hyperparameters_experiments.json"
# Load the constants from the file
if os.path.exists(basic_params_file):
    with open(basic_params_file, 'r') as f:
        basic_params = json.load(f)
else:
    print(f"File {basic_params_file} not found. Please create it with the necessary parameters.")
    exit()  

# NWAY = 5
NWAY = basic_params["NWAY"][exp_num] 
# ### Some datasets banned if 10 in SPC_OW
# SPC_OW  = [1, 5] #[10] #[1, 5]
SPC_OW = basic_params["SPC_OW"][exp_num] 

BOOTSTRAP_ITERS = basic_params["BOOTSTRAP_ITERS"][exp_num]

# MODEL_TS = ["PrototypicalNetwork"] #["RelationNetwork"] # ["PrototypicalNetwork", "MatchingNetwork"]
MODEL_TS = basic_params["MODEL_TS"][exp_num] 

if __name__ == "__main__":
    if exp_num == "1":
        ## Exp 1. permutation src datasets
        print("Running experiment 1: Permuted source datasets w/o validation")
        # use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"ReusePretrained": False} ]
        use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
        exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp1_permuted_src_datasets", group_exp_name="exp1")
    
    if exp_num == "2":
        ## Exp 2. permutation src datasets with validation
        print("Running experiment 2: Permuted source datasets with validation")
        use_validation_params = [{"VALIDATION_SRC_SRC_DATA": True}, {"VALIDATION_PERC": 0.2}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS}]
        exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp2_permuted_src_datasets_WithValidation", group_exp_name="exp2")

    if exp_num == "3":
        ## Exp 3. No source training  only Fine tuning with validation
        print("Running experiment 3: No source training only ft")
        # if "PrototypicalNetwork" in MODEL_TS:
        #     BOOTSTRAP_ITERS = 10

        use_validation_params = [{"NoSrcDataset": True}, {"epochsFineTuning": 3}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS}]

        if "EXHAUSTIVE_LOSS_CURVES" in basic_params:
            use_validation_params.append({"EXHAUSTIVE_LOSS_CURVES": basic_params["EXHAUSTIVE_LOSS_CURVES"]})

        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp3_no_src_datasets", group_exp_name="exp3")

    if exp_num == "4":
        ## Exp 4. Division of classes among the same dataset with validation
        print("Running experiment 4: All training over the same dataset")
        # {"VALIDATION_SRC_SRC_DATA": True}, , {"VALIDATION_PERC": 0.2}, # Seems to work better without
        use_validation_params = [{"ALL_DATASETS": False}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS}]
        # use_validation_params = [{"ALL_DATASETS": False}, {"BOOTSTRAP_ITERS": 1}, {"epochsFineTuning": 1}, ]
        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp4_same_dataset", group_exp_name="exp4")

    ### Experiment prepared for tests only, not final paper-experiments
    if exp_num == "5":
        # os.system("rm -r utils/stored_sets")
        OVERWRITE_LOGS = True
        # ## Exp 4. Division of classes among the same dataset with validation
        print("Running experiment 4.5 baseline with repetition to compare: All training over the same dataset")
        use_validation_params = [{"ALL_DATASETS": False}, {"BOOTSTRAP_ITERS": 1}, {"epochsFineTuning": 1}, {"MODEL_TYPE": MODEL_TS}  ]
        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp5_4_rep_test_same_dataset", group_exp_name="exp5")

        ## Exp 1. Proposal
        # print("Running experiment 1.5 nosrc")
        # use_validation_params = [{"NoSrcDataset": True}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 1}  ]
        # exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp5_premute_datasets", group_exp_name="exp5")


        ## Exp3. Random weights
        # print("Running experiment 3.5 baseline with repetition to compare: All training over the same dataset")
        # use_validation_params = [{"ALL_DATASETS": False}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS},  {"epochsFineTuning": 1} ]
        # exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp5_noInit", group_exp_name="exp5")

        # ## Exp 5. Change of Hyperparametesrs    
        # input_sizes = [ (22, 22), (28, 28), (84, 84), (126,126)] # (40, 40) already executed on exp1
        # for i_s in input_sizes:
        #     # use_validation_params = [{"INPUT_SIZE": i_s}]
        #     print("Running experiment 5: Hyperparameters over best previous cases")
        #     print("INPUT SIZE:", i_s)
        #     exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp5_hyperparameters", group_exp_name="exp5", custom_dsts=custom_datasets_best_cases)

    if exp_num == "6":
        ## Exp 6.     
        print("Running experiment 6: DEBUG")
        exp1_2_5_run_source_permutation_experiments(exp_name="exp6_DEBUG", group_exp_name="exp6")

        # m_ = 22
        ## Debug for Exp 9. Clustering permutation src datasets
        # print("Running experiment 9: Permuted source datasets w/o validation") # {"CLUSTERING": "KMEANS"}, {"M-labels-SRC": m_},
        # use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"epochsFineTuning": 3}, {"CLUSTERING": "CONSTRAINED-KMEANS"}, {"M-labels-SRC": m_},
        #                         {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
        # exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp6_cluster_kmeans_m_" + str(m_), group_exp_name="exp6")
        # exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp6_like1", group_exp_name="exp6")


    if exp_num == "7":
        ## Exp 7.     
        print("Running experiment 7: Variation of 4 more subtle FT")
        ## By default parameters: FINE_TUNING_EPISODES = 1000; epochsFineTuning = 3
        # use_validation_params = [{"ALL_DATASETS": False}, {"FINE_TUNING_EPISODES": 16}, 
        #                          {"epochsFineTuning" : 187} ] # Original / batch size as a minimum
        use_validation_params = [{"ALL_DATASETS": False}, {"FINE_TUNING_EPISODES": 16}, 
                                 {"epochsFineTuning" : 25}, {"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"MODEL_TYPE": MODEL_TS} ] # Original / batch size as a minimum
        

        exp3_4__run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp7s_FT", group_exp_name="exp7")

    ### TODO Same as 7 but for exp1
    if exp_num == "8":
        ## Exp 8.     
        print("Running experiment 8: Variation of 1 more subtle FT")
        use_validation_params = [{"FINE_TUNING_EPISODES": 16}, 
                                 {"epochsFineTuning" : 25}, {"BOOTSTRAP_ITERS": 2}, {"MODEL_TYPE": MODEL_TS} ] # Original / batch size as a minimum
        

        exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp8s_1FT", group_exp_name="exp8")

    ## Clustering. Just like exp1 but clustering the src dataset
    # if exp_num == "9":
    #     ## Exp 9. Clustering permutation src datasets
    #     print("Running experiment 9: Permuted source datasets w/o validation")
    #     use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"CLUSTERING": "KMEANS"}, {"M-labels-SRC": -1}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
    #     exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp9_cluster_kmeans_m_eq_n", group_exp_name="exp9")

    #     if NWAY == 5:
    #         new_m = [25, 250, 1000] 
    #     elif NWAY == 20:
    #         new_m = [250, 1000]

    #     for m_ in new_m:
    #         ## Exp 9. Clustering permutation src datasets
    #         print("Running experiment 9: Permuted source datasets w/o validation")
    #         use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"CLUSTERING": "KMEANS"}, {"M-labels-SRC": m_}, 
    #                                 {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
    #         exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp9_cluster_kmeans_m_" + str(m_), group_exp_name="exp9")

    ## Clustering. Just like exp1 but clustering the src dataset constrained
    if exp_num == "10":
        ## Exp 10. Clustering permutation src datasets
        # print("Running experiment 9: Permuted source datasets w/o validation")
        # use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"CLUSTERING": "CONSTRAINED-KMEANS"}, {"M-labels-SRC": -1}, {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
        # exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp9_cluster_constr_kmeans_m_eq_n", group_exp_name="exp10")

        if NWAY == 5:
            new_m = [5, 20, 250, 1000] 
        elif NWAY == 20:
            new_m = [20, 250, 1000]

        for m_ in new_m:
            ## Exp 9. Clustering permutation src datasets
            print("Running experiment 9: Permuted source datasets w/o validation")
            use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"CLUSTERING": "CONSTRAINED-KMEANS"}, {"M-labels-SRC": m_}, 
                                    {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
            exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp9_cluster_constr_kmeans_m_" + str(m_), group_exp_name="exp10")


    ## Set of concrete experiments
    if exp_num == "11":

        ## Set of exp1:
        MODEL_TS = ["PrototypicalNetwork", "MatchingNetwork"] #["RelationNetwork"] # ["PrototypicalNetwork", "MatchingNetwork"]
        exc_pars = [ {"DATASETS_NAMES": "omniglot_SOTA_trainSet", "TGT_DATASETS": {Const_c.DATASETS.GREEK.value:{}}, "nway": 20, "spc": [5]},
                     {"DATASETS_NAMES": "omniglot_SOTA_trainSet", "TGT_DATASETS": {Const_c.DATASETS.GREEK.value:{}}, "nway": 20, "spc": [10]},
                     ]

        for pars in exc_pars: 
            print("Running experiment 1: Permuted source datasets w/o validation")
            # use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"ReusePretrained": False} ]
            use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"LIMIT_N_WAY_TRAIN": pars["nway"]}, {"LIMIT_N_WAY_TEST": pars["nway"]}, {"SAMPLES_PER_CLASS": pars["spc"]}, {"MODEL_TYPE": MODEL_TS},
                                     {"DATASETS_NAMES": pars["DATASETS_NAMES"]}, {"TGT_DATASETS": pars["TGT_DATASETS"]} ]
            exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp1_permuted_src_datasets", group_exp_name="exp1")


    ## SimCLR only on query
    if exp_num == "12":

        ## Exp 12. Exp1 with SimCLR
        print("Running experiment 12: Exp1 with SimCLR")
        use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"GROUP_EXPERIMENT": "SIMCLR_only-q_aug"}, {"SIMCLR": "query"}, 
                                 {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
        exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp12_simclr", group_exp_name="exp12")

    ## SimCLR on each support sample. It acts like a class
    if exp_num == "13":

        ## Exp 13. Exp1 with SimCLR on each support sample
        print("Running experiment 13: Exp1 with SimCLR on each support sample")
        use_validation_params = [{"BOOTSTRAP_ITERS": BOOTSTRAP_ITERS}, {"epochsFineTuning": 3}, {"GROUP_EXPERIMENT": "SIMCLR_supp_aug"}, {"SIMCLR": "supportSet"}, 
                                 {"LIMIT_N_WAY_TRAIN": NWAY}, {"LIMIT_N_WAY_TEST": NWAY}, {"SAMPLES_PER_CLASS": SPC_OW}, {"MODEL_TYPE": MODEL_TS} ]
        exp1_2_5_run_source_permutation_experiments(moreParams=use_validation_params, exp_name="exp13_simclr_sup", group_exp_name="exp13")

