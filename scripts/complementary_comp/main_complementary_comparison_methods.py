## Call this script as "python3 scripts/complementary_comparison_methods.py 1 exp1"
import numpy as np
import sys, json
if len(sys.argv) < 2:
    sys.argv[1:] = ["exp1_permuted_src_datasets--src_ds_Greek", "exp1"]

# 'PrototypicalNetwork--4000_Episodes--40x40_INPUT_SIZE/False_FIXED_SUPP_SET--Greek--5_NWAY_TRAIN--16_Batch_Size--1_KSamples_per_Class--0.001_lr--0.0_ValSrc----GROUP_EXP_source_permutation--NWAY-test_5--tgt_dataset_b-59-850']

import fire, os

sys.path.append("./")


from utils.constants import Const_c




ALL_INPUT_SIZES = {
    "b-59-850"                   : (40,40),
    "TKH"                        : (40,40),
    "Egyptian"                   : (40,40),
    "Greek"                      : (40,40),
    "BreaKHis_formatted"         : (84,84),
    
    "omniglot"                   : (28,28),
    "omniglot_SOTA_testSet"      : (28,28),
    "omniglot_SOTA_trainvalSet"  : (28,28),

    "miniImageNet"               : (84,84),
    "miniImageNet_SOTA_testSet"  : (84,84),
    "miniImageNet_SOTA_trainSet" : (84,84),
}

def check_one_file(PARAMS, file, ft):
    with open(file, "r") as f:
        lines = f.readlines()

        ## Indexes : ['DATASETS_NAMES', 'LIMIT_N_WAY_TRAIN', 'LIMIT_N_WAY_TEST', 'MODEL_TYPE', 'BATCH_SIZE', 'SAMPLES_PER_CLASS', 'GROUP_EXPERIMENT', 'Experiment', 'seed_n', 'TGT_DATASETS', 'Fine_Tuned']
        indexes = lines[0].split(",")[1:12]
        for line in lines[1:]:
            ls = line.split(",")
            all_coinc = True

            for i_n, ind in enumerate(indexes):
                if i_n + 1 in [1, 4, 7, 8, 10]:
                    if ls[i_n + 1] != PARAMS[ind]:
                        all_coinc = False
                        break
                elif i_n + 1 == 11:
                    if int(ls[i_n + 1]) != int(ft):
                        all_coinc = False
                        break
                else:
                    if int(ls[i_n + 1]) != (PARAMS[ind]):
                        all_coinc = False
                        break
            if all_coinc:
                return True
            
    return False

def check_if_done(PARAMS, metrics_file, metrics_ft_file):
    if not os.path.exists(metrics_file) or not os.path.exists(metrics_ft_file):
        print("Metrics file does not exist for: ", PARAMS['DATASETS_NAMES'], PARAMS["TGT_DATASETS"])
        return False
    if not check_one_file(PARAMS, metrics_file, ft=False):
        print("Missing pre-ft lines for: ", PARAMS['DATASETS_NAMES'], PARAMS["TGT_DATASETS"])
        return False
    if not check_one_file(PARAMS, metrics_ft_file, ft=True): 
        print("Missing ft lines for: ", PARAMS['DATASETS_NAMES'], PARAMS["TGT_DATASETS"])
        return False
    
    return True


def execute_iter(PARAMS):
    # importlib.reload(run)
    # run.main(exp_name, group_exp_name=group_exp_name)
    string_id_base, string_id_ft = Const_c.get_experiment_id(PARAMS=PARAMS, boots_iter=0) ## Assure the last iteration planned is done

    os.system("python3 scripts/complementary_comp/callable_complementary_comp.py " + sys.argv[1] + " " + sys.argv[2] + " " + string_id_ft)
    

if __name__ == '__main__':
    ## TODO change params depending on dataset. For ex capitan is size 84x84 only for mini and breakhis
    
    PARAMS = Const_c.BASE_PARAMETERS.copy()
    new_PARAMS = {
        'EPISODES': 4000,
        'BATCH_SIZE': 16, 
        'USE_ORIGINAL_FIXED_SUPP_SET': False, 
        'lr': 0.001,
        'VALIDATION_PERC': 0.0,
        ## FT params
        'GROUP_EXPERIMENT': "source_permutation",
        }
    
    for new_p in new_PARAMS:
        PARAMS[new_p] = new_PARAMS[new_p]

    metrics_file, metrics_ft_file = "logs_csv/comparison_embeddings_finetuned.csv", "logs_csv/comparison_embeddings.csv"
    # os.system("rm -f " + metrics_file + " " + metrics_ft_file)

    model_type_ = ["MatchingNetwork", "PrototypicalNetwork"]
    inmutable_datasets = ["b-59-850", "Greek", "TKH", "Egyptian", "BreaKHis_formatted"] 
    src_datasets_ = inmutable_datasets + ["omniglot_SOTA_trainvalSet", "miniImageNet_SOTA_trainSet"]
    TGT_DATASETS_ = inmutable_datasets + ["omniglot_SOTA_testSet", "miniImageNet_SOTA_testSet"]
    LIMIT_N_WAY_TRAIN_ = [5]
    LIMIT_N_WAY_TEST_  = [5]
    samples_per_class_ = [1,5,10]
    seeds_ = np.arange(100)
    seeds_ = [int(seed) for seed in seeds_]
    for seed_n in seeds_:
        PARAMS['seed_n'] = seed_n
        for model_type in model_type_:
            for src_datasets in src_datasets_:
                for LIMIT_N_WAY_TRAIN in LIMIT_N_WAY_TRAIN_:
                    for samples_per_class in samples_per_class_:
                        for LIMIT_N_WAY_TEST in LIMIT_N_WAY_TEST_:
                            ### FT params
                            PARAMS['MODEL_TYPE'] = model_type
                            PARAMS['DATASETS_NAMES'] = {src_datasets: {}}
                            PARAMS['LIMIT_N_WAY_TRAIN'] = LIMIT_N_WAY_TRAIN
                            PARAMS['SAMPLES_PER_CLASS'] = samples_per_class
                            PARAMS['LIMIT_N_WAY_TEST'] = LIMIT_N_WAY_TEST

                            ## Loop over different datasets
                            for tgt_datasets in TGT_DATASETS_:
                                if tgt_datasets[:3] == src_datasets[:3]:
                                    continue
                                PARAMS['TGT_DATASETS'] = {tgt_datasets: {}}

                                # Export dict params to file
                                with open("scripts/complementary_comp/PARAMS_NOT_PARALLELIZABLE.json", "w") as f:
                                    json.dump(PARAMS, f)

                                # Check if it is already done
                                if not check_if_done(PARAMS, metrics_file, metrics_ft_file):
                                    execute_iter(PARAMS)
