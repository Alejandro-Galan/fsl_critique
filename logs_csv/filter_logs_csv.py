# Filter data by columns and unify in a bigger csv
# And show main statistics

import os, glob, math
import pandas as pd
import numpy as np

from enum import Enum

expected_col_counts = {"exp1-like": [840, 840, 300, 135, 135, 60], 
                       "exp4-like": [120, 120, 75, 45, 45, 30]}


class DATASETS(Enum): 
    CAPITAN  = 'b-59-850'
    TKH      = 'TKH'
    BREAKHIS = 'BreaKHis_formatted' 
    EGYPTIAN = 'Egyptian'
    GREEK    = 'Greek'

def extract_metadata_from_namefile(f_name):
    metadata = {}
    metadata["exp_name"] = f_name.split('-')[0]
    metadata["tgt-ds"] = f_name.split('-')[1]
    metadata["src-dss"] = f_name.split('src_ds_')[1].split('tgt_ds-')[0].split("_")
    metadata["src-dss"] = [s for s in metadata["src-dss"] if s != '']

    return metadata


def check_complete_executions(exp_name, df, exhaustive=False):

    df = df[df['bootstrap_iter'] < 5]

    # group_df = df.groupby(['model_type', 'n_way_test', 'n_way_train', 'tgt_dataset', 'src_datasets','samples_per_class', 'num_total_episodes', 'num_bootstrap_iters', 'fixed_support','exp_name', 'CLUSTERING', 'M-labels-SRC', 'bootstrap_iter']) #, 'ft_eval_num'])
    # cols_to_group = ['n_way_test', 'samples_per_class', 'exp_name', 'bootstrap_iter', 'n_way_train', 'num_total_episodes', 'CLUSTERING', 'M-labels-SRC', 'ft_eval_num']
    if not exhaustive:
        cols_to_group = ['n_way_test', 'samples_per_class', 'exp_name', "lr","batch_size","input_size","fixed_support", "num_total_episodes","exp_name", 'CLUSTERING', 'M-labels-SRC', "ft_eval_num"]
    else:
        cols_to_group = ['n_way_test', 'samples_per_class', 'exp_name', "lr","batch_size",'tgt_dataset', 'src_datasets',"input_size","fixed_support", "num_total_episodes","exp_name", 'CLUSTERING', 'M-labels-SRC', "ft_eval_num"]

    # if not "exp1_" in exp_name:
    #     continue
    if "exp3_" in exp_name or "exp4_" in exp_name:
        expected_cc = expected_col_counts["exp4-like"]
    else:
        expected_cc = expected_col_counts["exp1-like"]

    group_df = df.groupby(cols_to_group) #, 'ft_eval_num'])
    counts = group_df.count()
    for c_ in counts.index:
        # ft_eval_num
        if c_[-1] != 2:
            continue
        # # Only 5 bootstraps
        # if c_[3] > 4:
        #     continue 
        line = counts.loc[c_]
        limit = -1
        # nway
        if c_[0] == 5:
            if c_[1] == 1:
                limit = expected_cc[0]
            elif c_[1] == 5:
                limit = expected_cc[1]
            elif c_[1] == 10:
                limit = expected_cc[2]
        else:
            if c_[1] == 1:
                limit = expected_cc[3]
            elif c_[1] == 5:
                limit = expected_cc[4]
            elif c_[1] == 10:
                limit = expected_cc[5]
        if limit == -1:
            # print("OTHER SPC", c_)
            continue    
        if exhaustive:
            limit = 15
        # limit = int(limit / 5)
        obtained_c = line['ft_eval_acc']
        
        if obtained_c < limit:

            if not exhaustive:
                print("\nFor experiment: ", exp_name, "spc", c_[1], "n-way", c_[0])
                # print("DIFFERENT FOR", cols_to_group, "\n", c_)
                check_complete_executions(exp_name, df, exhaustive=True)
            else:
                for model in ['RelationNetwork', "MatchingNetwork", "PrototypicalNetwork"]:
                    path_file = "logs_csv/last_exec_exp" + str(c_[10]) + "/" + str(exp_name) + "-"
                    src_ds = str(c_[6]) if c_[10] != 4 else str(c_[6]).replace("train", "test")   
                    path_file += "src_ds_" + src_ds + "_tgt_ds-" + str(c_[5])
                    path_file += "-spc_" + str(c_[1]) + "-model-" + model
                    path_file += "-" + str(c_[0]) + "-nwayTrain_.csv"
                    
                    print(obtained_c, "!=", limit, path_file)
                    # Check for each model
                    df_filtered = df.copy()
                    for idx, col in enumerate(cols_to_group):
                        df_filtered = df_filtered[df_filtered[col] == c_[idx]]
                    df_filtered = df_filtered[df_filtered["model_type"] == model]
                    
                    if len(df_filtered) < 5 and os.path.exists(path_file):
                        os.remove(path_file)
        else:
            # print("CORRECT")
            pass







def error_ft_eval_iters(counter, path_file, df):
    if os.path.exists(path_file):
        if counter < 3: # Minimumn stablished limit of fune_tuning evaluations
            print("Not enough fine_tuning iterations")
            # Move file to folder outside
            new_path = os.path.join("discarded_logs/not_enough_bootstrap_iters", path_file )
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # if "Egyptian" in path_file and "NO_SRC_DATASET" in path_file:
            #     if "PrototypicalNetwork" in path_file and "spc_1-" in path_file and "5-nway" in path_file:
            #         breakpoint()

            os.rename(path_file, new_path)



def extend_and_remove_non_relevant_rows(df, path_file, exp_num):   
    if 'tr lr' in df.columns:
       df['tr lr'] = df['tr lr'].ffill()
       df['tr lr'] = df['tr lr'].bfill()
    else:
        df['tr lr'] = ""
    df['lr'] = df['lr'].ffill()
    df['lr'] = df['lr'].bfill()

    if 'tr_acc' in df.columns:
        df['tr_acc'] = df['tr_acc'].ffill()
        df['tr_acc'] = df['tr_acc'].bfill()
    else:
        df['tr_acc'] = ""
       
    df['ft_acc'] = df['ft_acc'].ffill()
    df['ft_acc'] = df['ft_acc'].bfill()

    if 'tr_eval_acc_src_data' in df.columns:
        df['tr_eval_acc_src_data'] = df['tr_eval_acc_src_data'].ffill()
        df['tr_eval_acc_src_data'] = df['tr_eval_acc_src_data'].bfill()
    else:
        df['tr_eval_acc_src_data'] = ""

    if 'before_ft_eval_acc' in df.columns:
        size_offset = math.ceil(len(df['bootstrap_iter']) / 5)

        df['before_ft_eval_acc'] = df['before_ft_eval_acc'].ffill()
        df['before_ft_eval_acc'] = df['before_ft_eval_acc'].bfill()
    else:
        print("Not before_ft_eval_acc column")
        # if "Egyptian" in path_file and "NO_SRC_DATASET" in path_file:
        #     if "PrototypicalNetwork" in path_file and "spc_1-" in path_file and "5-nway" in path_file:
        #         breakpoint()

        new_path = os.path.join("discarded_logs/not_before_ft_eval_acc", path_file )
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.rename(path_file, new_path)
        return df, "Not before_ft_eval_acc column for path: " + path_file

    if 'M-labels-SRC' in df.columns:
        df['M-labels-SRC'] = df['M-labels-SRC'].ffill()
        df['M-labels-SRC'] = df['M-labels-SRC'].bfill()
    else:
        df['M-labels-SRC'] = ""

    if 'CLUSTERING' in df.columns:
        df['CLUSTERING'] = df['CLUSTERING'].ffill()
        df['CLUSTERING'] = df['CLUSTERING'].bfill()
    else:
        df['CLUSTERING'] = ""



    df['ft_acc'] = df['ft_acc'].ffill()
    df['ft_acc'] = df['ft_acc'].bfill()

    # Change all dict values by the first key
    df['src_datasets'] = df['src_datasets'].apply(lambda x: list(eval(x).keys())[0] if type(x) is dict else x)

    try:
        # Remove rows where ft_eval_acc is empty
        df = df[~df['ft_eval_acc'].isnull()]
    except:
        new_path = os.path.join("discarded_logs/ft_eval_acc_not_exists", path_file )
        os.makedirs(os.path.dirname(new_path), exist_ok=True)        
        os.rename(path_file, new_path)
        return df, "ft_eval_acc not exists for path: " + path_file  
    # Add column with the number of fine-tuning evaluations

    new_df = df.copy()
    index = [int(i) for i in range(len(df)) ]
    new_df['index'] = index
    new_df.set_index('index', inplace=True)
    new_ft_num, counter, prev_boots_iter = [], 0, df['bootstrap_iter'].iloc[index[0]] 

    for idx in index:
        if df['bootstrap_iter'].iloc[idx]  != prev_boots_iter:
            error_ft_eval_iters(counter, path_file, df)

            prev_boots_iter = df['bootstrap_iter'].iloc[idx]  
            counter = 0
        new_df.loc[idx, 'ft_eval_num'] = int(counter)
        counter += 1
    
    error_ft_eval_iters(counter, path_file, df)
    df = new_df
    
    

    # Remove rows where before_ft_eval_acc is empty
    # df = df[~df['before_ft_eval_acc'].isnull()]
    # Or where before_ft_eval_acc is ""
    df = df[~df['before_ft_eval_acc'].isin(["", " ", "nan"])]
    
    df = reduce_redundancy(df, exp_num)

    return df, ""


def reduce_redundancy(df, exp_num):
    # Remove rows where exp_name do not contains "src_ds"

    df['exp_name'] = exp_num
    # exception_exps = [3,9,10]
    # if not exp_num in exception_exps:        
    #     df_c = df[df['exp_name'].str.contains("src_ds", na=False)]
    #     if len(df_c) == 0:
    #         df_c = df[df['exp_name'].str.contains("same_dataset", na=False)]
    #     if len(df_c) != 0:
    #         df = df_c
    # Mean rows where everything is the same except for ft_eval_acc
    common_cols = ["bootstrap_iter","lr","batch_size","input_size","model_type","n_way_test","n_way_train","tgt_dataset","src_datasets","fixed_support","samples_per_class","num_total_episodes","num_bootstrap_iters","exp_name","ft_eval_num", 'CLUSTERING', 'M-labels-SRC']
    uncommon_cols = df.columns.difference(common_cols)
    df_g = df.groupby(common_cols)

    # means, stds = [], []
    # # Size of each group
    # for name, group in df_g:
    #     means.append(len(group))
    #     stds.append(group['ft_eval_acc'].std())
    #     if group['ft_eval_acc'].std() > 0.1:
    #         breakpoint()

    #     if len(group) > 1:
    #         diff_cols = []         
    #         for col in uncommon_cols:
    #             nunique_per_group = group[col].nunique()
    #             if (nunique_per_group > 1):
    #                 # breakpoint()
    #                 diff_cols.append(col)
    #         # print("Group with more than one row: ", group[diff_cols], name)
    #         pass
    #         # print("REPEATED", group)
    #         # breakpoint()

    df = df_g.mean(numeric_only=True).reset_index()

    df = comprobations_coherence_columns(df)
    return df

def comprobations_coherence_columns(df):

    expected_order = ["_step","_runtime","bootstrap_iter","_timestamp","epoch","tr_loss","tr_acc","tr lr","tr_eval_acc","ft_eval_acc","ft_loss","ft_acc","lr","epochs","batch_size","input_size","model_type","n_way_test","n_way_train","tgt_dataset","src_datasets","fixed_support","samples_per_class","num_total_episodes","num_bootstrap_iters","exp_name","ft_eval_num","tr_eval_acc_src_data","before_ft_eval_acc","before_ft_eval_loss", "CLUSTERING", "M-labels-SRC"]
    for ex in expected_order:
        if ex not in df.columns:
            df[ex] = ""
    for ex in df.columns:
        if ex not in expected_order:
            df[ex] = ""

    ## Order columns
    df = df[expected_order]
    return df

def undesired_to_print_cases(df, path_file):
    if "BreaKHis" in df["src_datasets"].unique()[0] or "BreaKHis" in df["tgt_dataset"].unique()[0]:
        # print("Undesired case: BreakHis in src or tgt dataset")
        return True
    
    if "exp1" in df["exp_name"].unique()[0]:
        if "omniglot_SOTA_trainSet" in df["src_datasets"].unique()[0] and "omniglot_SOTA_testSet" in df["tgt_dataset"].unique()[0]:
            return True
        if "miniImageNet_SOTA_trainSet" in df["src_datasets"].unique()[0] and "miniImageNet_SOTA_testSet" in df["tgt_dataset"].unique()[0]:
            return True
        if "cifar100_SOTA_trainSet" in df["src_datasets"].unique()[0] and "cifar100_SOTA_testSet" in df["tgt_dataset"].unique()[0]:
            return True
        
    to_cluster_kmeans = False

    if "CLUSTERING" in df: 
        clusts = df["CLUSTERING"].unique()
        clusts = [c for c in clusts if not pd.isna(c)]
        np.testing.assert_equal(len(clusts), 1)
        clusts = clusts[0]

        if clusts == "KMEANS" or clusts == "CONSTRAINED-KMEANS":
            df_ = df[~df["CLUSTERING"].isna()]
            if len(df_["CLUSTERING"].unique()) != 1:
                raise ValueError("Expected exactly one non-null CLUSTERING value.")
            #remove nan values from df["CLUSTERING"]
            if len(df_["CLUSTERING"].unique()) != 1:
                raise ValueError("Expected exactly one non-null CLUSTERING value.")
            np.testing.assert_equal(len(df_["CLUSTERING"].unique()), 1)
            to_cluster_kmeans = True

    if df["n_way_train"].unique()[0] > 18:
        banned = [DATASETS.EGYPTIAN.value, DATASETS.BREAKHIS.value] # "miniImageNet_SOTA_trainSet", "cifar100_SOTA_trainSet",
        if to_cluster_kmeans:
            banned = [] #["miniImageNet_SOTA_trainSet", "cifar100_SOTA_trainSet"]
        for b in banned:
            if b == df["src_datasets"].unique()[0]:
                return True
    if df["n_way_test"].unique()[0] > 18:
        banned = [DATASETS.EGYPTIAN.value, 
                   DATASETS.BREAKHIS.value, DATASETS.CAPITAN.value 
                  ] # "miniImageNet_SOTA_testSet", "cifar100_SOTA_testSet",  
        for b in banned:
            if b == df["tgt_dataset"].unique()[0]:
                return True

    # Not all datasets for 10 samples per class
    if 10 == df["samples_per_class"].unique()[0]:

        banned = [ DATASETS.BREAKHIS.value] #"omniglot_SOTA_trainSet", "omniglot_SOTA_testSet", "miniImageNet_SOTA_trainSet", "miniImageNet_SOTA_testSet", 
                                            #"cifar100_SOTA_trainSet", "cifar100_SOTA_testSet",
                   
        for b in banned:
            if b == df["src_datasets"].unique()[0] or b == df["tgt_dataset"].unique()[0]:
                return True


    return False



def check_banned_cases(df):


    if len(df["src_datasets"].unique()) == 0:
        print("No source datasets in df")
        return True

    if len(df["tgt_dataset"].unique()) > 1 and df["tgt_dataset"].unique()[1] == "omniglot_SOTA_trainvalSet":
        df["tgt_dataset"] = "omniglot_SOTA_trainSet"

    if len(df["src_datasets"].unique()) == 1 and df["src_datasets"].unique()[0] == "omniglot_SOTA_trainvalSet":
        df["src_datasets"] = "omniglot_SOTA_trainSet"

    if "n_way_train" not in df.columns:
        print("n_way_train not in df columns")
        return True

    np.testing.assert_equal(len(df["n_way_train"].unique()), len(df["n_way_test"].unique()))
    np.testing.assert_equal(len(df["n_way_train"].unique()), len(df["samples_per_class"].unique()))
    np.testing.assert_equal(len(df["n_way_train"].unique()), len(df["src_datasets"].unique()))
    np.testing.assert_equal(len(df["n_way_train"].unique()), len(df["tgt_dataset"].unique()))
    np.testing.assert_equal(len(df["n_way_train"].unique()), 1)

    
            
    # Omniglot_trainvalSet and miniImageNet_SOTA_trainSet for test
    if df["tgt_dataset"].unique()[0] == "omniglot_SOTA_trainSet" or df["tgt_dataset"].unique()[0] == "miniImageNet_SOTA_trainSet":
        print("Banned case: ", df["tgt_dataset"].unique()[0], "as target dataset")
        return True
    
    all_ds = df["exp_name"].unique()
    all_ds = [c for c in all_ds if not pd.isna(c)]
    if "1" in all_ds[0]:
        if "NO_SRC_DATASET" in df["src_datasets"].unique()[0]:
            print("Banned case: ", df["src_datasets"].unique()[0], "as source dataset in exp1")
            return True

    if 'ft_acc' not in df.columns:
        print("Banned case: ft_acc not in df columns")
        return True

    return False


logs_base_path = "logs_csv/"

# Fix migrations
# os.system("python3 scripts/fix_migrations_id.py")

outdated_exps = ["last_exec_exp12", "last_exec_exp2", "last_exec_exp6", "last_exec_exp9"]
# outdated_exps += ["last_exec_exp3", "last_exec_exp4", "last_exec_exp10", "last_exec_exp13"]

## Extract data by num_experiments and group:
for root, folders, files in os.walk(logs_base_path):
    for folder in folders:

        if not "last_exec_exp" in folder:
        # if not folder.startswith("last_exec_exp3"):
            continue
        
        forbidden_folder = False
        ## Num used for testing experiments
        for exp_b in outdated_exps: 
            if folder.endswith(exp_b):
                forbidden_folder = True

        if forbidden_folder:
            continue


        exp_tables = {}
        folder_path = os.path.join(root, folder)
        if os.path.exists(folder_path):   
            exp_num = int(folder.split("last_exec_exp")[1])
            for f_name in os.listdir(folder_path):
                
                # DEBUG
                # if not "PrototypicalNetwork" in f_name:
                #     continue

                if not f_name.endswith(".csv"):
                    continue
                metadata = extract_metadata_from_namefile(f_name)
                
                path_file = os.path.join(folder_path, f_name)
                df = pd.read_csv(os.path.join(path_file), index_col=None)
                df['index'] = range(len(df))


                if check_banned_cases(df):
                    print("Banned case: ", path_file)
                    # Move file to folder outside
                    new_path = os.path.join("discarded_logs/banned_cases", path_file )
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    # if "Egyptian" in path_file and "NO_SRC_DATASET" in path_file:
                    #     if "PrototypicalNetwork" in path_file and "spc_1-" in path_file and "5-nway" in path_file:
                    #         breakpoint()
                    

                    os.rename(path_file, new_path)
                    continue

                if undesired_to_print_cases(df, path_file):
                    new_path = os.path.join("discarded_logs/undesired_cases", path_file )
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    os.rename(path_file, new_path)
                    # print("Undesired case: ", path_file)
                    continue                


                df, error = extend_and_remove_non_relevant_rows(df, path_file, exp_num=exp_num)
                    
                if error != "":
                    print(error)
                    continue

                ### Exception columns
                df = comprobations_coherence_columns(df)

                if not metadata['exp_name'] in exp_tables:
                    exp_tables[metadata['exp_name']] = df
                else:
                    exp_tables[metadata['exp_name']] = pd.concat([exp_tables[metadata['exp_name']], df])

                if len(df.columns) != len(exp_tables[metadata['exp_name']].columns):
                    raise ValueError(
                        f"Column mismatch while merging {metadata['exp_name']}: {list(df.columns)}"
                    )

                exp_tables[metadata['exp_name']] = reduce_redundancy(exp_tables[metadata['exp_name']], exp_num)
                # if len(exp_tables[metadata['exp_name']]["_timestamp"].unique()) != len(exp_tables[metadata['exp_name']]):
                #     breakpoint()
                



            ## Save the data
            for exp_name, df in exp_tables.items():
                print("######################### STARTING EXP", exp_name, "from", folder_path)
                df.to_csv('logs_csv/reduced_logs_csv/' + exp_name + '.csv', index=False)


                check_complete_executions(exp_name, df, exhaustive=True)
                    

                    # print(counts)

                # print("Path: ", folder_path)
                # print("Finished boots iters: ", len(df["bootstrap_iter"].unique()))
                # print("N-way train: ", df["n_way_train"].unique())
                # print("N-way test: ", df["n_way_test"].unique())
                # print("Model type: ", df["model_type"].unique())
                # print("Samples per class: ", df["samples_per_class"].unique())
                # print("FT eval num: ", len(df["ft_eval_num"].unique()))





