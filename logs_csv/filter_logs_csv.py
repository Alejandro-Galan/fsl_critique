# Filter data by columns and unify in a bigger csv
# And show main statistics

import os, glob
import pandas as pd


def extract_metadata_from_namefile(f_name):
    metadata = {}
    metadata["exp_name"] = f_name.split('-')[0]
    metadata["tgt-ds"] = f_name.split('-')[1]
    metadata["src-dss"] = f_name.split('src_ds_')[1].split('tgt_ds-')[0].split("_")
    metadata["src-dss"] = [s for s in metadata["src-dss"] if s != '']

    return metadata

def extend_and_remove_non_relevant_rows(df):   
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
        df['before_ft_eval_acc'] = df['before_ft_eval_acc'].ffill()
        df['before_ft_eval_acc'] = df['before_ft_eval_acc'].bfill()
    else:
        df['before_ft_eval_acc'] = ""

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


    # Remove rows where ft_eval_acc is empty
    df = df[~df['ft_eval_acc'].isnull()]
    
    # Add column with the number of fine-tuning evaluations
    new_ft_num, counter, prev_boots_iter = [], 0, 0
    new_df = df.copy()
    for idx in df.index:
        if df.loc[idx, 'bootstrap_iter'] != prev_boots_iter:
            prev_boots_iter = df.loc[idx, 'bootstrap_iter'] 
            counter = 0
        new_df.loc[idx, 'ft_eval_num'] = counter
        counter += 1
    df = new_df
    # Select rows where ft_eval_acc is empty
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


logs_base_path = "logs_csv/"

## Extract data by num_experiments and group:
for root, folders, files in os.walk(logs_base_path):
    for folder in folders:
        if not folder.startswith("last_exec_"):
            continue
        exp_tables = {}
        folder_path = os.path.join(root, folder)
        if os.path.exists(folder_path):   
            for f_name in os.listdir(folder_path):
                if not f_name.endswith(".csv"):
                    continue
                metadata = extract_metadata_from_namefile(f_name)
                
                df = pd.read_csv(os.path.join(folder_path, f_name), index_col=0)
                df = extend_and_remove_non_relevant_rows(df)

                ### Exception columns
                df = comprobations_coherence_columns(df)
            
                if not metadata['exp_name'] in exp_tables:
                    exp_tables[metadata['exp_name']] = df
                else:
                    exp_tables[metadata['exp_name']] = pd.concat([exp_tables[metadata['exp_name']], df])
            
                
            ## Save the data
            for exp_name, df in exp_tables.items():
                df.to_csv('logs_csv/reduced_logs_csv/' + exp_name + '.csv', index=False)
                print("\nFor experiment: ", exp_name)
                print("Finished boots iters: ", len(df["bootstrap_iter"].unique()))
                print("N-way train: ", df["n_way_train"].unique())
                print("N-way test: ", df["n_way_test"].unique())
                print("Model type: ", df["model_type"].unique())
                print("Samples per class: ", df["samples_per_class"].unique())
                print("FT eval num: ", len(df["ft_eval_num"].unique()))











