# Filter data by columns and unify in a bigger csv

import os, glob
import pandas as pd


def extract_metadata_from_namefile(f_name):
    metadata = {}
    metadata["exp_name"] = f_name.split('--')[0]
    metadata["tgt-ds"] = f_name.split('--')[1]
    metadata["src-dss"] = f_name.split('src_ds_')[1].split('tgt_ds--')[0].split("_")
    metadata["src-dss"] = [s for s in metadata["src-dss"] if s != '']

    return metadata

def extend_and_remove_non_relevant_rows(df):    
    if 'tr lr' in df.columns:
       df['tr lr'] = df['tr lr'].ffill()
       df['tr lr'] = df['tr lr'].bfill()
    df['lr'] = df['lr'].ffill()
    df['lr'] = df['lr'].bfill()

    if 'tr_acc' in df.columns:
        df['tr_acc'] = df['tr_acc'].ffill()
        df['tr_acc'] = df['tr_acc'].bfill()
    df['ft_acc'] = df['ft_acc'].ffill()
    df['ft_acc'] = df['ft_acc'].bfill()

    if 'tr_eval_acc_src_data' in df.columns:
        df['tr_eval_acc_src_data'] = df['tr_eval_acc_src_data'].ffill()
        df['tr_eval_acc_src_data'] = df['tr_eval_acc_src_data'].bfill()
    df['ft_acc'] = df['ft_acc'].ffill()
    df['ft_acc'] = df['ft_acc'].bfill()


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

experiments = ["exp1", "exp2", "exp3", "exp4"]

## Extract data by num_experiments and group:
for ex in experiments:
    exp_tables = {}
    folder_path = 'logs_csv/last_exec_' + ex + '/'
    if os.path.exists(folder_path):   
        for f_name in os.listdir(folder_path):
            if not f_name.endswith(".csv"):
                continue
            metadata = extract_metadata_from_namefile(f_name)
            
            df = pd.read_csv(os.path.join(folder_path, f_name))
            df = extend_and_remove_non_relevant_rows(df)

            if not metadata['exp_name'] in exp_tables:
                exp_tables[metadata['exp_name']] = df
            else:
                exp_tables[metadata['exp_name']] = pd.concat([exp_tables[metadata['exp_name']], df])

        ## Save the data
        for exp_name, df in exp_tables.items():
            df.to_csv('logs_csv/reduced_logs_csv/' + exp_name + '.csv', index=False)











