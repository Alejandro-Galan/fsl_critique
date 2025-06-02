
import os
import socket
import pandas as pd
import numpy as np

def rules_to_change(filename, full_path):

    filen = filename.replace("Episodes", "E")
    filen = filen.replace("----", "-")
    filen = filen.replace("--", "-")
    filen = filen.replace("FIXED_SUPP_SET", "FSS")
    filen = filen.replace("KSamples_per_Class", "KSpC")
    filen = filen.replace("tgt_dataset", "tgt_ds")
    filen = filen.replace("INPUT_SIZE", "I_S")
    filen = filen.replace("trained_finetuned_model", "tr_ftm")
    filen = filen.replace("cluster_kmeans_m_-1", "cluster_kmeans_m_eq_n")
    filen = filen.replace("cluster_kmeans_m_.csv", "cluster_kmeans_m_eq_n.csv")

    filen = filen.replace("-nwayTrain-5-nwayTrain_",  "-nwayTrain_")
    filen = filen.replace("-nwayTrain-20-nwayTrain_", "-nwayTrain_")
    filen = filen.replace("-nwayTrain-3-nwayTrain_",  "-nwayTrain_")

    if "_.csv" in filename:
        df = pd.read_csv(full_path)
        real_nway = df["n_way_train"].values[0]
        np.testing.assert_equal(len(df["n_way_train"].unique()), 1)
        if "Network_.csv" in filename:
            filen = filen.replace("Network_.csv", "Network-" + str(real_nway) + "-nwayTrain_.csv")
        else:
            if not "Network-" + str(real_nway) + "-nwayTrain_.csv" in filename:
                print(real_nway, filen)
                filen = filen.replace("Network-3-nwayTrain_.csv", "Network-" + str(real_nway) + "-nwayTrain_.csv")
                filen = filen.replace("Network-5-nwayTrain_.csv", "Network-" + str(real_nway) + "-nwayTrain_.csv")
                filen = filen.replace("Network-20-nwayTrain_.csv", "Network-" + str(real_nway) + "-nwayTrain_.csv")
                
    # machine = socket.gethostname()
    # if machine == "bilbo":
    #     NWAY = 5 
    # elif machine == "multiscore":
    #     NWAY = 20
    # filen = filen.replace("Network_.csv", "Network--" + str(NWAY) +"-nwayTrain_.csv")
    
    return filen

# Iterate through all weights and logs files
def rename(root_dir):
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Renombrar archivos
        for filename in filenames:
            new_filename = rules_to_change(filename, full_path=os.path.join(dirpath, filename))
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_filename)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
        
        # Renombrar carpetas
        for dirname in dirnames:
            new_dirname = rules_to_change(dirname, full_path=os.path.join(dirpath, dirname))
            old_path = os.path.join(dirpath, dirname)
            new_path = os.path.join(dirpath, new_dirname)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

# Main
if __name__ == "__main__":
    #root_dir_w = "WEIGHTS/"
    #rename(root_dir_w)
    root_dir_l = "logs_csv/"
    rename(root_dir_l)
