import os, re
import numpy as np
import pandas as pd




# Read np array distances and accs
def read_file(path):
    data = np.load(path)
    accs = np.load(path.replace(".npy", "_accs.npy"))

    return data, accs

def extract_metadata_from_file(full_path):
    dir_name = os.path.basename(os.path.dirname(full_path))

    pattern = (
        r"src_ds_name-(?P<src_ds_name>.+?)"       
        r"--spc_(?P<spc>\d+)"                     
        r"-model-(?P<model>.+?)"                  
        r"-(?P<nway>\d+)-nwayTrain"               
    )

    match = re.match(pattern, dir_name)

    if match:
        trained_ds = match.group("src_ds_name")
        spc = int(match.group("spc"))
        model = match.group("model")
        nway = int(match.group("nway"))
        # print(f"trained_ds: {trained_ds}, spc: {spc}, model: {model}, nway: {nway}")
    else:
        print("Formato no reconocido.", full_path)    
        breakpoint()
    if "ft_distances_over" in full_path:
        unseen_ds = full_path.split("ft_distances_over_")[1].split("as_tgt_ds_")[0]
        unseen_ds = unseen_ds.replace("_accs.npy", "")
        
        if unseen_ds.startswith("omniglot_SOTA"):
            unseen_ds = "omniglot_SOTA_testSet"
        elif unseen_ds.startswith("miniImageNet_SOTA"):
            unseen_ds = "miniImageNet_SOTA_testSet"
        elif unseen_ds.startswith("cifar100_SOTA"):
            unseen_ds = "cifar100_SOTA_testSet"
        
        domain = "OOD" 
    else:
        unseen_ds = trained_ds
        if trained_ds.startswith("omniglot_SOTA"):
            unseen_ds = "omniglot_SOTA_testSet"
        elif trained_ds.startswith("miniImageNet_SOTA"):
            unseen_ds = "miniImageNet_SOTA_testSet"
        elif trained_ds.startswith("cifar100_SOTA"):
            unseen_ds = "cifar100_SOTA_testSet"
        
        domain = "ID" 

    metadata = {"trained_ds": trained_ds, "spc": spc, "model": model, "nway": nway, "unseen_ds": unseen_ds, "domain": domain}

    return metadata


def quartile_ratio(ordered_dists, normalize):
    values = ordered_dists.reshape(-1, ordered_dists.shape[2])
    
    sum_min_values = 0
    ratios_sum, ratios_limits = [], []
    for i, query_s in enumerate(values):
        # freq, bins = np.histogram(query_s, bins=50)
        sum_min_values += np.min(query_s)

        max_values = np.sum(query_s)
        if normalize:
            query_s = query_s / max_values
        
        # if i == 1:
        #     print(i, "unnorm_s:", query_s * max_values)
        #     print(i, "query_s:", query_s)


        q1 = np.percentile(query_s, 25)
        q2 = np.percentile(query_s, 50)  
        q3 = np.percentile(query_s, 75)
        q4 = np.percentile(query_s, 100)

        # Get all values from each quartile
        q4_values = np.sum(query_s[query_s > q3])
        query_s = query_s[query_s <= q3]
        q3_values = np.sum(query_s[query_s > q2])
        query_s = query_s[query_s <= q2]
        q2_values = np.sum(query_s[query_s > q1])
        q1_values = np.sum(query_s[query_s <= q1])
        
        ratio_sum = (q2_values + q1_values) / (q4_values + q3_values)
        ratio_lim = (q2 + q1) / (q4 + q3)
        
        # if i == 1:
        #     print("ratio_sum:", ratio_sum, "ration_lim:", ratio_lim, "qs:", q1, q2, q3, q4)
        
        ratios_sum.append(ratio_sum)
        ratios_limits.append(ratio_lim)
    return np.mean(ratios_sum), np.mean(ratios_limits), sum_min_values

def calculate_metrics(af_distances, af_accs, base_dists):
    # Calculate the mean difference
    mean_diff = np.mean(af_distances) - np.mean(base_dists)
    
    # Permute to order by class [Samples, N_classes, Querys]
    perm_dists = np.transpose(af_distances, (0, 2, 1))
    # Order the distances by class [Samples, Querys,  N_classes]
    ordered_dists = np.sort(perm_dists, axis=2)

    # ratio      = quartile_ratio(ordered_dists, normalize=False)
    norm_ratio_sum, norm_ratio_limits, sum_min_values = quartile_ratio(ordered_dists, normalize=True)

    # Calculate the accuracy
    acc = np.mean(af_accs)

    return {"mean_diff": mean_diff, "norm_ratio_sum": norm_ratio_sum, "norm_ratio_limits": norm_ratio_limits, "acc": acc, "sum_min_values": sum_min_values}

def bt_suffix(bt_epoch):
    return  "_bt_" + str(bt_epoch)

def main(sufix):
    # Extract distances from DISTANCES folder
    root = "DISTANCES/" + sufix + "/"
    folders = os.listdir(root)
    
    af_sufix = "_after_ft_distances"

    df = pd.DataFrame()
    for folder in folders:

        folder_path = os.path.join(root, folder)
        if os.path.isdir(folder_path):
            print("Processing folder:", folder)
            files = os.listdir(folder_path)
            metadatas = {}
            boots_iters = []
            for file in files:
                file_path = os.path.join(folder_path, file)
                metadata = extract_metadata_from_file(file_path)
                if file.endswith("accs.npy"):
                    # Skip accuracy files
                    continue
                if file.startswith("3_ft_distances") or file.startswith("_after_3_ft_distances"):
                    # Remove file
                    print("Removing old file:", file_path)
                    os.remove(file_path)
                    continue
                metadata["bt_iter"] = int(file_path.split("bt_")[-1].split(".npy")[0])
                if metadata["bt_iter"] not in boots_iters:
                    boots_iters.append(metadata["bt_iter"])

                if file.startswith(af_sufix):
                    # print(file_path)
                    af_distances, af_accs = read_file(file_path)      
                    metadatas["base" + bt_suffix(metadata["bt_iter"])] = metadata
                elif file.startswith("ft_distances_over"):
                    af_distances, af_accs = read_file(file_path)       
                    metadatas[metadata['unseen_ds'] + bt_suffix(metadata["bt_iter"])] = metadata
                else:
                    # print("Skipping file:", file_path)
                    continue
                metadata["af_distances"] = af_distances
                metadata["af_accs"] = af_accs

            if "base_bt_0" not in metadatas:
                print("Empty folder:", folder_path)
                try:
                    os.rmdir(folder_path)  
                except:
                    print("Could not remove folder:", folder_path)
                continue 
            
            for bt_ep in boots_iters:
                
                # filter metadata with elements finishing with bt_epoch only 
                bt_suffix_str = bt_suffix(bt_ep)
                metadatas_ft = {k: v for k, v in metadatas.items() if k.endswith(bt_suffix_str)}

                base_dists = metadatas_ft["base" + bt_suffix(bt_ep)]["af_distances"]

                # for name, met in metadatas_ft.items():
                for name, met in metadatas_ft.items():
                    
                    # # Forbidden cases
                    # if met["trained_ds"] == "omniglot_SOTA_trainvalSet" or met["trained_ds"] == "omniglot_SOTA_trainSet" or met["trained_ds"] == "miniImageNet_SOTA_trainSet":
                    #     continue

                    # if met["unseen_ds"] == "omniglot_SOTA_testSet":
                    #     continue

                    print("trained_ds:", met["trained_ds"], "unseen_ds:", name, "\n\tspc:", met["spc"], "nway:", met["nway"])
                    if met["unseen_ds"].startswith(met["trained_ds"]) and met["domain"] == "OOD":
                        print("pillada")
                        continue
                    
                    metrics = calculate_metrics(met["af_distances"], met["af_accs"], base_dists)
                    
                    # Add new row to the DataFrame
                    new_row = {
                        "unseen_ds": met["unseen_ds"],
                        "spc": met["spc"],
                        "model": met["model"],
                        "nway": met["nway"],
                        "trained_ds": met["trained_ds"],
                        "mean_diff": metrics["mean_diff"],
                        "acc": met["af_accs"][0],
                        "dis_mean": np.mean(met["af_distances"]),
                        "norm_ratio_sum": metrics["norm_ratio_sum"],
                        "norm_ratio_limits": metrics["norm_ratio_limits"],
                        "domain": met["domain"],
                        "acc_std": met["af_accs"][1],
                        "bt_iter": met["bt_iter"],
                        "sum_min_values": metrics["sum_min_values"]

                    }

                    # new_row = {
                    #     "unseen_ds": met["unseen_ds"],
                    #     "spc": met["spc"],
                    #     "model": met["model"],
                    #     "nway": met["nway"],
                    #     "trained_ds": met["trained_ds"],
                    #     "acc": met["af_accs"][0],
                    #     "domain": met["domain"],

                    # }
                    
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv("distances_analysis_" + sufix + ".csv", index=False)
    print(df)
    # breakpoint()

main("RelationNetwork")
# main("PrototypicalNetwork")
