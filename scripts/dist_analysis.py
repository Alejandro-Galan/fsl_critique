import os, re
import numpy as np
import pandas as pd


# Extract distances from DISTANCES folder
root = "DISTANCES/"
folders = os.listdir(root)

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
        src_ds_name = match.group("src_ds_name")
        spc = int(match.group("spc"))
        model = match.group("model")
        nway = int(match.group("nway"))
        print(f"src_ds_name: {src_ds_name}, spc: {spc}, model: {model}, nway: {nway}")
    else:
        print("Formato no reconocido.", full_path)    
        breakpoint()
    
    if "3_ft_distances_over" in full_path:
        tgt_ds = full_path.split("3_ft_distances_over_")[1].replace("as_tgt_ds.npy", "")
        tgt_ds = tgt_ds.replace("_accs.npy", "")
    else:
        tgt_ds = src_ds_name

    metadata = {"src_ds_name": src_ds_name, "spc": spc, "model": model, "nway": nway, "tgt_ds": tgt_ds}

    return metadata


def quartile_ratio(ordered_dists, normalize):
    values = ordered_dists.reshape(-1, ordered_dists.shape[2])
    
    ratios = []
    for i, query_s in enumerate(values):
        # freq, bins = np.histogram(query_s, bins=50)

        max_values = np.sum(query_s)
        if normalize:
            query_s = query_s / max_values
        if i == 1:
            print(i, "unnorm_s:", query_s * max_values)
            print(i, "query_s:", query_s)


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
        
        ratio = (q2_values + q1_values) / (q4_values + q3_values)
        if i == 1:
            print("ratio:", ratio, "qs:", q1, q2, q3, q4)
        ratios.append(ratio)
    return np.mean(ratios)

def calculate_metrics(af_distances, af_accs, base_dists):
    # Calculate the mean difference
    mean_diff = np.mean(af_distances) - np.mean(base_dists)
    
    # Permute to order by class [Samples, N_classes, Querys]
    perm_dists = np.transpose(af_distances, (0, 2, 1))
    # Order the distances by class [Samples, Querys,  N_classes]
    ordered_dists = np.sort(perm_dists, axis=2)

    # ratio      = quartile_ratio(ordered_dists, normalize=False)
    norm_ratio = quartile_ratio(ordered_dists, normalize=True)

    # Calculate the accuracy
    acc = np.mean(af_accs)
    
    return {"mean_diff": mean_diff, "norm_ratio": norm_ratio, "acc": acc}


af_sufix = "_after_3_ft_distances.npy"

df = pd.DataFrame()
for folder in folders:

    folder_path = os.path.join(root, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        metadatas = {}
        for file in files:
            file_path = os.path.join(folder_path, file)
            metadata = extract_metadata_from_file(file_path)
            if file.startswith(af_sufix):
                af_distances, af_accs = read_file(file_path)            
                metadatas["base"] = metadata
            elif file.startswith("3_ft_distances_over") and not file.endswith("accs.npy"):
                af_distances, af_accs = read_file(file_path)            
                metadatas[metadata['tgt_ds']] = metadata
            else:
                continue
            metadata["af_distances"] = af_distances
            metadata["af_accs"] = af_accs

        base_dists = metadatas["base"]["af_distances"]

        for name, met in metadatas.items():
            print("src_ds_name:", met["src_ds_name"], "tgt_ds:", name, "spc:", met["spc"], "nway:", met["nway"])
            metrics = calculate_metrics(met["af_distances"], met["af_accs"], base_dists)
            # Add new row to the DataFrame
            new_row = {
                "src_ds_name": met["src_ds_name"],
                "spc": met["spc"],
                "model": met["model"],
                "nway": met["nway"],
                "tgt_ds": met["tgt_ds"],
                "mean_diff": metrics["mean_diff"],
                "acc": met["af_accs"],
                "dis_mean": np.mean(met["af_distances"]),
                "norm_ratio": metrics["norm_ratio"],
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv("distances_analysis.csv", index=False)
print(df)
# breakpoint()
