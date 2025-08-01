## Rename all "PrototypicalNetwork" files to "" and remove "RelationNetwork" and "MatchingNetwork"
import os, re

def delete_redundancy_names_stored_sets(directory):

    # Define the patterns to match
    patterns = [
        r"PrototypicalNetwork",
        r"RelationNetwork",
        r"MatchingNetwork"
    ]
    
    # Iterate through all files in the directory
    for folder in os.listdir(directory):
        # Check if the folder name matches any of the patterns
        if any(re.search(pattern, folder) for pattern in patterns):
            file_path = os.path.join(directory, folder)
            if "PrototypicalNetwork" in folder:
                # Rename the file to an empty string
                new_file_path = file_path.replace("PrototypicalNetwork", "")
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} to {new_file_path}")
            else:
                os.system("rm -r " + file_path)
                print(f"Deleted: {file_path}")  # Optional: Print deleted file path for confirmation

base_p = "utils/stored_sets/"
folders = ["CONSTR-KMEANS_labels_clust_m25", "CONSTR-KMEANS_labels_clust_m250", 
           "CONSTR-KMEANS_labels_clust_m-1", "CONSTR-KMEANS_labels_clust_m1000"]
for folder in folders:
    directory = os.path.join(base_p, folder)
    if os.path.exists(directory):
        print(f"Processing directory: {directory}")
        delete_redundancy_names_stored_sets(directory)
