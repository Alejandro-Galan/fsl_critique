## Open all folders and remove all files that contain "omniglot", "breakhis", "cifar100" or "miniImageNet" in the name 
import os, shutil
from pathlib import Path

# Define the keywords to search for
# keywords = ["omniglot", "breaKHis", "cifar100", "miniImageNet"]
# keywords = ["RelationNetwork"]
# keywords = ["CONSTR", "constr_kmeans"]

def remove_folder(base_folder):
    number_of_files = 0
    # Iterate through all subdirectories
    for subdir in Path(base_folder).rglob("*"):
        if subdir.is_dir():
            
            if any(keyword in subdir.name for keyword in keywords):
                print(f"Removing directory: {subdir}")
                for file in subdir.glob("*"):
                    if file.is_file():
                        print(f"Removing file: {file}")
                        number_of_files += 1
                        file.unlink()                
                        number_of_files += 1
                    elif file.is_dir():
                        print(f"Removing subdirectory: {file}")
                        shutil.rmtree(file)
                subdir.rmdir()  
            else:            
                # Iterate through all files in the subdirectory
                for file in subdir.glob("*"):
                    if any(keyword in file.name for keyword in keywords):
                        if file.is_file():
                            print(f"Removing file: {file}")
                            number_of_files += 1
                            file.unlink()  # Remove the file
                        elif file.is_dir():
                            print(f"Removing subdirectory: {file}")
                            # Remove the subdirectory even if it is not empty
                            shutil.rmtree(file)

    print(f"Total number of files removed: {number_of_files}")


remove_folder("logs_csv/")

remove_folder("WEIGHTS/")

remove_folder("utils/stored_sets/")

