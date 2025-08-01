from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE
import sys, os, json

import importlib
from datasets.loader import extract_validation_from_test
from utils import constants
from utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants



def write_tsne_representation(filepath, x, y, w2i):
    i2w = {v: k for k, v in w2i.items()}
    # Sometimes n_samples is lower than default perplexity (30)
    perp = min(30.0, float(x.shape[0]) - 1 )
    x_embedded = TSNE(n_components=2, perplexity=perp).fit_transform(x)
    with open(filepath, "w") as datfile:
        for i, components in enumerate(x_embedded):
            datfile.write(f"{components[0]}\t{components[1]}\t{y[i]}\t{i2w[y[i]]}\n")


def write_plot_results(
    filepath,
    from_weights,
    epochs,
    batch_size,
    results,
    samples_per_class,
):
    # NOTE: results is a list of accuracies!
    if not filepath.exists():
        with open(filepath, "w") as datfile:
            header = [
                "from_weights",
                "samples_per_class",
                "bootstrap_runs",
                "epochs",
                "batch_size",
                "mean_accuracy",
                "std_accuracy",
            ]
            datfile.write("\t".join(header) + "\n")
    with open(filepath, "a") as datfile:
        values = [
            from_weights,
            samples_per_class,
            len(results),
            epochs,
            batch_size,
            np.mean(results),
            np.std(results),
        ]
        datfile.write("\t".join([str(value) for value in values]) + "\n")


def extract_support_set(X, Y, samples_per_class, model_type):


    if model_type in Constants['AllowedModels']:
        supp_X, supp_Y = [], []
        reduced_X, reduced_Y = [], []
        # Extract in sequential order to preserve always the same 
        unique_ys = np.unique(Y, axis=0)
        for unique_y in unique_ys:
            indices = np.argwhere(Y == unique_y).flatten()

            ## All training set is used for the sampels calculations in each batch
            if True:
                supp_X.extend(X[indices])
                supp_Y.extend(Y[indices])
                reduced_X.extend(X[indices])
                reduced_Y.extend(Y[indices])

            ## Support will be calculated on the fly, duplicate train
            if False:
                # supp_samples = indices[:samples_per_class]
                supp_samples = np.random.choice(indices, samples_per_class, replace=False)
                supp_X.extend(X[supp_samples])
                supp_Y.extend(Y[supp_samples])
                reduced_X, reduced_Y = X, Y
            
            ## Fixed support at the start
            if False:
                ## Arbitrary support set
                if False:
                    supp_samples = np.random.choice(indices, samples_per_class, replace=False)
                    remainder_samples = np.setdiff1d(indices, supp_samples)
                ## Fixed support set
                if True:
                    supp_samples = indices[:samples_per_class] # Choose only k on k-shot samples
                    remainder_samples = indices[samples_per_class:]

                supp_X.extend(X[supp_samples])
                supp_Y.extend(Y[supp_samples])
                reduced_X.extend(X[remainder_samples])
                reduced_Y.extend(Y[remainder_samples])

        return (
            np.asarray(supp_X, dtype=np.float32),
            np.asarray(supp_Y, dtype=np.int64),
            np.asarray(reduced_X, dtype=np.float32),
            np.asarray(reduced_Y, dtype=np.int64),
        )

    return  None, None, X, Y

## Whole tgt dataset is expected to be provided by parameter
## Divide the dataset into train and test
def split_train_test(PARAMS, X, Y):

    XTrain, YTrain = [], []
    val_X, val_Y = [], []

    if not os.path.exists(PARAMS["file_stored_indexes_dict"]): # or not os.path.exists(PARAMS["file_stored_train_set"]):

        XTest, YTest = [], [] 
        supp_X, supp_Y = [], []
        indexes = {"test_1": [], "test_2": [], "supp": []}

        unique_ys = np.unique(Y, axis=0)
        # Do not shuffle to preserve the same order after loading indexes
        # np.random.shuffle(unique_ys)
        for unique_idx, unique_y in enumerate(unique_ys):

            indices = np.argwhere(Y == unique_y).flatten()

            ################ If same classes are repeated, not contemplated #####################
            # Sets cause any duplicated elements to be removed -> We have no repeated elements!
            # test_samples = list(set(indices) - set(train_samples))
            # XTrain.extend(reduced_X[train_samples])
            # YTrain.extend(reduced_Y[train_samples])
            # XTest.extend(reduced_X[test_samples])
            # YTest.extend(reduced_Y[test_samples])
            #####################################################################################

            # In case it is smaller than the minimum needed
            limit_test_classes = max(Constants["TEST_SET_PERC"] * len(unique_ys), Constants["LIMIT_N_WAY_TEST"]) # Starts from 0
            
            if unique_idx >= limit_test_classes:
                XTrain.extend(X[indices])
                YTrain.extend(Y[indices])
            else:
                # if not Constants["ALL_DATASETS"]:
                #     fixed_supp_samples = np.random.choice(indices, PARAMS['spc'], replace=False)
                #     test_samples = list(set(indices) - set(fixed_supp_samples))
                    
                #     supp_X.extend(X[fixed_supp_samples])
                #     supp_Y.extend(Y[fixed_supp_samples])
                #     indexes["supp"].extend(fixed_supp_samples)
        
                # else:
                test_samples = list(set(indices))
            
                XTest.extend(X[test_samples])
                YTest.extend(Y[test_samples])
                indexes["test_1"].extend(test_samples)

        # Else, loaded on "load_datasaets" function
        if not Constants["ALL_DATASETS"]:
            ### Not tested yet on not all datasets
            if Constants["VALIDATION_SRC_SRC_DATA"]:
                XTr, YTr = extract_validation_from_test(XTrain, YTrain, val_X, val_Y)
        
        # Reload it to uniform all iterations # XTrain, YTrain, XTest, YTest, supp_X, supp_Y =  
        supp_split(PARAMS, XTrain, YTrain, XTest, YTest, supp_X, supp_Y, indexes=indexes)

    # else:
    indexes = json.load(open(PARAMS["file_stored_indexes_dict"], 'r'))
    test_samples_prev, test_samples, supp_samples = indexes["test_1"], indexes["test_2"], indexes["supp"]
    train_samples = [i for i, s in enumerate(Y) if i not in test_samples_prev]
    
    # First removal
    XTest_1, YTest_1 = X[test_samples_prev], Y[test_samples_prev]

    # if exp == "exp3":
    limit_test_classes = max(int(Constants["TEST_SET_PERC"] * len(np.unique(Y)) ), Constants["LIMIT_N_WAY_TEST"]) + 1
    len1 = len(np.unique(YTest_1))

    if len1 > limit_test_classes:
        os.remove(PARAMS["file_stored_indexes_dict"])
        print("REMOVING FILE", PARAMS["file_stored_indexes_dict"])
    np.testing.assert_equal(len1 <= limit_test_classes, True, str(len1) + "_" + str(limit_test_classes) +" Make sure Y not corrupt and Y test sets must have the same length")

    # After removing supp samples
    for sample, _ in enumerate(YTest_1):
        if sample not in test_samples and sample not in supp_samples:
            train_samples.append(sample)

    XTrain, YTrain = X[train_samples], Y[train_samples]
    # From the second iter of base X/Y
    XTest, YTest = XTest_1[test_samples], YTest_1[test_samples]
    supp_X, supp_Y = XTest_1[supp_samples], YTest_1[supp_samples]
    if not os.path.exists(PARAMS["file_stored_supp_set_Y"]):
        adjust_n_way_to_supp_set(PARAMS, supp_X, supp_Y)
    
    ## Supp loaded like that because it is changed by n-way for example
    supp_X = np.load(PARAMS["file_stored_supp_set_X"], allow_pickle=True) 
    supp_Y = np.load(PARAMS["file_stored_supp_set_Y"], allow_pickle=True)

    return (
            np.asarray(XTrain, dtype=np.float32),
            np.asarray(YTrain, dtype=np.int64),
            np.asarray(XTest, dtype=np.float32),
            np.asarray(YTest, dtype=np.int64),
            np.asarray(supp_X, dtype=np.float32),
            np.asarray(supp_Y, dtype=np.int64),
            np.asarray(val_X, dtype=np.float32),
            np.asarray(val_Y, dtype=np.int64),
        )

# Extract supp_set from test
def supp_split(
    PARAMS,  XTrain, YTrain, XTest_or, YTest_or, supp_X, supp_Y, indexes
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    

    # supp_X, supp_Y, reduced_X, reduced_Y = extract_support_set(X, Y, samples_per_class, model_type)
    supp_X, supp_Y, XTest, YTest = [], [], [], []
    XTest_ext_r, YTest_ext_r = np.array(XTest_or), np.array(YTest_or)

    unique_ys = np.unique(YTest_ext_r, axis=0)

    # ## Use more available train set  
    # if Constants["ALL_DATASETS"]:
        
    # model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":
    ### With the same seed, the subset chosen is the same as the original paper
    for unique_y in unique_ys:
        indices = np.argwhere(YTest_ext_r == unique_y).flatten()

        if Constants["SMALL_TEST_SET"]:
            # min_size = max(3 * samples_per_class, len(indices)*train_set_perc) #max(len(indices)*train_set_perc, samples_per_class*10)
            min_size = (3 * samples_per_class + 2) 
            indices = indices[:round(min_size)]

        supp_samples = np.random.choice(indices, PARAMS['spc'], replace=False)
        ###### DEBUG ########
        # supp_samples = np.arange(0, samples_per_class)
        #####################

        # Sets cause any duplicated elements to be removed -> We have no repeated elements!
        test_samples = list(set(indices) - set(supp_samples))
        supp_X.extend(XTest_ext_r[supp_samples])
        supp_Y.extend(YTest_ext_r[supp_samples])
        XTest.extend(XTest_ext_r[test_samples])
        YTest.extend(YTest_ext_r[test_samples])
        indexes["test_2"].extend(test_samples)
        indexes["supp"].extend(supp_samples)
        ## Only the necessary for train set...

    # else:
        
        # if Constants["VALIDATION_SRC_SRC_DATA"]:
        #     # Reserve 5% classes for validation
        #     y_classes = np.unique(reduced_Y)
        #     np.random.shuffle(y_classes)
        #     # Reserve a small percentage of classes for validation

        #     for y in range(int(len(y_classes) * Constants["VALIDATION_PERC"])):
        #         indexes = [ind for ind, val in enumerate(reduced_Y) if val == y_classes[y]]
                
        #         XVal.extend([reduced_X[i] for i in indexes])
        #         YVal.extend([reduced_Y[i] for i in indexes])

        #         reduced_X = [reduced_X[i] for i in range(len(reduced_X)) if i not in indexes]
        #         reduced_Y = [reduced_Y[i] for i in range(len(reduced_Y)) if i not in indexes]
        #     reduced_X, reduced_Y = np.array(reduced_X), np.array(reduced_Y)
        #     # Update unique_ys
        #     unique_ys = np.unique(reduced_Y, axis=0)


        # Cases without dataset source
        # supp_X, supp_Y = XTrain, YTrain
        # supp_X, supp_Y = [], []

    ## Moved to outside function
    # adjust_n_way_to_supp_set(PARAMS, supp_X, supp_Y)

    ## Store dictionary indexes in file:
    with open(PARAMS["file_stored_indexes_dict"], 'w') as file:

        for key, value in indexes.items():
            for i in range(len(value)):
                value[i] = int(value[i])
            indexes[key] = value   
        json.dump(indexes, file)

    return (
        np.asarray(XTrain, dtype=np.float32),
        np.asarray(YTrain, dtype=np.int64),
        np.asarray(XTest, dtype=np.float32),
        np.asarray(YTest, dtype=np.int64),
        np.asarray(supp_X, dtype=np.float32),
        np.asarray(supp_Y, dtype=np.int64),
    )

def adjust_n_way_to_supp_set(PARAMS, supp_X, supp_Y):
    
    ## Choose only N-way classes from the tgt dataset
    if Constants["LIMIT_N_WAY_TEST"]:
        unique_ys = np.unique(supp_Y, axis=0)
        n_supp_X, n_supp_Y = [], []

        # Haz un shuffle de las clases
        np.random.shuffle(unique_ys)
        for new_class in range(Constants["LIMIT_N_WAY_TEST"]):


            indices = np.argwhere(supp_Y == unique_ys[new_class]).flatten()
            supp_samples = np.random.choice(indices, PARAMS['spc'], replace=False)

            n_supp_X.extend(np.array(supp_X)[supp_samples])
            n_supp_Y.extend(np.array(supp_Y)[supp_samples])
        supp_X, supp_Y = n_supp_X, n_supp_Y

    np.save(PARAMS["file_stored_supp_set_X"], np.array(supp_X))
    np.save(PARAMS["file_stored_supp_set_Y"], np.array(supp_Y))    

## I must delete the folder to be filled
def debug_images_and_dists(x_target, x_support_set, dists, debug_images):
    import matplotlib.pyplot as plt
    base_path = "DEBUG_IMAGES/" + debug_images
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
        all_supp, all_tgt = x_support_set.reshape(-1, 3, 40, 40), x_target.reshape(-1, 3, 40, 40)
        all_supp, all_tgt = all_supp.permute(0,2,3,1), all_tgt.permute(0,2,3,1)
        
        ## Only 2 examples
        all_supp = all_supp[:10]
        all_tgt = all_tgt[:10]  

        # Save .png images
        for i in range(len(all_supp)):
            img = all_supp[i].cpu().detach().numpy()
            plt.imsave(f"{base_path}support_set_image_{i}.png", img)

        for i in range(len(all_tgt)):
            img = all_tgt[i].cpu().detach().numpy()
            plt.imsave(f"{base_path}target_image_{i}.png", img)

        reduced_dists = dists[:2]
        # Store distances in txt
        with open(f"{base_path}distances.txt", "w") as f:
            for i in range(len(reduced_dists)):
                f.write(f"For batch {i}, supp (rows), querys (cols). Distances: \n{reduced_dists[i].cpu().detach().numpy()}\n")
    



    