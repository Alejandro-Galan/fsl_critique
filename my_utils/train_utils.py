from typing import Tuple

import numpy as np
from sklearn.manifold import TSNE

from my_utils.constants import Constants


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


    if model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":
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


def train_test_split(
    X: np.ndarray, Y: np.ndarray, samples_per_class: int, model_type=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    XTrain, YTrain = [], []
    XTest, YTest = [], []

    # supp_X, supp_Y, reduced_X, reduced_Y = extract_support_set(X, Y, samples_per_class, model_type)
    supp_X, supp_Y, reduced_X, reduced_Y = [], [], X, Y

    try:
        unique_ys = np.unique(reduced_Y, axis=0)
    except:
        breakpoint()


    ## Use more available train set  
    if Constants.ALL_DATASETS:
        # model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":
        ### With the same seed, the subset chosen is the same as the original paper
        supp_X, supp_Y = [], []
        train_set_perc = 0.05
        for unique_y in unique_ys:
            indices = np.argwhere(reduced_Y == unique_y).flatten()
            if Constants.SMALL_TEST_SET:
                # min_size = max(3 * samples_per_class, len(indices)*train_set_perc) #max(len(indices)*train_set_perc, samples_per_class*10)
                min_size = (3 * samples_per_class + 2) 
                indices = indices[:round(min_size)]

            train_samples = np.random.choice(indices, samples_per_class, replace=False)
            ###### DEBUG ########
            # train_samples = np.arange(0, samples_per_class)
            #####################

            # Sets cause any duplicated elements to be removed -> We have no repeated elements!
            test_samples = list(set(indices) - set(train_samples))

            supp_X.extend(reduced_X[train_samples])
            supp_Y.extend(reduced_Y[train_samples])
            XTest.extend(reduced_X[test_samples])
            YTest.extend(reduced_Y[test_samples])
            ## Only the necessary for train set...

    else:
        for unique_y in unique_ys:
            indices = np.argwhere(reduced_Y == unique_y).flatten()
            train_samples = np.random.choice(indices, samples_per_class, replace=False)
            # Sets cause any duplicated elements to be removed -> We have no repeated elements!
            test_samples = list(set(indices) - set(train_samples))
            XTrain.extend(reduced_X[train_samples])
            YTrain.extend(reduced_Y[train_samples])
            XTest.extend(reduced_X[test_samples])
            YTest.extend(reduced_Y[test_samples])
        # Cases without dataset source
        supp_X, supp_Y = XTrain, YTrain

    ## Choose only N-way classes
    if Constants.LIMIT_N_WAY:
        unique_ys = np.unique(supp_Y, axis=0)
        n_supp_X, n_supp_Y = [], []

        # Haz un shuffle de las clases
        np.random.shuffle(unique_ys)
        for new_class in range(Constants.LIMIT_N_WAY):
            indices = np.argwhere(supp_Y == unique_ys[new_class]).flatten()
            supp_samples = np.random.choice(indices, samples_per_class, replace=False)

            n_supp_X.extend(np.array(supp_X)[supp_samples])
            n_supp_Y.extend(np.array(supp_Y)[supp_samples])
        supp_X, supp_Y = n_supp_X, n_supp_Y


    return (
        np.asarray(XTrain, dtype=np.float32),
        np.asarray(YTrain, dtype=np.int64),
        np.asarray(XTest, dtype=np.float32),
        np.asarray(YTest, dtype=np.int64),
        np.asarray(supp_X, dtype=np.float32),
        np.asarray(supp_Y, dtype=np.int64),
    )
