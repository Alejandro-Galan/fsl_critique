import gc, importlib
import random

import fire, wandb, sys
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report
from torchinfo import summary
from models.MatchingNetwork import MatchingNetwork
from models.PrototypicalNetwork import PrototypicalNetwork

import datasets.config as config
import pandas as pd

from datasets.loader import load_supervised_data, convert_clustering_experiment
from utils.generators import supervised_data_generator
from utils.train_utils import split_train_test, write_plot_results
from models.FewShotModel import FewShotTrain
from network.model import (
    ResnetClassifier,
    SupervisedClassifier,
    VggClassifier,
)

from utils import constants
importlib.reload(constants)
from utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp_name = str(sys.argv[1])
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)

Constants = Constants_c.Constants

if Constants["DEACTIVATE_WANDB"]:
    wandb.init(mode="disabled")


import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './.cache'




def run_bootstrap(
    *,
    ds_name: str,
    samples_per_class: int,
    min_occurence: int = 50,
    model_type: str = "", 
    epochs: int = 2,
    batch_size: int = 16,
    num_runs: int = 5,
    encoder_features_dim: int = 1600,
    episodes=1000,

):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------SUPERVISED " + model_type + " CLASSIFICATION EXPERIMENT--------")
    print(f"src_Datasets: {[i for i in Constants['DATASETS_NAMES'] if i != ds_name]}")
    print(f"tgt_Dataset: {[ds_name]}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of bootstrap runs: {num_runs}")
    print(f"NWAY TRAIN: {Constants['LIMIT_N_WAY_TRAIN']}")
    print(f"NWAY TEST: {Constants['LIMIT_N_WAY_TEST']}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    ## Use source datastes



    # 3) RUN BOOTSTRAP
    results = []
    train_best_acc = 0.0
    for run in range(num_runs):

        string_id_base, string_id_ft, src_datasets, src_input_size = Const_c.get_experiment_id(PARAMS=Constants, boots_iter=run, return_extra_params=True)

        if Const_c.all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter=run):
            continue
        

        path_weights_base = "WEIGHTS/" + string_id_base
        path_weights_base_pretrained = path_weights_base + "_trained_model.pt"

        print("Base weights", path_weights_base)

        wandb.config.update({"src_datasets": src_datasets, "tgt_dataset": ds_name, "samples_per_class": samples_per_class, 
                            "num_total_episodes": Constants["EPISODES"], "batch_size": Constants["BATCH_SIZE"],
                            "num_bootstrap_iters": Constants["BOOTSTRAP_ITERS"], "input_size": src_input_size,
                            "fixed_support": Constants["USE_ORIGINAL_FIXED_SUPP_SET"], "model_type": model_type,
                            "n_way_train": Constants["LIMIT_N_WAY_TRAIN"], "lr": Constants["lr"],
                            "n_way_test": Constants["LIMIT_N_WAY_TEST"]})
        
        # Reproducibility
        torch.manual_seed(run)
        random.seed(run)
        np.random.seed(run)


        PARAMS = {'ds_name': ds_name,"spc": samples_per_class, "nway_test": Constants["LIMIT_N_WAY_TEST"],
                "nway_train": Constants['LIMIT_N_WAY_TRAIN'], "boots_iter": run}
        file_stored_test_set = "utils/stored_sets/"  + PARAMS['ds_name'] + "/_spc_" + str(PARAMS['spc']) + "n_way_test_" + str(PARAMS['nway_test']) \
                            + "_n_way_train_" + str(PARAMS['nway_train']) + "/_boots_iter_" + str(PARAMS['boots_iter']) + "_test_indexes.txt"
        # file_stored_train_set = "utils/stored_sets/" + PARAMS['ds_name'] + "_boots_iter_" + str(PARAMS['boots_iter']) + "_train.txt"
        # PARAMS['file_stored_test_set']  = file_stored_test_set
        # PARAMS['file_stored_train_set'] = file_stored_train_set
        PARAMS['file_stored_indexes_dict'] = file_stored_test_set
        fs_set = "utils/stored_sets/" + PARAMS['ds_name'] + "/_spc_" + str(PARAMS['spc']) + "n_way_test_" + str(PARAMS['nway_test']) \
                            + "_n_way_train_" + str(PARAMS['nway_train']) + "/_boots_iter_" + str(PARAMS['boots_iter'])
        PARAMS['file_stored_supp_set_X']  = fs_set + "_X_" + "_supp.npy"
        PARAMS['file_stored_supp_set_Y']  = fs_set + "_Y_" + "_supp.npy"

        os.makedirs(os.path.dirname(fs_set), exist_ok=True)
        
        pretrained_sources = os.path.exists(path_weights_base_pretrained) if Constants["ReusePretrained"] else False
        if Constants["ALL_DATASETS"] or Constants["NoSrcDataset"]:
            pretrained_sources = True if Constants["NoSrcDataset"] else pretrained_sources

            # If no validation src paramenter, Xval is empty
            data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence, 
                                            all_datasets=Constants["ALL_DATASETS"], pretrained_sources=pretrained_sources, boots_iter=run)
            
            _, _, XTest, YTest, XSupp, YSupp, _, _ = split_train_test(
                PARAMS=PARAMS,
                X=data_dict["X_tgt"],
                Y=data_dict["Y_tgt"],
            )

            X_val, Y_val = data_dict["X_val"], data_dict["Y_val"]
            
            XTrain, YTrain, w2i = data_dict["X_src"], data_dict["Y_src"], data_dict["w2i"]

            if len(np.unique(YTest)) == 0:
                breakpoint()
            
        else:
            data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence, boots_iter=run)

            XTrain, YTrain, XTest, YTest, XSupp, YSupp, XVal, YVal = split_train_test(
                PARAMS=PARAMS,
                X=data_dict["X_tgt"], 
                Y=data_dict["Y_tgt"],
            )

            X_val, Y_val, w2i = data_dict["X_val"], data_dict["Y_val"], data_dict["w2i"]

            # print(f"\tTotal number of samples: {len(Y)}")
            print(f"\tNumber of classes: {len(w2i)}")


        #### In case of clustering experiment
        if Constants["CLUSTERING"]:
            wandb.log({"CLUSTERING": Constants["CLUSTERING"], "M-labels-SRC": Constants["M-labels-SRC"]})            
            if not pretrained_sources: # Else the weights are right, could be reused
                XTrain, YTrain = convert_clustering_experiment(XTrain, YTrain, Constants=Constants, boots_iter=run)


        print(f"Dataset {ds_name} information:")
        print("----------------------------------------------------")

        # 2) SET OUTPUT DIR
        output_dir = config.output_dir / "supervised"
        output_dir = output_dir / f"{model_type}" #-Pretrained{pretrained}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Bootstrap run {run + 1}/{num_runs}")
        if not Constants["DEACTIVATE_WANDB"]:
            wandb.log({"bootstrap_iter": run})            

        ### Not anymore, unified with the stored files 
        # if not Constants["ALL_DATASETS"]:
        #     # 3.1) Get samples
        #     XTrain, YTrain, XTest, YTest, XSupp, YSupp, X_val, Y_val = train_test_split(X=X, Y=Y, samples_per_class=samples_per_class, model_type=model_type)


        print(f"\tTotal number of samples: {len(YTrain) + len(YTest), len(Y_val)}")
        print(f"\tVal samples: {len(Y_val)}")
        print(f"\tTest samples: {len(YTest)}")
        
        print(f"\tNumber of Source classes (m) [Train+Val (Train, Val)]: {len(np.unique(YTrain)) + len(np.unique(Y_val))} ({len(np.unique(YTrain))}, {len(np.unique(Y_val))})")
        print(f"\tNumber of Target classes: {len(np.unique(YTest))}")
        if len(np.unique(YTest)) > len(np.unique(YTrain)) and not Constants["LIMIT_N_WAY_TRAIN"]:
            print("NUMBER OF CLASSES IN TRAIN LOWER THAN TEST")
            exit()
        if Constants["LIMIT_N_WAY_TEST"] or Constants["LIMIT_N_WAY_TRAIN"]:
            if Constants["LIMIT_N_WAY_TEST"] > Constants["LIMIT_N_WAY_TRAIN"]:
                print("NUMBER OF N-WAY IN TRAIN LOWER THAN TEST")
                exit()
            print("\tN-WAY Train: ", Constants["LIMIT_N_WAY_TRAIN"], "\n\tN-WAY Test: ", Constants["LIMIT_N_WAY_TEST"])
        if Constants["LIMIT_N_WAY_TEST"] < 2 or Constants["LIMIT_N_WAY_TRAIN"] < 2:
            print("NUMBER OF N-WAY LOWER THAN 2, makes no sense")
            exit()
            
        if "exp6" in exp or "exp6" in exp_name:
            variables = {name: sys.getsizeof(obj) for name, obj in globals().items()}
            variables_ordenadas = sorted(variables.items(), key=lambda x: x[1], reverse=True)
            print("GLOBALES", variables_ordenadas)
            variables = {name: sys.getsizeof(obj) for name, obj in locals().items()}
            variables_ordenadas = sorted(variables.items(), key=lambda x: x[1], reverse=True)
            print("LOCALES", variables_ordenadas)
            breakpoint()

        # 3.2) Train and test model
        class_rep, train_best_acc = train_and_test_model(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            XSupp=XSupp, 
            YSupp=YSupp,
            model_type=model_type,
            batch_size=batch_size,
            epochs=epochs,
            samples_per_class=samples_per_class,
            best_acc=train_best_acc,
            encoder_features = encoder_features_dim,
            path_weights_base = path_weights_base,
            X_val=torch.from_numpy(X_val),
            Y_val=torch.from_numpy(Y_val),
        )
        # NOTE: So far, only accuracy is saved
        if model_type != "MatchingNetwork" and model_type != "PrototypicalNetwork":
            accuracy = 100 * class_rep["accuracy"]
        else:
            accuracy = 100 * train_best_acc

        results.append(accuracy)

        print("EEEEXP", exp)
        if "exp6" in exp or "exp6" in exp_name:
            variables = {name: sys.getsizeof(obj) for name, obj in globals().items()}
            variables_ordenadas = sorted(variables.items(), key=lambda x: x[1], reverse=True)
            print("GLOBALES", variables_ordenadas)
            variables = {name: sys.getsizeof(obj) for name, obj in locals().items()}
            variables_ordenadas = sorted(variables.items(), key=lambda x: x[1], reverse=True)
            print("LOCALES", variables_ordenadas)
            breakpoint()

    if results:
        # 4) SAVE RESULTS
        print("----------------------------------------------------")
        print("BOOTSTRAP SUMMARY:")
        print(f"\tSamples per class: {samples_per_class}")
        print(f"\tNumber of bootstrap runs: {num_runs}")
        print(f"\tMean accuracy: {np.mean(results):.2f}")
        print(f"\tStandard deviation: {np.std(results):.2f}")
        write_plot_results(
            filepath=output_dir / "results.txt",
            from_weights="-",
            epochs=epochs,
            batch_size=batch_size,
            results=results,
            samples_per_class=samples_per_class,
        )


############################################# UTILS:


def test_model(*, model, X, Y, device):
    YHAT = []

    model.eval()
    with torch.no_grad():
        for x in tqdm.tqdm(X, position=0, leave=True):
            x = x.unsqueeze(0).to(device)
            yhat = model(x)[0]
            yhat = yhat.softmax(dim=0)
            yhat = torch.argmax(yhat, dim=0)
            YHAT.append(yhat.item())

    class_rep = classification_report(y_true=Y.tolist(), y_pred=YHAT, output_dict=True)
    accuracy = 100 * class_rep["accuracy"]

    return accuracy, class_rep


def train_and_test_model(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    XSupp: np.ndarray,
    YSupp: np.ndarray,
    model_type: str = "", 
    batch_size: int = 16,
    epochs: int = 150,
    samples_per_class = None,
    best_acc = 0.0,
    encoder_features = 1600,
    path_weights_base="",
    X_val=None, 
    Y_val=None,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    if model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":
        if not Constants["LIMIT_N_WAY_TRAIN"] or not Constants["LIMIT_N_WAY_TEST"]:
            num_c_tr = min(len(np.unique(YTrain)), len(np.unique(YTest)))
            num_c_ts = num_c_tr
        else: 
            num_c_tr = Constants["LIMIT_N_WAY_TRAIN"]
            num_c_ts = Constants["LIMIT_N_WAY_TEST"]

        if model_type == "MatchingNetwork":
            model = MatchingNetwork(batch_size=batch_size, encoder_features=encoder_features,
                                keep_prob=torch.FloatTensor(1), num_channels=3,
                                fce=False,
                                num_classes_per_set_train=num_c_tr, 
                                num_classes_per_set_test=num_c_ts,
                                num_samples_per_class=samples_per_class,
                                nClasses=0, image_size = Constants["INPUT_SIZE"][list(Constants['TGT_DATASETS'].keys())[0]][0], best_accuracy=best_acc)
        elif model_type == "PrototypicalNetwork":
            model = PrototypicalNetwork(x_dim=3, hid_dim=64, z_dim=64)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Constants["lr"], weight_decay=Constants["weight_decay"])
        # Reduce lr per half every 20 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Constants["step_size"], gamma=0.5)


    else:
        raise NotImplementedError(
            f"Model type {model_type} not supported"
        )
    model = model.to(device)
    
    
    if model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":

        metrics = {'XSupp': XSupp, 'YSupp': YSupp, 'best_accuracy': 0.0, 'PATIENCE': Constants["PATIENCE"]}

        path_weights_base_pretrained = path_weights_base + "_trained_model.pt"
        
        # This is the "pretraining" on the source dataset
        # if Constants["ALL_DATASETS"]:
        ## Train from 0
        if not Constants["NoSrcDataset"]:
            if not os.path.exists(path_weights_base_pretrained) or not Constants["ReusePretrained"]:
                os.makedirs(os.path.dirname(path_weights_base), exist_ok=True)
                for epoch in range(epochs):
                    # Train matching
                    if not Constants["DEACTIVATE_WANDB"]:
                        wandb.log({"epoch": epoch})

                    train_best_acc, train_loss, optimizer, best_class_rep = FewShotTrain.train_few_shot_net(batch_size=batch_size, encoder=model, 
                                                            X=torch.from_numpy(XTrain), Y=torch.from_numpy(YTrain), 
                                                            device=device, model_type=model_type, samples_per_class=samples_per_class, 
                                                            classes_per_set=num_c_tr, optimizer= optimizer, checkpoint_path=path_weights_base,
                                                            X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest), metrics=metrics,
                                                            scheduler=scheduler, X_val=X_val, Y_val=Y_val)
                    if train_best_acc > metrics['best_accuracy']:
                        metrics['best_accuracy'] = train_best_acc
                        # model.best_class_rep = best_class_rep            


        ## Trained on source datasets, fine-tune on tgt dataset
        if Constants["FineTuning"]:
            print("\nFineTune!!")
            best_acc_train = metrics['best_accuracy']
            metrics ['best_accuracy'] = 0.0

            # Load best previous model
            optimizer = torch.optim.Adam(model.parameters(), lr=Constants["lrFineTuning"], weight_decay=1e-6)

            # if Constants["ALL_DATASETS"] and ### Why was I doing that?? 
            if not Constants["NoSrcDataset"]:
                # checkpoint = torch.load(checkpoint_path.replace("encoder.pt", "_trained_model.pt"), map_location=device)
                checkpoint = torch.load(path_weights_base_pretrained, map_location=device)

                # (Initialize only the encoder)
                model.load_state_dict(checkpoint['model_state_dict'])

            for epoch in range(Constants["epochsFineTuning"]):
                train_best_acc, train_loss, optimizer, best_class_rep = FewShotTrain.finetune_few_shot_net(batch_size=batch_size, encoder=model, 
                                        X=torch.from_numpy(XSupp), Y=torch.from_numpy(YSupp), 
                                        device=device, model_type=model_type, samples_per_class=samples_per_class, 
                                        classes_per_set=num_c_ts, optimizer= optimizer, checkpoint_path=path_weights_base,
                                        X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest), metrics=metrics, ft_epoch_num=epoch )

            print("BEST TRAIN ACC", best_acc_train, "AFTER FINETUNING", metrics['best_accuracy']) 

        return None, metrics['best_accuracy'] 


if __name__ == "__main__":
    fire.Fire(run_bootstrap)
