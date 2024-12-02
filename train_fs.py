import gc
import random

import fire, wandb
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import classification_report
from torchinfo import summary
from models.MatchingNetwork import MatchingNetwork
from models.PrototypicalNetwork import PrototypicalNetwork

import datasets.config as config
from datasets.loader import load_supervised_data
from my_utils.generators import supervised_data_generator
from my_utils.train_utils import train_test_split, write_plot_results
from models.FewShotModel import FewShotTrain
from network.model import (
    ResnetClassifier,
    SupervisedClassifier,
    VggClassifier,
)

from my_utils.constants import Constants

if Constants.DEACTIVATE_WANDB:
    wandb.init(mode="disabled")


import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './.cache'

# Seed
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


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
    print(f"src_Datasets: {Constants.DATASETS_NAMES}")
    print(f"tgt_Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of bootstrap runs: {num_runs}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    ## Use source datastes
    src_datasets = "_".join(Constants.DATASETS_NAMES - Constants.TGT_DATASETS.keys())
    src_input_size = str(Constants.INPUT_SIZE[0]) + "x" + str(Constants.INPUT_SIZE[1]) 
    path_weights_base = "WEIGHTS/" + model_type + "--" + str(Constants.EPISODES) + "_Episodes--" \
                                + src_input_size + "_INPUT_SIZE/" \
                                + src_datasets + "--" + str(Constants.LIMIT_N_WAY) + "_NWAY--" \
                                + str(Constants.BATCH_SIZE) + "_Batch_Size--" \
                                + str(samples_per_class) + "_KSamples_per_Class--" \
                                + str(Constants.lr) + "_lr--" 
                                # + str(Constants.BOOTSTRAP_ITERS) + "_Boots_Iters"
    path_weights_base_pretrained = path_weights_base + "_trained_model.pt"

    
    if Constants.ALL_DATASETS:
        pretrained_sources = os.path.exists(path_weights_base_pretrained)

        data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence, 
                                         all_datasets=Constants.ALL_DATASETS, pretrained_sources=pretrained_sources)
        _, _, XTest, YTest, XSupp, YSupp = train_test_split(
            X=data_dict["X_test"],
            Y=data_dict["Y_test"],
            samples_per_class=samples_per_class,
            model_type=model_type,
        )
        
        
        XTrain, YTrain, w2i = data_dict["X_train"], data_dict["Y_train"], data_dict["w2i"]
        print(f"\tTotal number of samples: {len(YTrain) + len(YTest)}")
        print(f"\tNumber of Source classes: {len(np.unique(YTrain))}")
        print(f"\tNumber of Target classes: {len(np.unique(YTest))}")
        if len(np.unique(YTest)) > len(np.unique(YTrain)) and not Constants.LIMIT_N_WAY:
            print("NUMBER OF CLASSES IN TRAIN LOWER THAN TEST")
            exit()
        if Constants.LIMIT_N_WAY:
            print("\tN-WAY: ", Constants.LIMIT_N_WAY)
    ## The whole target dataset to train/test
    else:
        data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence)
        X, Y, w2i = data_dict["X"], data_dict["Y"], data_dict["w2i"]
        print(f"\tTotal number of samples: {len(Y)}")
        print(f"\tNumber of classes: {len(w2i)}")
    print(f"Dataset {ds_name} information:")
    print("----------------------------------------------------")

    # 2) SET OUTPUT DIR
    output_dir = config.output_dir / "supervised"
    output_dir = output_dir / f"{model_type}" #-Pretrained{pretrained}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP
    results = []
    train_best_acc = 0.0
    for run in range(num_runs):
        print(f"Bootstrap run {run + 1}/{num_runs}")
        if not Constants.DEACTIVATE_WANDB:
            wandb.log({"bootstrap_iter": run})

        if not Constants.ALL_DATASETS:
            # 3.1) Get samples
            XTrain, YTrain, XTest, YTest, XSupp, YSupp = train_test_split(X=X, Y=Y, samples_per_class=samples_per_class, model_type=model_type)

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
            episodes=episodes,
            path_weights_base = path_weights_base,
        )
        # NOTE: So far, only accuracy is saved
        if model_type != "MatchingNetwork" and model_type != "PrototypicalNetwork":
            accuracy = 100 * class_rep["accuracy"]
        else:
            accuracy = 100 * train_best_acc

        results.append(accuracy)

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
    episodes=1000,
    path_weights_base="",
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    if model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":
        if not Constants.LIMIT_N_WAY:
            num_c = min(len(np.unique(YTrain)), len(np.unique(YTest)))
        else: 
            num_c = Constants.LIMIT_N_WAY

        if model_type == "MatchingNetwork":
            model = MatchingNetwork(batch_size=batch_size, encoder_features=encoder_features,
                                keep_prob=torch.FloatTensor(1), num_channels=3,
                                fce=False,
                                num_classes_per_set=num_c,
                                num_samples_per_class=samples_per_class,
                                nClasses=0, image_size = Constants.INPUT_SIZE[0], best_accuracy=best_acc,
                                XSupp=XSupp, YSupp=YSupp)
        elif model_type == "PrototypicalNetwork":
            model = PrototypicalNetwork(x_dim=3, hid_dim=64, z_dim=64)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Constants.lr, weight_decay=1e-6)


    else:
        raise NotImplementedError(
            f"Model type {model_type} not supported"
        )
    model = model.to(device)
    
    
    if model_type == "MatchingNetwork" or model_type == "PrototypicalNetwork":

        metrics = {'XSupp': XSupp, 'YSupp': YSupp, 'best_accuracy': 0.0, 'PATIENCE': Constants.PATIENCE}

        path_weights_base_pretrained = path_weights_base + "_trained_model.pt"
        
        # This is the "pretraining" on the source dataset
        if Constants.ALL_DATASETS:
            ## Train from 0
            if not os.path.exists(path_weights_base_pretrained):
                os.makedirs(os.path.dirname(path_weights_base), exist_ok=True)
                for epoch in range(epochs):
                    # Train matching
                    if not Constants.DEACTIVATE_WANDB:
                        wandb.log({"epoch": epoch})

                    train_best_acc, train_loss, optimizer, best_class_rep = FewShotTrain.train_few_shot_net(batch_size=batch_size, encoder=model, 
                                                            X=torch.from_numpy(XTrain), Y=torch.from_numpy(YTrain), 
                                                            device=device, model_type=model_type, samples_per_class=samples_per_class, 
                                                            classes_per_set=num_c, optimizer= optimizer, checkpoint_path=path_weights_base,
                                                            X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest), episodes=episodes, metrics=metrics )
                    if train_best_acc > metrics['best_accuracy']:
                        metrics['best_accuracy'] = train_best_acc
                        # model.best_class_rep = best_class_rep            


        ## Trained on source datasets, fine-tune on tgt dataset
        if Constants.FineTuning:
            print("\nFineTune!!")
            best_acc_train = metrics['best_accuracy']
            metrics ['best_accuracy'] = 0.0

            # Load best previous model
            optimizer = torch.optim.Adam(model.parameters(), lr=Constants.lrFineTuning, weight_decay=1e-6)
            if Constants.ALL_DATASETS:
                # checkpoint = torch.load(checkpoint_path.replace("encoder.pt", "_trained_model.pt"), map_location=device)
                checkpoint = torch.load(path_weights_base_pretrained, map_location=device)

                # (Initialize only the encoder)
                model.load_state_dict(checkpoint['model_state_dict'])

            for epoch in range(Constants.epochsFineTuning):
                train_best_acc, train_loss, optimizer, best_class_rep = FewShotTrain.finetune_few_shot_net(batch_size=batch_size, encoder=model, 
                                        X=torch.from_numpy(XSupp), Y=torch.from_numpy(YSupp), 
                                        device=device, model_type=model_type, samples_per_class=samples_per_class, 
                                        classes_per_set=num_c, optimizer= optimizer, checkpoint_path=path_weights_base,
                                        X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest), episodes=episodes, metrics=metrics )

            print("BEST TRAIN ACC", best_acc_train, "AFTER FINETUNING", metrics['best_accuracy']) 
        
        return None, metrics['best_accuracy'] 


if __name__ == "__main__":
    fire.Fire(run_bootstrap)
