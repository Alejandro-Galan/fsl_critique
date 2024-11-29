import gc
import random

import fire, tqdm
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

import datasets.config as config
from datasets.loader import load_supervised_data
from my_utils.train_utils import (
    train_test_split,
    write_plot_results,
    write_tsne_representation,
)
from network.model import (
    CustomCNN,
    ResnetEncoder,
    ResnetEncoderVICReg,
    VggEncoder,
    VggEncoderVICReg,
    MatchingNetwork,
)
    # CustomMatchingNetworkEncoderVICReg

# Seed
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def run_bootstrap(
    *,
    ds_name: str,
    samples_per_class: int,
    min_occurence: int = 50,
    model_type: str = "CustomCNN",
    pretrained: bool = False,
    checkpoint_path: str = "",
    num_runs: int = 5,
    batch_size: int = 16, 
    encoder_features_dim: int = 1600,
    expander_features_dim: int = 1024,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------KNN CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Pretrained: {pretrained}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Number of bootstrap runs: {num_runs}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence)
    X, Y, w2i = data_dict["X"], data_dict["Y"], data_dict["w2i"]
    print(f"Dataset {ds_name} information:")
    print(f"\tTotal number of samples: {len(Y)}")
    print(f"\tNumber of classes: {len(w2i)}")
    print("----------------------------------------------------")

    # 2) SET OUTPUT DIR
    output_dir = config.output_dir / "knn"
    output_dir = output_dir / f"{model_type}-Pretrained{pretrained}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_dir = output_dir / "tsne"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP
    results = []
    for run in range(num_runs):
        print(f"Bootstrap run {run + 1}/{num_runs}")
        # 3.1) Get samples
        XTrain, YTrain, XTest, YTest, XSupp, YSupp = train_test_split(
            X=X,
            Y=Y,
            samples_per_class=samples_per_class,
        )
        # 3.2) Train and test KNN classifier
        accuracy, XTrain_embedded = train_and_test_knn(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            XSupp=XSupp,
            YSupp=YSupp,
            model_type=model_type,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_classes_per_set=len(w2i),
            num_samples_per_class=samples_per_class,
            encoder_features=encoder_features_dim,
        )
        # NOTE: So far, only accuracy is saved
        results.append(accuracy)

        # TESTING MATCHING THIS IS TEMPORAL WILL BE NEEDED?
        if False:
            # 3) SAVE TSNE REPRESENTATION
            tsne_filepath = ""
            if checkpoint_path != "":
                tsne_filepath += checkpoint_path.split("/")[-1] + "-"
            tsne_filepath += (
                f"test_on_{samples_per_class}spc_with{min_occurence}-run{run}.dat"
            )
            tsne_filepath = tsne_dir / tsne_filepath
            write_tsne_representation(
                filepath=tsne_filepath,
                x=XTrain_embedded,
                y=YTrain,
                w2i=w2i,
            )

    # 4) SAVE RESULTS
    print("----------------------------------------------------")
    print("BOOTSTRAP SUMMARY:")
    print(f"\tSamples per class: {samples_per_class}")
    print(f"\tNumber of bootstrap runs: {num_runs}")
    print(f"\tMean accuracy: {np.mean(results):.2f}")
    print(f"\tStandard deviation: {np.std(results):.2f}")
    from_weights = checkpoint_path if pretrained else "-"
    from_weights = "imagenet" if pretrained and checkpoint_path == "" else from_weights
    write_plot_results(
        filepath=output_dir / "results.txt",
        from_weights=from_weights,
        epochs="-",
        batch_size="-",
        results=results,
        samples_per_class=samples_per_class,
    )


############################################# UTILS:


def get_images_representations(*, encoder, X, device, model_type="Original", supp_set=None):
    Y = []
    with torch.no_grad():
        for i, x in enumerate(X):
            x = x.unsqueeze(0).to(device)
            if model_type == "MatchingNetwork":
                print("TODO")
                # y = x.reshape(x.shape[0], -1)
            else:
                y = encoder(x)[0].cpu().detach().numpy()
            Y.append(y)
    return np.asarray(Y).reshape(X.shape[0], -1)

def extend_dimenstions_bs(arr, batch_size):
    arr_expanded = np.expand_dims(arr, axis=0)

    # Duplicamos la información x veces
    arr_duplicated = np.repeat(arr_expanded, batch_size, axis=0)

    return arr_duplicated








def train_and_test_knn(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    XSupp: np.ndarray,
    YSupp: np.ndarray,
    model_type: str = "Flatten",
    pretrained: bool = False,
    checkpoint_path: str = "",
    batch_size: int = 16,
    num_classes_per_set,
    num_samples_per_class,
    encoder_features = 1600,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    # 1) GET IMAGES REPRESENTATIONS
    if model_type == "Flatten" and not pretrained and checkpoint_path == "":
        # No model is used, just the flatten images
        XTrain = XTrain.reshape(XTrain.shape[0], -1)
        XTest = XTest.reshape(XTest.shape[0], -1)


    elif pretrained and not model_type in ["MatchingNetwork"]:
    
        # Pretrained with VICReg
        if model_type in ["CustomCNN", "Resnet34", "Vgg19"] and checkpoint_path != "":
            print(
                f"Using a VICReg-pretrained {model_type} to obtain images' representations"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if model_type == "CustomCNN":
                encoder = CustomCNN(encoder_features=checkpoint["encoder_features"])
            elif model_type == "Resnet34":
                encoder = ResnetEncoderVICReg(
                    encoder_features=checkpoint["encoder_features"]
                )
            elif model_type == "Vgg19":
                encoder = VggEncoderVICReg(
                    encoder_features=checkpoint["encoder_features"]
                )
            # elif model_type == "MatchingNetwork":
            #     params_extra = {"layer_size": 64, "num_channels": 3, 
            #             "nClasses": 0, "image_size": 64}
            #     encoder = CustomMatchingNetworkEncoderVICReg(
            #         encoder_features=checkpoint["encoder_features"],
            #         checkpoint=checkpoint,
            #         params_extra=params_extra
            #     )

        
        # Pretrained with IMAGENET
        elif model_type in ["Resnet34", "Vgg19"] and checkpoint_path == "":
            print(
                f"Using a IMAGENET-pretrained {model_type} to obtain images' representations"
            )
            if model_type == "Resnet34":
                encoder = ResnetEncoder(pretrained=pretrained)
            elif model_type == "Vgg19":
                encoder = VggEncoder(pretrained=pretrained)


        encoder = encoder.to(device)
        encoder.eval()

        XTrain = get_images_representations(
            encoder=encoder, X=torch.from_numpy(XTrain), device=device
        )
        XTest = get_images_representations(
            encoder=encoder, X=torch.from_numpy(XTest), device=device
        )

    elif pretrained and model_type in ["MatchingNetwork"]:
        pass
    else:
        raise NotImplementedError(
            f"Model type {model_type} with pretrained={pretrained} and checkpoint_path={checkpoint_path} not supported"
        )

    if model_type == "MatchingNetwork":

        model =  MatchingNetwork(batch_size=batch_size, keep_prob=torch.FloatTensor(1), num_channels=3,
                                         fce=False, num_classes_per_set=num_classes_per_set,
                                         num_samples_per_class=num_samples_per_class, nClasses = 0, image_size = XTrain.shape[0],
                                         XSupp=XSupp, YSupp=YSupp)
        
        

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

        if not pretrained:
            
            # ## DEBUG            
            # epochs = 1 # TODO move to train.py
            # for epoch in range(epochs):
            #     # Train matching
            #     train_accs, train_loss, optimizer = model.train_match_network(batch_size=batch_size, encoder=model, X=torch.from_numpy(XTrain), Y=torch.from_numpy(YTrain), 
            #                                             device=device, model_type=model_type, samples_per_class=num_samples_per_class, 
            #                                             classes_per_set=num_classes_per_set, optimizer= optimizer)
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, checkpoint_path)
            pass

        elif pretrained:
            checkpoint = torch.load(checkpoint_path.replace("encoder.pt", "_trained_model.pt"), map_location=device)

            # model_state_dict = model.state_dict()
            # filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict}

            # # Update the model's state_dict with the filtered state_dict
            # model_state_dict.update(filtered_state_dict)
            # model.load_state_dict(model_state_dict, strict=False)

            
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


            # if epoch % 100 == 4:
        # Test evaluation
        supp_set = {"imgs": XTrain, "labels": YTrain}
        test_accs, test_loss, test_outs = model.eval_match_network(batch_size=batch_size, encoder=model, 
                                                X_train=torch.from_numpy(XTrain), Y_train=torch.from_numpy(YTrain),
                                                X=torch.from_numpy(XTest), Y=torch.from_numpy(YTest),
                                                samples_per_class=num_samples_per_class, classes_per_set=num_classes_per_set, 
                                                device=device, model_type=model_type, supp_set=supp_set)

        # class_rep = classification_report(y_true=torch.from_numpy(YTest).cpu().tolist(), y_pred=test_outs.cpu(), output_dict=True)

        # XTrain = get_images_representations(
        #     encoder=model, X=torch.from_numpy(XTrain), device=device, model_type=model_type, supp_set=supp_set
        # )
        # XTest = get_images_representations(
        #     encoder=model, X=torch.from_numpy(XTest), device=device, model_type=model_type, supp_set=supp_set
        # )

        # accuracy = 100 * class_rep["accuracy"]
        accuracy = 100 * test_accs
        print(f"Accuracy: {accuracy:.2f}")

        return accuracy, None



        print(f"Train Accuracy: {train_accs:.2f} Test Accuracy: {test_accs:.2f}")

        return test_accs, XTrain

    else:
        # 2) TRAIN AND TEST KNN CLASSIFIER
        knnClassifier = KNeighborsClassifier(n_neighbors=1)
        knnClassifier.fit(XTrain, YTrain)
        predictions = knnClassifier.predict(XTest)
        class_rep = classification_report(
            y_true=YTest, y_pred=predictions, output_dict=True
        )
        # NOTE: So far, only accuracy is saved
        accuracy = 100 * class_rep["accuracy"]
        print(f"Accuracy: {accuracy:.2f}")

        return accuracy, XTrain


if __name__ == "__main__":
    fire.Fire(run_bootstrap)
