"""
Main entry point for Few-Shot Learning bootstrap experiments.

Usage:
    python train_fs.py <exp_name> <exp> <full_name> [fire args]
"""

import gc
import importlib
import os
import random
import sys
import warnings

import fire
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import classification_report

import datasets.config as config
from datasets.loader import (
    check_SOTA_case,
    convert_clustering_experiment,
    get_params_for_loading_datasets,
    load_supervised_data,
)
from models.FewShotModel import FewShotTrain
from models.MatchingNetwork import MatchingNetwork
from models.PrototypicalNetwork import PrototypicalNetwork
from models.RelationNetwork import RelationNetwork
from network.model import ResnetClassifier, SupervisedClassifier, VggClassifier
from utils import constants
from utils.constants import Const_c
from utils.train_utils import split_train_test, write_plot_results

importlib.reload(constants)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./.cache"

# ---------------------------------------------------------------------------
# Global experiment constants (loaded once at import time)
# ---------------------------------------------------------------------------
exp_name  = str(sys.argv[1])
exp       = str(sys.argv[2])
full_name = str(sys.argv[3])

Constants_c = Const_c(exp, full_name)
Constants   = Constants_c.Constants



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
    episodes: int = 1000,
):
    """Run a bootstrap evaluation of a few-shot model."""
    torch.cuda.empty_cache()
    gc.collect()

    _print_experiment_header(ds_name, samples_per_class, min_occurence, model_type, epochs, batch_size, num_runs)

    accumulated_history = pd.DataFrame()
    results = []
    train_best_acc = 0.0

    path_logs     = Const_c.get_logs_csv_path(exp, exp_name, ds_name, Constants)
    starting_iter = _get_starting_bootstrap_iter(exp, path_logs, Constants)
    if starting_iter > 0:
        accumulated_history = _load_existing_history(path_logs, Constants)

    for run in range(starting_iter, num_runs):
        string_id_base, _, src_datasets, src_input_size = Const_c.get_experiment_id(
            PARAMS=Constants, boots_iter=run, return_extra_params=True
        )

        if _should_skip_run(exp, exp_name, ds_name, run, string_id_base):
            continue

        path_weights_base            = f"WEIGHTS/{string_id_base}"
        path_weights_base_pretrained = path_weights_base + "_trained_model.pt"
        print("Base weights:", path_weights_base)

        _set_random_seeds(run)

        PARAMS = get_params_for_loading_datasets(src_datasets, ds_name, samples_per_class, run, Constants=Constants)
        data   = _load_data(ds_name, min_occurence, samples_per_class, run, PARAMS)

        XTrain, YTrain = data["XTrain"], data["YTrain"]
        XTest,  YTest  = data["XTest"],  data["YTest"]
        XSupp,  YSupp  = data["XSupp"],  data["YSupp"]
        X_val,  Y_val  = data["X_val"],  data["Y_val"]

        if Constants["CLUSTERING"] and not data.get("pretrained_sources"):
            XTrain, YTrain = _apply_clustering(XTrain, YTrain, PARAMS, path_weights_base_pretrained)

        output_dir = _prepare_output_dir(model_type)

        print(f"Bootstrap run {run + 1}/{num_runs}")
        XTrain, YTrain, XTest, YTest = filter_per_number_samples(
            XTrain, YTrain, XTest, YTest, samples_per_class
        )
        _assert_input_sizes(XTrain, XTest)
        _print_split_summary(YTrain, YTest, Y_val)
        _validate_nway_config(YTrain, YTest)

        _, train_best_acc, metrics = train_and_test_model(
            XTrain=XTrain, YTrain=YTrain,
            XTest=XTest,   YTest=YTest,
            XSupp=XSupp,   YSupp=YSupp,
            model_type=model_type,
            batch_size=batch_size,
            epochs=epochs,
            samples_per_class=samples_per_class,
            best_acc=train_best_acc,
            encoder_features=encoder_features_dim,
            path_weights_base=path_weights_base,
            X_val=torch.from_numpy(X_val),
            Y_val=torch.from_numpy(Y_val),
            reuse_logs=True,
            boots_iter=run,
        )

        accuracy = 100 * train_best_acc
        results.append(accuracy)

        run_history = _build_run_history_row(
            run=run,
            accuracy=accuracy,
            ds_name=ds_name,
            src_datasets=src_datasets,
            src_input_size=src_input_size,
            samples_per_class=samples_per_class,
            model_type=model_type,
            metrics=metrics,
        )
        accumulated_history = pd.concat([accumulated_history, pd.DataFrame([run_history])], ignore_index=True)
        _save_history(accumulated_history, path_logs)

    if results:
        _print_and_save_summary(results, samples_per_class, num_runs, epochs, batch_size, output_dir)

    return accumulated_history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_test_model(
    *,
    XTrain, YTrain, XTest, YTest, XSupp, YSupp,
    model_type: str = "",
    batch_size: int = 16,
    epochs: int = 150,
    samples_per_class=None,
    best_acc: float = 0.0,
    encoder_features: int = 1600,
    path_weights_base: str = "",
    X_val=None,
    Y_val=None,
    reuse_logs: bool = True,
    boots_iter: int = 0,
):
    torch.cuda.empty_cache()
    gc.collect()

    if model_type not in Constants["AllowedModels"]:
        raise NotImplementedError(f"Model type '{model_type}' not supported")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    num_c_tr, num_c_ts = _resolve_nway(YTrain, YTest)
    model, optimizer, scheduler = _build_model_and_optimizer(
        model_type, batch_size, encoder_features, samples_per_class,
        num_c_tr, num_c_ts, best_acc, device
    )

    metrics = {
        "XSupp": XSupp, "YSupp": YSupp,
        "best_accuracy": 0.0,
        "PATIENCE": Constants["PATIENCE"],
    }
    path_weights_pretrained = path_weights_base + "_trained_model.pt"

    if not Constants["NoSrcDataset"]:
        _pretrain(
            model, optimizer, scheduler,
            XTrain, YTrain, XTest, YTest, X_val, Y_val,
            batch_size, epochs, num_c_tr, samples_per_class,
            model_type, path_weights_base, path_weights_pretrained, metrics, device,
        )

    if Constants["FineTuning"]:
        _finetune(
            model, XSupp, YSupp, XTest, YTest,
            batch_size, num_c_ts, samples_per_class,
            model_type, path_weights_base, path_weights_pretrained,
            metrics, reuse_logs, boots_iter, device,
        )
        print(f"AFTER FINETUNING best_accuracy: {metrics['best_accuracy']:.4f}")

    metrics["ft_eval_acc"] = float(metrics.get("ft_eval_acc", metrics["best_accuracy"]))
    metrics["ft_acc"] = float(metrics.get("ft_acc", metrics["best_accuracy"]))
    return None, metrics["best_accuracy"], metrics


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_per_number_samples(XTrain, YTrain, XTest, YTest, samples_per_class):
    """Drop classes with fewer than `samples_per_class` examples."""
    if not Constants["EXCLUDE_CLASSES_WITH_LESS_SAMPLES"]:
        return XTrain, YTrain, XTest, YTest
    XTrain, YTrain = _filter_set(XTrain, YTrain, samples_per_class, "Train")
    XTest,  YTest  = _filter_set(XTest,  YTest,  samples_per_class, "Test")
    return XTrain, YTrain, XTest, YTest


def _filter_set(X, Y, samples_per_class, label: str):
    valid = [c for c in np.unique(Y) if (Y == c).sum() >= samples_per_class]
    print(f"\nREDUCED {label}: {len(valid)} from {len(np.unique(Y))}")
    mask = np.isin(Y, valid)
    return X[mask], Y[mask]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_model(*, model, X, Y, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for x in tqdm.tqdm(X, position=0, leave=True):
            yhat = model(x.unsqueeze(0).to(device))[0].softmax(dim=0)
            predictions.append(torch.argmax(yhat, dim=0).item())
    report = classification_report(y_true=Y.tolist(), y_pred=predictions, output_dict=True)
    return 100 * report["accuracy"], report


# ---------------------------------------------------------------------------
# Private helpers – experiment lifecycle
# ---------------------------------------------------------------------------

def _print_experiment_header(ds_name, spc, min_occ, model_type, epochs, batch_size, num_runs):
    src = [i for i in Constants["DATASETS_NAMES"] if i != ds_name]
    print(f"-------- SUPERVISED {model_type} CLASSIFICATION EXPERIMENT --------")
    print(f"src_Datasets: {src}")
    print(f"tgt_Dataset:  {[ds_name]}")
    print(f"Samples/class: {spc}  |  Min occurence: {min_occ}")
    print(f"Epochs: {epochs}  |  Batch size: {batch_size}  |  Runs: {num_runs}")
    print(f"NWAY TRAIN: {Constants['LIMIT_N_WAY_TRAIN']}  |  NWAY TEST: {Constants['LIMIT_N_WAY_TEST']}")
    print("------------------------------------------------------------------")


def _get_starting_bootstrap_iter(exp, path_logs, Constants) -> int:
    """Return the index of the first bootstrap iteration that still needs to run."""
    if exp == "exp3":
        last_dist = Const_c.get_distances_path(
            Constants=Constants,
            sufix_path="_after_ft_distances",
            boots_iter=Constants["BOOTSTRAP_ITERS"] - 1,
        )
        if not os.path.exists(last_dist):
            return 0

    if os.path.exists(path_logs):
        df         = pd.read_csv(path_logs)
        df         = Const_c.update_filter_search(df, Constants)
        boots_done = df["bootstrap_iter"].dropna().unique()
        print("Bootstraps done:", boots_done)
        np.testing.assert_equal(len(boots_done), int(max(boots_done)) + 1)
        return int(max(boots_done)) + 1

    return 0


def _load_existing_history(path_logs, Constants) -> pd.DataFrame:
    df = pd.read_csv(path_logs)
    return Const_c.update_filter_search(df, Constants)



def _should_skip_run(exp, exp_name, ds_name, run, string_id_base) -> bool:
    if Const_c.all_boots_iter_done(
        exp, exp_name, ds_name, Constants, boots_iter=run + 1, string_id_base=string_id_base
    ):
        print(f"Skipping run {run} – already done")
        return True
    return False



def _set_random_seeds(run: int):
    torch.manual_seed(run)
    random.seed(run)
    np.random.seed(run)


def _load_data(ds_name, min_occurence, samples_per_class, run, PARAMS) -> dict:
    pretrained_sources = False

    if Constants["ALL_DATASETS"] or Constants["NoSrcDataset"]:
        pretrained_sources = bool(Constants["NoSrcDataset"])
        data = load_supervised_data(
            ds_name=ds_name, min_occurence=min_occurence,
            all_datasets=Constants["ALL_DATASETS"],
            pretrained_sources=pretrained_sources, boots_iter=run,
        )
        _, _, XTest, YTest, XSupp, YSupp, _, _ = split_train_test(PARAMS=PARAMS, X=data["X_tgt"], Y=data["Y_tgt"])
        XTrain, YTrain = data["X_src"], data["Y_src"]
        X_val,  Y_val  = data["X_val"], data["Y_val"]
        w2i            = data["w2i"]
        assert len(np.unique(YTest)) > 0, "Target test set has no classes!"

    else:
        data = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence, boots_iter=run)

        if check_SOTA_case(ds_name, Constants["DATASETS_NAMES"]):
            XTrain, YTrain = data["X_src"], data["Y_src"]
            _, _, XTest, YTest, XSupp, YSupp, _, _ = split_train_test(PARAMS=PARAMS, X=data["X_tgt"], Y=data["Y_tgt"])
        else:
            XTrain, YTrain, XTest, YTest, XSupp, YSupp, _, _ = split_train_test(PARAMS=PARAMS, X=data["X_tgt"], Y=data["Y_tgt"])

        X_val, Y_val = data["X_val"], data["Y_val"]
        w2i          = data["w2i"]
        print(f"\tNumber of classes: {len(w2i)}")

    return dict(
        XTrain=XTrain, YTrain=YTrain, XTest=XTest, YTest=YTest,
        XSupp=XSupp, YSupp=YSupp, X_val=X_val, Y_val=Y_val,
        w2i=w2i, pretrained_sources=pretrained_sources,
    )


def _apply_clustering(XTrain, YTrain, PARAMS, path_weights_pretrained):
    try:
        XTrain, YTrain = convert_clustering_experiment(XTrain, YTrain, Constants=Constants, PARAMS=PARAMS)
    except Exception:
        if os.path.exists(path_weights_pretrained):
            os.remove(path_weights_pretrained)
    return XTrain, YTrain


def _prepare_output_dir(model_type):
    output_dir = config.output_dir / "supervised" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _assert_input_sizes(XTrain, XTest):
    tgt_ds = list(Constants["TGT_DATASETS"].keys())[0]
    np.testing.assert_equal((XTest.shape[-2], XTest.shape[-1]), Constants["INPUT_SIZE"][tgt_ds])
    if len(XTrain) and list(Constants["DATASETS_NAMES"])[0] != "NO_SRC_DATASET":
        src_ds = list(Constants["DATASETS_NAMES"])[0]
        np.testing.assert_equal((XTrain.shape[-2], XTrain.shape[-1]), Constants["INPUT_SIZE"][src_ds])


def _print_split_summary(YTrain, YTest, Y_val):
    print(f"\tTotal (train+test, val): {len(YTrain)+len(YTest)}, {len(Y_val)}")
    print(f"\tTest: {len(YTest)}  |  Val: {len(Y_val)}")
    print(
        f"\tSource classes [Train+Val]: "
        f"{len(np.unique(YTrain))+len(np.unique(Y_val))} "
        f"({len(np.unique(YTrain))}, {len(np.unique(Y_val))})"
    )
    print(f"\tTarget classes: {len(np.unique(YTest))}")


def _validate_nway_config(YTrain, YTest):
    if len(np.unique(YTest)) > len(np.unique(YTrain)) and not Constants["LIMIT_N_WAY_TRAIN"]:
        sys.exit("ERROR: fewer training classes than test classes")
    if Constants["LIMIT_N_WAY_TEST"] or Constants["LIMIT_N_WAY_TRAIN"]:
        if Constants["LIMIT_N_WAY_TEST"] > Constants["LIMIT_N_WAY_TRAIN"]:
            sys.exit("ERROR: N-WAY TEST > N-WAY TRAIN")
        print(f"\tN-WAY Train: {Constants['LIMIT_N_WAY_TRAIN']}  |  N-WAY Test: {Constants['LIMIT_N_WAY_TEST']}")
    if Constants["LIMIT_N_WAY_TEST"] < 2 or Constants["LIMIT_N_WAY_TRAIN"] < 2:
        sys.exit("ERROR: N-WAY < 2 makes no sense")



def _build_run_history_row(run, accuracy, ds_name, src_datasets, src_input_size, samples_per_class, model_type, metrics=None) -> dict:
    metrics = metrics or {}
    ft_eval_acc = float(metrics.get("ft_eval_acc", metrics.get("best_accuracy", accuracy / 100.0)))
    ft_acc = float(metrics.get("ft_acc", ft_eval_acc))
    before_ft_eval_acc = float(metrics.get("before_ft_eval_acc", 0.0))
    before_ft_eval_loss = metrics.get("before_ft_eval_loss", "")

    return {
        "bootstrap_iter": run,
        "accuracy": accuracy,
        "ft_acc": ft_acc,
        "ft_eval_acc": ft_eval_acc,
        "before_ft_eval_acc": before_ft_eval_acc,
        "before_ft_eval_loss": before_ft_eval_loss,
        "tgt_dataset": ds_name,
        "src_datasets": "__".join(src_datasets) if isinstance(src_datasets, (list, tuple)) else str(src_datasets),
        "samples_per_class": samples_per_class,
        "model_type": model_type,
        "input_size": src_input_size,
        "num_total_episodes": Constants["EPISODES"],
        "batch_size": Constants["BATCH_SIZE"],
        "num_bootstrap_iters": Constants["BOOTSTRAP_ITERS"],
        "fixed_support": Constants["USE_ORIGINAL_FIXED_SUPP_SET"],
        "n_way_train": Constants["LIMIT_N_WAY_TRAIN"],
        "n_way_test": Constants["LIMIT_N_WAY_TEST"],
        "lr": Constants["lr"],
    }


def _save_history(history: pd.DataFrame, path_logs: str) -> None:
    os.makedirs(os.path.dirname(path_logs), exist_ok=True)
    history.to_csv(path_logs, index=False)


def _print_and_save_summary(results, samples_per_class, num_runs, epochs, batch_size, output_dir):
    print("----------------------------------------------------")
    print("BOOTSTRAP SUMMARY:")
    print(f"\tSamples/class: {samples_per_class}  |  Runs: {num_runs}")
    print(f"\tMean accuracy: {np.mean(results):.2f}  ±  {np.std(results):.2f}")
    write_plot_results(
        filepath=output_dir / "results.txt",
        from_weights="-",
        epochs=epochs,
        batch_size=batch_size,
        results=results,
        samples_per_class=samples_per_class,
    )


# ---------------------------------------------------------------------------
# Private helpers – model building & training phases
# ---------------------------------------------------------------------------

def _resolve_nway(YTrain, YTest):
    if not Constants["LIMIT_N_WAY_TRAIN"] or not Constants["LIMIT_N_WAY_TEST"]:
        n = min(len(np.unique(YTrain)), len(np.unique(YTest)))
        return n, n
    return Constants["LIMIT_N_WAY_TRAIN"], Constants["LIMIT_N_WAY_TEST"]


def _build_model_and_optimizer(model_type, batch_size, encoder_features, samples_per_class,
                                num_c_tr, num_c_ts, best_acc, device):
    tgt_ds     = list(Constants["TGT_DATASETS"].keys())[0]
    image_size = Constants["INPUT_SIZE"][tgt_ds][0]

    if model_type == "MatchingNetwork":
        model = MatchingNetwork(
            batch_size=batch_size,
            encoder_features=encoder_features,
            keep_prob=torch.FloatTensor(1),
            num_channels=3,
            fce=False,
            num_classes_per_set_train=num_c_tr,
            num_classes_per_set_test=num_c_ts,
            num_samples_per_class=samples_per_class,
            nClasses=0,
            image_size=image_size,
            best_accuracy=best_acc,
        )
    elif model_type == "PrototypicalNetwork":
        model = PrototypicalNetwork(x_dim=3, hid_dim=64, z_dim=64)
    elif model_type == "RelationNetwork":
        model = RelationNetwork(
            feature_dimension=64, ds_name=tgt_ds,
            spc=samples_per_class, input_size=image_size,
        )
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    model = model.to(device)

    if model_type == "RelationNetwork":
        model.feature_encoder_optim      = torch.optim.Adam(model.feature_encoder.parameters(), lr=Constants["lr"])
        model.feature_encoder_scheduler  = torch.optim.lr_scheduler.StepLR(model.feature_encoder_optim,  step_size=Constants["step_size"], gamma=0.5)
        model.relation_network_optim     = torch.optim.Adam(model.relation_network.parameters(), lr=Constants["lr"])
        model.relation_network_scheduler = torch.optim.lr_scheduler.StepLR(model.relation_network_optim, step_size=Constants["step_size"], gamma=0.5)
        return model, None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=Constants["lr"], weight_decay=Constants["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Constants["step_size"], gamma=0.5)
    return model, optimizer, scheduler


def _pretrain(model, optimizer, scheduler, XTrain, YTrain, XTest, YTest, X_val, Y_val,
              batch_size, epochs, num_c_tr, samples_per_class,
              model_type, path_weights_base, path_weights_pretrained, metrics, device):
    need_train = not Constants["ReusePretrained"] or not os.path.exists(path_weights_pretrained)
    if not need_train:
        return
    if Constants["ReusePretrained"]:
        print("Weights not found – training from scratch:", path_weights_pretrained)

    os.makedirs(os.path.dirname(path_weights_base), exist_ok=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_best_acc, _, optimizer, _ = FewShotTrain.train_few_shot_net(
            batch_size=batch_size, encoder=model,
            X=torch.from_numpy(XTrain), Y=torch.from_numpy(YTrain),
            device=device, model_type=model_type,
            samples_per_class=samples_per_class, classes_per_set=num_c_tr,
            optimizer=optimizer, checkpoint_path=path_weights_base,
            X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest),
            metrics=metrics, scheduler=scheduler, X_val=X_val, Y_val=Y_val,
        )
        if train_best_acc > metrics["best_accuracy"]:
            metrics["best_accuracy"] = train_best_acc
        if metrics.get("perfect_accuracy_early_stop"):
            break


def _finetune(model, XSupp, YSupp, XTest, YTest, batch_size, num_c_ts, samples_per_class,
              model_type, path_weights_base, path_weights_pretrained,
              metrics, reuse_logs, boots_iter, device):
    print("\nFineTune!!")
    best_before = float(metrics["best_accuracy"])
    metrics["before_ft_eval_acc"] = best_before
    metrics["before_ft_eval_loss"] = ""
    metrics["best_accuracy"] = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=Constants["lrFineTuning"], weight_decay=1e-6)

    if not Constants["NoSrcDataset"]:
        checkpoint = torch.load(path_weights_pretrained, map_location=device)
        if model_type == "RelationNetwork":
            model.feature_encoder.load_state_dict(checkpoint["feature_encoder"])
            model.relation_network.load_state_dict(checkpoint["relation_network"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    for epoch in range(Constants["epochsFineTuning"]):
        FewShotTrain.finetune_few_shot_net(
            batch_size=batch_size, encoder=model,
            X=torch.from_numpy(XSupp), Y=torch.from_numpy(YSupp),
            device=device, model_type=model_type,
            samples_per_class=samples_per_class, classes_per_set=num_c_ts,
            optimizer=optimizer, checkpoint_path=path_weights_base,
            X_eval=torch.from_numpy(XTest), Y_eval=torch.from_numpy(YTest),
            metrics=metrics, ft_epoch_num=epoch,
            reuse_logs=reuse_logs, boots_iter=boots_iter,
        )
        if metrics.get("perfect_accuracy_early_stop"):
            break

    metrics["ft_acc"] = float(metrics.get("ft_acc", metrics["best_accuracy"]))
    metrics["ft_eval_acc"] = float(metrics.get("ft_eval_acc", metrics["best_accuracy"]))
    print(f"BEST before FT: {best_before:.4f}  →  after FT: {metrics['best_accuracy']:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fire.Fire(run_bootstrap)
