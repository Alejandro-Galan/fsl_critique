## Call this script as "python3 scripts/complementary_comparison_methods.py 1 exp1"

import numpy as np
import sys, json

# 'PrototypicalNetwork--4000_Episodes--40x40_INPUT_SIZE/False_FIXED_SUPP_SET--Greek--5_NWAY_TRAIN--16_Batch_Size--1_KSamples_per_Class--0.001_lr--0.0_ValSrc----GROUP_EXP_source_permutation--NWAY-test_5--tgt_dataset_b-59-850']

import os, gc
import torch, random
import numpy as np
import pandas as pd
from sklearn import metrics


sys.path.append("./")

from models.MatchingNetwork import MatchingNetwork
from models.PrototypicalNetwork import PrototypicalNetwork

from torch.autograd import Variable

from datasets.loader import load_one_supervised_data
from models.FewShotModel import FewShotTrain
from network.loss import prototypical_loss, proto_loss_batch

import torch.nn.functional as F
from utils.constants import Const_c


### TODO embeddings and result-based comparison method

def load_empty_model(PARAMS, device):
    if PARAMS['MODEL_TYPE'] == "MatchingNetwork":
        model = MatchingNetwork(batch_size=PARAMS['BATCH_SIZE'], encoder_features=1600,
                            keep_prob=torch.FloatTensor(1), num_channels=3,
                            fce=False,
                            num_classes_per_set_train=PARAMS["LIMIT_N_WAY_TRAIN"], 
                            num_classes_per_set_test=PARAMS["LIMIT_N_WAY_TEST"],
                            num_samples_per_class=PARAMS['SAMPLES_PER_CLASS'],
                            nClasses=0, image_size = int(PARAMS['src_input_size'][0]), best_accuracy=0.0)
    elif PARAMS['MODEL_TYPE'] == "PrototypicalNetwork":
        model = PrototypicalNetwork(x_dim=3, hid_dim=64, z_dim=64)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['lr'], weight_decay=1e-6)
    # Reduce lr per half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    return model, scheduler


## Init model:
def load_models(path_to_model, PARAMS, device, fineTuned):

    if not fineTuned:
        # Load the base model to compare
        path_weights_base = path_to_model + "_trained_model.pt"
        print("Loading model:", path_weights_base)

    else:

        path_weights_base = Const_c.get_id_extensions(PARAMS, path_to_model)

        print("Loading finetuned model:", path_weights_base)
        

    model, scheduler = load_empty_model(PARAMS, device)
    checkpoint = torch.load(path_weights_base, map_location=device)
    # (Initialize only the encoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model
    
    # ft_models = []
    # other_ft_models = glob.glob(path_to_model + '*')
    # for ft_m in other_ft_models:
    #     if ft_m.endswith("_trained_model.pt"):
    #         continue

    #     print("Loading finetuned model:", ft_m)

    #     model_fine_tuned, scheduler_fine_tuned = load_empty_model(PARAMS, device)
    #     checkpoint = torch.load(ft_m, map_location=device)
    #     # (Initialize only the encoder)
    #     model_fine_tuned.load_state_dict(checkpoint['model_state_dict'])
    #     model_fine_tuned.eval()
    #     nway_test = ft_m.split("NWAY-test_")[1].split("--")[0]
    #     tgt_ds = ft_m.split("tgt_dataset_")[1].split("--")[0]
    #     new_model = {"model": model_fine_tuned, "LIMIT_N_WAY_TEST": nway_test, 
    #                  "TGT_DATASETS": tgt_ds}
    #     ft_models.append(new_model)
    # return model_trained, ft_models



def eval_image(model, x_support_set, y_support_set, x_target, y_target, PARAMS):

    x_support_set = Variable(torch.from_numpy(x_support_set)).float()
    y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
    x_target = Variable(torch.from_numpy(x_target)).float()
    y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

    # convert to one hot encoding
    y_support_set = torch.unsqueeze(y_support_set, 2)
    sequence_length = y_support_set.size()[1]
    y_support_set_one_hot = torch.FloatTensor(PARAMS['BATCH_SIZE'], sequence_length,
                                                PARAMS['LIMIT_N_WAY_TEST']).zero_()


    y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
    y_support_set_one_hot = Variable(y_support_set_one_hot)


    if PARAMS['MODEL_TYPE']   == "MatchingNetwork":
        acc, c_loss, outs, outs_embeds = model(support_set_images=x_support_set.cuda(), support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda(), get_output_embeddings=True)

        # Use the whole y_data as labels 
        labels_gt = torch.cat( (y_support_set.squeeze(2).T ,y_target.unsqueeze(1).T), dim=0)
        labels_gt = labels_gt.view(-1)
        embeddings = outs_embeds.view(-1, outs_embeds.size(2))

        gt_query = y_target
        predictions = outs

    elif PARAMS['MODEL_TYPE'] == "PrototypicalNetwork":
        all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, model)

        ### Extracted from prototypical loss
        all_dists = []
        for b in range(PARAMS['BATCH_SIZE']):
            all_dists.append(proto_loss_batch(all_outputs[b], inputs_y[b], PARAMS['SAMPLES_PER_CLASS']))
        dists = torch.stack(all_dists)
        n_classes = dists.shape[1]
        log_p_y = F.log_softmax(-dists, dim=2).view(PARAMS['BATCH_SIZE'], n_classes, 1, -1)
        predictions = log_p_y.max(3)[1]
        ###

        acc, c_loss = prototypical_loss(all_outputs, target=inputs_y, n_support=PARAMS['SAMPLES_PER_CLASS'], samples_per_class=PARAMS['SAMPLES_PER_CLASS'], batch_size=PARAMS['BATCH_SIZE'])
        embeddings = all_outputs #.flatten(start_dim=1)
        gt = inputs_y.flatten(start_dim=1)

        # mix "batch" and "class" dimensions
        embeddings = embeddings.view(-1, embeddings.size(2))
        labels_gt = gt.view(-1)
        gt_query = gt[:,:n_classes].reshape(-1)
        predictions = predictions.view(-1)

    return embeddings.detach().cpu(), labels_gt.detach().cpu(), acc.item(), predictions.detach().cpu(), gt_query.detach().cpu()


ALL_INPUT_SIZES = {
    "b-59-850"                   : (40,40),
    "TKH"                        : (40,40),
    "Egyptian"                   : (40,40),
    "Greek"                      : (40,40),
    "BreaKHis_formatted"         : (84,84),
    
    "omniglot"                   : (28,28),
    "omniglot_SOTA_testSet"      : (28,28),
    "omniglot_SOTA_trainvalSet"  : (28,28),

    "miniImageNet"               : (84,84),
    "miniImageNet_SOTA_testSet"  : (84,84),
    "miniImageNet_SOTA_trainSet" : (84,84),
}


#### TODO POR AHORA SOLO LOS QUE ME PASÓ JAVI
def compare_embeddings(embeddings, labels, predictions, new_line, PARAMS, labels_q):

    ###############################################################
    #### EMBEDDINGS
    ###############################################################
    calinski_score = metrics.calinski_harabasz_score(embeddings, labels)
    new_line['calinski_score'] = calinski_score


    silhouette_score = metrics.silhouette_score(embeddings, labels)
    new_line['silhouette_score'] = silhouette_score
    ###############################################################

    ###############################################################
    #### PREDICTION LABELS
    ###############################################################
    homogeneity_score = metrics.homogeneity_score(labels_q, predictions)
    new_line['homogeneity_score'] = homogeneity_score


    completeness_score = metrics.completeness_score(labels_q, predictions)
    new_line['completeness_score'] = completeness_score
    ###############################################################



def get_path_base(PARAMS):

    PARAMS['src_input_size'] = str(ALL_INPUT_SIZES[list(PARAMS["TGT_DATASETS"].keys())[0]][0]) + "x" + str(ALL_INPUT_SIZES[list(PARAMS["TGT_DATASETS"].keys())[0]][1]) 

    # path_to_model = "WEIGHTS/" + PARAMS['MODEL_TYPE'] + "--" + str(PARAMS["EPISODES"]) + "_Episodes--" \
    #                             + PARAMS['src_input_size'] + "_INPUT_SIZE/" \
    #                             + str(PARAMS["USE_ORIGINAL_FIXED_SUPP_SET"]) + "_FIXED_SUPP_SET--" \
    #                             + PARAMS['DATASETS_NAMES'] + "--" \
    #                             + str(PARAMS["LIMIT_N_WAY_TRAIN"]) + "_NWAY_TRAIN--" \
    #                             + str(PARAMS["BATCH_SIZE"]) + "_Batch_Size--" \
    #                             + str(PARAMS['SAMPLES_PER_CLASS']) + "_KSamples_per_Class--" \
    #                             + str(PARAMS["lr"]) + "_lr--" \
    #                             + str(PARAMS["VALIDATION_PERC"]) + "_ValSrc--" 

    string_id_base, string_id_ft = Const_c.get_experiment_id(PARAMS, boots_iter=0)
    path_to_model = "WEIGHTS/" + string_id_base 
    return path_to_model

def add_line_to_metrics(metrics_, new_line):
    for key in new_line.keys():
        if key not in metrics_:
            metrics_[key] = []
        metrics_[key].append(new_line[key])

def main_loop(PARAMS, fineTuned, metrics_):
    
    # Seed
    torch.manual_seed(PARAMS['seed_n'])
    random.seed(PARAMS['seed_n'])
    np.random.seed(PARAMS['seed_n'])
    
    ## First load the model
    path_to_model = get_path_base(PARAMS)

    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")


    ## Load the data of the tgt dataset
    X, Y, w2i = load_one_supervised_data(ds_name=list(PARAMS['TGT_DATASETS'].keys())[0], min_occurence=50)
    X, Y = np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)

    # index_batch = indexes[ind:end_ind]
    index_batch = np.arange(0, 16)
    x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(torch.from_numpy(X), torch.from_numpy(Y), index_batch, model_type=PARAMS['MODEL_TYPE'], metrics={}, classes_per_set=PARAMS['LIMIT_N_WAY_TEST'], 
                                                                                set="eval", samples_per_class=PARAMS['SAMPLES_PER_CLASS'], only_nk=False)
    if list(PARAMS['TGT_DATASETS'].keys())[0] == "BreaKHis_formatted":
        PARAMS['LIMIT_N_WAY_TRAIN'] = 3
        PARAMS['LIMIT_N_WAY_TEST'] = 3
    if list(PARAMS['DATASETS_NAMES'].keys())[0] == "BreaKHis_formatted":
        PARAMS['LIMIT_N_WAY_TRAIN'] = 3
        PARAMS['LIMIT_N_WAY_TEST'] = 3

    model = load_models(path_to_model, PARAMS, device=device, fineTuned=fineTuned)

    # Eval base_model
    print("Comparing model of", list(PARAMS["DATASETS_NAMES"].keys())[0], "with dataset ", list(PARAMS['TGT_DATASETS'].keys())[0])
    embeddings, labels_gt, acc, predictions, labels_q = eval_image(model, x_support_set, y_support_set, x_target, y_target, PARAMS)

    filtered_keys = ['DATASETS_NAMES', 'LIMIT_N_WAY_TRAIN', 'LIMIT_N_WAY_TEST', 'MODEL_TYPE', 'BATCH_SIZE', 'SAMPLES_PER_CLASS', 'GROUP_EXPERIMENT', 'Experiment', 'seed_n']
    new_line = {key: PARAMS[key] for key in filtered_keys if key in PARAMS}
    new_line['TGT_DATASETS'] = list(PARAMS['TGT_DATASETS'].keys())[0]
    new_line['DATASETS_NAMES'] = list(PARAMS['DATASETS_NAMES'].keys())[0]
    new_line['Fine_Tuned'] = fineTuned
    new_line['acc'] = acc
    compare_embeddings(embeddings, labels=labels_gt, new_line=new_line, predictions=predictions, PARAMS=PARAMS, labels_q=labels_q)
    add_line_to_metrics(metrics_, new_line)


if __name__ == '__main__':

    with open("scripts/complementary_comp/PARAMS_NOT_PARALLELIZABLE.json", "r") as f:
        PARAMS = json.load(f)

    metrics_file, metrics_ft_file = "logs_csv/comparison_embeddings_finetuned.csv", "logs_csv/comparison_embeddings.csv"

    metrics_, metrics_ft = {}, {}

    if os.path.exists(metrics_file):
        metrics_ = pd.read_csv(metrics_file, index_col=0).to_dict(orient="list")
    if os.path.exists(metrics_ft_file):
        metrics_ft = pd.read_csv(metrics_ft_file, index_col=0).to_dict(orient="list")

    
    # try:
    # Table with embeddings from src only
    main_loop(PARAMS, fineTuned=False, metrics_=metrics_)
    # Table with embeddings from src fineTuned to tgt
    main_loop(PARAMS, fineTuned=True, metrics_=metrics_ft)

    # print("\nUPDATE ON METRICS!!\n", metrics_)
    print("\n\nUPDATE ON METRICS!!\n\n")
    df = pd.DataFrame(metrics_)
    df['DATASETS_NAMES'] = df['DATASETS_NAMES'].apply(lambda x: x.split("'")[1] if x.startswith("{") else x)
    df.to_csv("logs_csv/comparison_embeddings.csv")
    
    df_ft = pd.DataFrame(metrics_ft)
    df_ft['DATASETS_NAMES'] = df_ft['DATASETS_NAMES'].apply(lambda x: x.split("'")[1] if x.startswith("{") else x)
    df_ft.to_csv("logs_csv/comparison_embeddings_finetuned.csv")
    # except Exception as e:
    #     # Probably not executed yet
    #     print(e)

