## Call this script as "python3 scripts/complementary_comparison_methods.py 1 exp1"

import sys
if len(sys.argv) < 2:
    sys.argv[1:] = ["1", "exp1"]

import fire, wandb, itertools, os, gc, glob
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






### TODO embeddings and result-based comparison method

def load_empty_model(PARAMS, device):
    if PARAMS['model_type'] == "MatchingNetwork":
        model = MatchingNetwork(batch_size=PARAMS['BATCH_SIZE'], encoder_features=1600,
                            keep_prob=torch.FloatTensor(1), num_channels=3,
                            fce=False,
                            num_classes_per_set_train=PARAMS["LIMIT_N_WAY_TRAIN"], 
                            num_classes_per_set_test=PARAMS["LIMIT_N_WAY_TEST"],
                            num_samples_per_class=PARAMS['samples_per_class'],
                            nClasses=0, image_size = int(PARAMS['src_input_size'][0]), best_accuracy=0.0)
    elif PARAMS['model_type'] == "PrototypicalNetwork":
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

        path_weights_base = path_to_model + "--GROUP_EXP_" + PARAMS["GROUP_EXPERIMENT"] + \
                                      "--NWAY-test_" + str(PARAMS['LIMIT_N_WAY_TEST']) + \
                                      "--tgt_dataset_" + PARAMS["TGT_DATASETS"] + "--_trained_finetuned_model.pt"

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


    if PARAMS['model_type']   == "MatchingNetwork":
        acc, c_loss, outs, outs_embeds = model(support_set_images=x_support_set.cuda(), support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda(), get_output_embeddings=True)

        # Use the whole y_data as labels 
        labels_gt = torch.cat( (y_support_set.squeeze(2).T ,y_target.unsqueeze(1).T), dim=0)
        labels_gt = labels_gt.view(-1)
        embeddings = outs_embeds.view(-1, outs_embeds.size(2))

        gt_query = y_target
        predictions = outs

    elif PARAMS['model_type'] == "PrototypicalNetwork":
        all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, model)

        ### Extracted from prototypical loss
        all_dists = []
        for b in range(PARAMS['BATCH_SIZE']):
            all_dists.append(proto_loss_batch(all_outputs[b], inputs_y[b], PARAMS['samples_per_class']))
        dists = torch.stack(all_dists)
        n_classes = dists.shape[1]
        log_p_y = F.log_softmax(-dists, dim=2).view(PARAMS['BATCH_SIZE'], n_classes, 1, -1)
        predictions = log_p_y.max(3)[1]
        ###

        acc, c_loss = prototypical_loss(all_outputs, target=inputs_y, n_support=PARAMS['samples_per_class'], samples_per_class=PARAMS['samples_per_class'], batch_size=PARAMS['BATCH_SIZE'])
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
    PARAMS['src_input_size'] = str(ALL_INPUT_SIZES[PARAMS["TGT_DATASETS"]][0]) + "x" + str(ALL_INPUT_SIZES[PARAMS["TGT_DATASETS"]][1]) 

    path_to_model = "WEIGHTS/" + PARAMS['model_type'] + "--" + str(PARAMS["EPISODES"]) + "_Episodes--" \
                                + PARAMS['src_input_size'] + "_INPUT_SIZE/" \
                                + str(PARAMS["USE_ORIGINAL_FIXED_SUPP_SET"]) + "_FIXED_SUPP_SET--" \
                                + PARAMS['src_datasets'] + "--" \
                                + str(PARAMS["LIMIT_N_WAY_TRAIN"]) + "_NWAY_TRAIN--" \
                                + str(PARAMS["BATCH_SIZE"]) + "_Batch_Size--" \
                                + str(PARAMS['samples_per_class']) + "_KSamples_per_Class--" \
                                + str(PARAMS["lr"]) + "_lr--" \
                                + str(PARAMS["VALIDATION_PERC"]) + "_ValSrc--" 
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
    X, Y, w2i = load_one_supervised_data(ds_name=PARAMS['TGT_DATASETS'], min_occurence=50)
    X, Y = np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.int64)

    # index_batch = indexes[ind:end_ind]
    index_batch = np.arange(0, 16)
    x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(torch.from_numpy(X), torch.from_numpy(Y), index_batch, model_type=PARAMS['model_type'], metrics={}, classes_per_set=PARAMS['LIMIT_N_WAY_TEST'], 
                                                                                set="eval", samples_per_class=PARAMS['samples_per_class'], only_nk=False)
    
    
    model = load_models(path_to_model, PARAMS, device=device, fineTuned=fineTuned)

    # Eval base_model
    print("Comparing model of", PARAMS["src_datasets"], "with dataset ", PARAMS['TGT_DATASETS'])
    embeddings, labels_gt, acc, predictions, labels_q = eval_image(model, x_support_set, y_support_set, x_target, y_target, PARAMS)

    new_line = PARAMS.copy()
    new_line['acc'] = acc
    compare_embeddings(embeddings, labels=labels_gt, new_line=new_line, predictions=predictions, PARAMS=PARAMS, labels_q=labels_q)
    add_line_to_metrics(metrics_, new_line)

    



if __name__ == '__main__':
    ## TODO change params depending on dataset. For ex capitan is size 84x84 only for mini and breakhis
    PARAMS = {
        'EPISODES': 4000,
        'BATCH_SIZE': 16, 
        'USE_ORIGINAL_FIXED_SUPP_SET': False, 
        'lr': 0.001,
        'VALIDATION_PERC': 0.0,
        ## FT params
        'GROUP_EXPERIMENT': "source_permutation",
        }
    
    ## TODO change params depending on dataset. For ex capitan is size 84x84 only for mini and breakhis

    metrics_, metrics_ft_ = {}, {}


    model_type_ = ["MatchingNetwork", "PrototypicalNetwork"]
    inmutable_datasets = ["b-59-850", "Greek", "TKH", "Egyptian", "BreaKHis_formatted"] 
    src_datasets_ = inmutable_datasets + ["omniglot_SOTA_trainvalSet", "miniImageNet_SOTA_trainSet"]
    TGT_DATASETS_ = inmutable_datasets + ["omniglot_SOTA_testSet", "miniImageNet_SOTA_testSet"]
    LIMIT_N_WAY_TRAIN_ = [5]
    LIMIT_N_WAY_TEST_  = [5]
    samples_per_class_ = [1,5,10]
    seeds_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for seed_n in seeds_:
        PARAMS['seed_n'] = seed_n
        for model_type in model_type_:
            for src_datasets in src_datasets_:
                for LIMIT_N_WAY_TRAIN in LIMIT_N_WAY_TRAIN_:
                    for samples_per_class in samples_per_class_:
                        for LIMIT_N_WAY_TEST in LIMIT_N_WAY_TEST_:
                            ### FT params
                            PARAMS['model_type'] = model_type
                            PARAMS['src_datasets'] = src_datasets
                            PARAMS['LIMIT_N_WAY_TRAIN'] = LIMIT_N_WAY_TRAIN
                            PARAMS['samples_per_class'] = samples_per_class
                            PARAMS['LIMIT_N_WAY_TEST'] = LIMIT_N_WAY_TEST

                            ## Loop over different datasets
                            for tgt_datasets in TGT_DATASETS_:
                                if tgt_datasets[:3] == src_datasets[:3]:
                                    continue
                                PARAMS['TGT_DATASETS'] = tgt_datasets
                                try:
                                    # Table with embeddings from src only
                                    main_loop(PARAMS, fineTuned=False, metrics_=metrics_)
                                    # Table with embeddings from src fineTuned to tgt
                                    main_loop(PARAMS, fineTuned=True, metrics_=metrics_ft_)

                                    print("\nUPDATE ON METRICS!!\n", metrics_)
                                    df = pd.DataFrame(metrics_)
                                    df.to_csv("logs_csv/comparison_embeddings.csv")
                                    
                                    df_ft = pd.DataFrame(metrics_ft_)
                                    df_ft.to_csv("logs_csv/comparison_embeddings_finetuned.csv")
                                except Exception as e:
                                    # Probably not executed yet
                                    print(e)
