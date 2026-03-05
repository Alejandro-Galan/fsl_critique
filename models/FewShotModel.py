import tqdm, torch, wandb, os, random, sys, importlib
import numpy as np
from torch.autograd import Variable

from utils import constants
importlib.reload(constants)
from utils.train_utils import split_train_test, debug_images_and_dists
from utils.constants import Const_c
from datasets.loader import apply_simclr_augmentation, get_params_for_loading_datasets
# Initialize reading the json constants file for each experiment
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants

from network.loss import prototypical_loss
from models.PrototypicalNetwork import PrototypicalNetwork

from datasets.loader import load_supervised_data

class FewShotTrain():

    def train_few_shot_net(batch_size, encoder, X, Y, device, classes_per_set, samples_per_class, X_eval, Y_eval, 
                           checkpoint_path, model_type="", optimizer = None, metrics=None, scheduler=None,
                           X_val=None, Y_val=None):


        encoder.train()
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)
        
        best_epoch = 0.0
        epochs_no_improve = 0
        
        new_path = checkpoint_path + "_trained_model.pt"
        os.makedirs(os.path.dirname(new_path), exist_ok=True )
        
        cps_test, cps_train = classes_per_set, classes_per_set
        # if Constants["LIMIT_N_WAY_TRAIN"]:
        #     cps_train = Constants["LIMIT_N_WAY_TRAIN"]
        # if Constants["LIMIT_N_WAY_TEST"]:
        #     cps_test = Constants["LIMIT_N_WAY_TEST"]

        # total_train_batches = (X.shape[0] + batch_size - 1) // batch_size
        # total_train_batches = X.shape[0] // batch_size
        total_train_batches = Constants["EPISODES"] // batch_size # As it is random, not fixed

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                # * number of samples
                # end_i = (i + 1) * batch_size # No need to cut, is not sequential
                # if end_i > X.shape[0]: 
                #     end_i = X.shape[0]
                # index_batch = indexes[i*batch_size:end_i]
                index_batch = np.random.choice(X.shape[0], batch_size, replace=False)
                x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, metrics=metrics, 
                                                                                                    classes_per_set=cps_train, samples_per_class=samples_per_class)
                np.testing.assert_equal(x_support_set.shape[1], classes_per_set * samples_per_class)

                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                            cps_train).zero_()

                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                if model_type == "MatchingNetwork":
                    acc, c_loss, _, _ = encoder(support_set_images=x_support_set.cuda(), 
                                    support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda())
                elif model_type == "PrototypicalNetwork":
                    all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                    acc, c_loss, _ = prototypical_loss(all_outputs, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)
                elif model_type == "RelationNetwork":
                    acc, c_loss, _, _ = encoder(x_support_set.cuda(), y_support_set.cuda(), x_target.cuda(), y_target.cuda(), 
                                    train=True, SAMPLE_NUM_PER_CLASS=samples_per_class, CLASS_NUM=classes_per_set)

                if not Constants["DEACTIVATE_WANDB"]:
                    wandb.log({"tr_acc": acc, "tr_loss": c_loss})
                    if not model_type == "RelationNetwork":
                        wandb.log({"tr lr": optimizer.param_groups[0]['lr']})

                # It is done inside the model
                if not model_type == "RelationNetwork":
                    # optimize process
                    optimizer.zero_grad()
                    c_loss.backward()
                    optimizer.step()
                    for _ in range(batch_size):
                        scheduler.step()

                    FewShotTrain.adjust_learning_rate(optimizer)

                total_c_loss += c_loss.item()
                total_accuracy += acc.item()
                iter_out = "tr_loss: {}, tr_accuracy: {}, mean_tr_acc: {}".format(total_c_loss/(i+1), acc.item(), total_accuracy/(i+1))
                if not model_type == "RelationNetwork":
                    iter_out += ", lr: {}".format(optimizer.param_groups[0]['lr'])
                pbar.set_description(iter_out)
                pbar.update(1)

                # limit = 20 if total_train_batches > 20 else total_train_batches
                limit_valTGT = total_train_batches // Constants["LIMIT_VALIDATION_SRC_TGT"]  # Twice # One per epoch
                limit_valSRC = total_train_batches // Constants["LIMIT_VALIDATION_SRC_SRC"]  
                    
                # Validate over tgt data, just for observation purposes
                if Constants["VALIDATION_SRC_SRC_DATA"] and (i + 1) % limit_valSRC == 0:
                    encoder.eval()

                    test_accs_std, test_loss, test_outputs, dists = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_val, Y=Y_val, X_train=X, Y_train=Y,
                                            device=device, model_type=model_type, supp_set=None, classes_per_set=cps_test, samples_per_class=samples_per_class, metrics=metrics, set="Val")
                    test_accs = test_accs_std[0] 
                    if not Constants["DEACTIVATE_WANDB"]:
                        wandb.log({"tr_eval_acc_src_data": test_accs, "tr_eval_loss_src_data": test_loss})

                    if test_accs > metrics['best_accuracy']:

                        metrics['best_accuracy'] = test_accs
                        best_epoch = i
                        # metrics['best_class_rep'] = classification_report(y_true=Y_eval.cpu().tolist(), y_pred=test_outputs.cpu(), output_dict=True)

                        FewShotTrain.store_encoder(encoder, optimizer=optimizer, new_path=new_path, model_type=model_type)

                        epochs_no_improve = 0

                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= metrics['PATIENCE']:
                            print("Early stopping triggered")
                            break
                encoder.train()

                if Constants["VALIDATION_SRC_TGT_DATA"] and (i + 1) % limit_valTGT == 0:
                                            
                    test_accs_std, test_loss, test_outputs, dists = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                                            device=device, model_type=model_type, supp_set=None, classes_per_set=cps_test, samples_per_class=samples_per_class, metrics=metrics)
                    test_accs = test_accs_std[0]
                    # print("TEST ACCS", test_accs)
                    if not Constants["DEACTIVATE_WANDB"]:
                        wandb.log({"tr_eval_acc": test_accs, "tr_eval_loss": test_loss})

                encoder.train()


        # Save only in last iteration
        # Otherwise the best is already saved
        if not Constants["VALIDATION_SRC_SRC_DATA"]:

            FewShotTrain.store_encoder(encoder, optimizer=optimizer, new_path=new_path, model_type=model_type)


        return metrics['best_accuracy'], best_epoch, optimizer, None

        # total_c_loss = total_c_loss / total_train_batches
        # total_accuracy = total_accuracy / total_train_batches
        # return total_c_loss, total_accuracy, optimizer


    def eval_few_shot_net(batch_size, encoder, X, Y, X_train, Y_train, device, classes_per_set, samples_per_class, metrics, model_type="Original", supp_set=None, finetune=False, set="Test", debug_images=False):
        All_acc, All_out, All_dists = [], [], []

        encoder.eval()
        with torch.no_grad():
            indexes = np.arange(X.shape[0])
            # np.random.shuffle(indexes) ### Must not shuffle, as the test set must be fixed to be comparable

            # offset_limit = (X.shape[0] + batch_size - 1)
            # offset_limit = X.shape[0] # No need to pick the extra
            # total_test_batches = offset_limit // batch_size
            total_test_batches = Constants["TEST_EPISODES"] // batch_size


            ####
            # Same for all images
            # labels = supp_set["labels"]
            # imgs   = supp_set["imgs"] 
            # one_hot_labels = np.zeros((len(labels), labels.max() + 1))
            # one_hot_labels[np.arange(len(labels)), labels] = 1.0
            ###################
            # imgs, labels = self.get_only_a_subsample_set(X_train, Y_train, batch_size=batch_size, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
            ###################

            with tqdm.tqdm(total=total_test_batches) as pbar:
                for ind in range(total_test_batches):
                # for ind in range(0, offset_limit, batch_size):
                    ind = ind % len(indexes) # In case the test size is lower than the test epoch size
                    end_ind = ind + batch_size

                    if end_ind <= len(indexes):
                        # end_ind = min(ind + batch_size, X.shape[0])

                        ### Iterating for each batch. Obtain that target and support from test set
                        # index_batch = np.random.choice(X.shape[0], batch_size, replace=False) 
                        index_batch = indexes[ind:end_ind] ## For test/eval set, to be comparable
                        x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, metrics=metrics, classes_per_set=classes_per_set, 
                                                                                                    set="eval", samples_per_class=samples_per_class, only_nk=True)
                        ###

                        x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                        y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                        x_target = Variable(torch.from_numpy(x_target)).float()
                        y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                        # convert to one hot encoding
                        y_support_set = torch.unsqueeze(y_support_set, 2)
                        sequence_length = y_support_set.size()[1]
                        y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                                    classes_per_set).zero_()


                        y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                        y_support_set_one_hot = Variable(y_support_set_one_hot)

                        # #### Prev eval
                        # one_hot_labels = np.zeros((len(y_support_set), y_support_set.max() + 1))
                        # one_hot_labels[np.arange(len(y_support_set)), y_support_set] = 1.0

                        # i = indexes[ind:end_ind]
                        # # x = X[i].to(device)
                        # # y = Y[i].to(device)

                        # ## It need to be duplicated batch_size times
                        # imgs_ext = extend_dimenstions_bs(x_support_set, len(i))
                        # one_hot_labels_ext = extend_dimenstions_bs(one_hot_labels, len(i))
                        ###############
                        output = None
                        if model_type == "MatchingNetwork":
                            acc, loss, _, _ = encoder(support_set_images=x_support_set.to(device), 
                                                    support_set_labels_one_hot=y_support_set_one_hot.to(device).float(), 
                                                    target_image=x_target.to(device), target_label=y_target.to(device))
                        elif model_type == "PrototypicalNetwork":
                            output, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                            acc, c_loss, dists = prototypical_loss(output, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)
                            if debug_images:
                                debug_images_and_dists(x_target, x_support_set, dists, debug_images=debug_images)
                        elif model_type == "RelationNetwork":
                            acc, c_loss, _, dists = encoder(x_support_set.cuda(), y_support_set.cuda(), x_target.cuda(), y_target.cuda(), train=False, SAMPLE_NUM_PER_CLASS=samples_per_class, CLASS_NUM=classes_per_set)
                            dists = torch.Tensor(np.array(dists))
                            if debug_images:
                                debug_images_and_dists(x_target, x_support_set, dists, debug_images=debug_images)
                        

                        # if exp == "exp3" or exp == "exp6" and model_type == "PrototypicalNetwork":
                        if exp == "exp3":
                            if model_type == "RelationNetwork":
                                if len(All_dists) == 0:
                                    All_dists = dists.cpu().numpy()
                                else:
                                    All_dists = np.concatenate( (All_dists, dists.cpu().numpy() ), axis=0)
                            if model_type == "PrototypicalNetwork":
                                if len(All_dists) == 0:
                                    All_dists = dists.cpu().numpy()
                                else:
                                    All_dists = np.concatenate( (All_dists, dists.cpu().numpy() ), axis=0)



                        All_acc.append(acc.cpu().item())

                        if not output is None:
                            if output.dim() == 0:
                                All_out.append(output.unsqueeze(0) )
                            else:                              
                                All_out.append(output )
                
                    if (ind / batch_size) % 5 == 0:
                        if finetune:
                            iter_out = "Finetune: " + set + " Accuracy: " + str(np.mean(All_acc) )
                        else:
                            iter_out = ">: " + set + " Accuracy: " + str(np.mean(All_acc) )
                        pbar.set_description(iter_out)
                        pbar.update(5)
                # Asegúrate de actualizar la barra de progreso al final
                pbar.update(total_test_batches - pbar.n)
        if not All_out:
            return [np.mean(All_acc), np.std(All_acc)], None, None, All_dists
        return [np.mean(All_acc), np.std(All_acc)], None, torch.cat(All_out), All_dists


    def finetune_few_shot_net(batch_size, encoder, X, Y, device, classes_per_set, samples_per_class, X_eval, Y_eval, checkpoint_path, boots_iter, 
                           model_type="", optimizer = None, metrics=None, ft_epoch_num=0, reuse_logs=True):

        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)

        
        best_epoch = 0.0
        epochs_no_improve = 0
        
        # total_train_batches = (X.shape[0] + batch_size - 1) // batch_size
        # total_train_batches = X.shape[0] // batch_size
        total_train_batches = Constants["FINE_TUNING_EPISODES"] // batch_size # As it is random, not fixed


        ### Eval the training before fine-tuning
        if ft_epoch_num == 0 and not Constants["DEACTIVATE_WANDB"]:
            encoder.eval()        
            test_accs_std, test_loss, test_outputs, dists = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                                    device=device, model_type=model_type, supp_set=None, classes_per_set=classes_per_set, samples_per_class=samples_per_class, metrics=metrics, finetune=False)
            test_accs = test_accs_std[0]
            wandb.log({"before_ft_eval_acc": test_accs, "before_ft_eval_loss": test_loss})

            # if (exp == "exp3" or exp == "exp6") and model_type == "PrototypicalNetwork":
            #     FewShotTrain.store_distances(dists, sufix_path="_before_ft_distances.npy")
        encoder.train()

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                # * number of samples
                # end_i = (i + 1) * batch_size # No need to cut, is not sequential
                # if end_i > X.shape[0]: 
                #     end_i = X.shape[0]
                # index_batch = indexes[i*batch_size:end_i]

                replace = False
                # It must correspond to one of those
                if Constants["USE_ORIGINAL_FIXED_SUPP_SET"]:
                    # Select only index where Y['supp'] == Y
                    subX = (Y == Y['supp']).nonzero(as_tuple=True)[0]
                    print("TODO check this is correct, the assignation of index equivalent")
                    if subX.shape[0] < batch_size:
                        replace = True
                    index_batch_index = np.random.choice(subX.shape[0], batch_size, replace=replace)
                    index_batch = subX[index_batch_index]
                    breakpoint()
                else:
                    if X.shape[0] < batch_size:
                        replace = True

                    index_batch = np.random.choice(X.shape[0], batch_size, replace=replace)
                x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, metrics=metrics, 
                                                                                                    classes_per_set=classes_per_set, samples_per_class=samples_per_class,
                                                                                                    only_nk=True, change_classes_alias=True)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = torch.unsqueeze(y_support_set, 2)
                sequence_length = y_support_set.size()[1]
                y_support_set_one_hot = torch.FloatTensor(batch_size, sequence_length,
                                                            classes_per_set).zero_()

                y_support_set_one_hot.scatter_(2, y_support_set.data, 1)
                y_support_set_one_hot = Variable(y_support_set_one_hot)

                if model_type == "MatchingNetwork":
                    acc, c_loss, _, _ = encoder(support_set_images=x_support_set.cuda(), support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda())
                elif model_type == "PrototypicalNetwork":
                    all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                    acc, c_loss, _ = prototypical_loss(all_outputs, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)

                elif model_type == "RelationNetwork":
                    acc, c_loss, _, _ = encoder(x_support_set.cuda(), y_support_set.cuda(), x_target.cuda(), y_target.cuda(), train=True, SAMPLE_NUM_PER_CLASS=samples_per_class, CLASS_NUM=classes_per_set)


                if not Constants["DEACTIVATE_WANDB"]:
                    wandb.log({"ft_acc": acc, "ft_loss": c_loss})


                # It is done inside the model
                if not model_type == "RelationNetwork":
                    # optimize process
                    optimizer.zero_grad()
                    c_loss.backward()
                    optimizer.step()

                    FewShotTrain.adjust_learning_rate(optimizer)

                total_c_loss += c_loss.item()
                total_accuracy += acc.item()
                iter_out = "ft_loss: {}, ft_accuracy: {}, lr: {}".format(total_c_loss/(i+1), acc.item(), optimizer.param_groups[0]['lr'])
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1

                # limit = 20 if total_train_batches > 20 else total_train_batches
                limit = total_train_batches  # One per epoch

                end_of_epoch = ( (i + 1) % limit == 0 )

                ## For each episode
                case_loss_curves_exp3 = ft_epoch_num == 2 and exp == "exp3" and (model_type == "PrototypicalNetwork" or model_type == "RelationNetwork")
                if not Constants["EXHAUSTIVE_LOSS_CURVES"]:
                    case_loss_curves_exp3 = case_loss_curves_exp3 and end_of_epoch
                
                
                ## DEBUG to print exhaustive loss curves, only
                if end_of_epoch or case_loss_curves_exp3:
                # if end_of_epoch:
                # if True:
                    encoder.eval()
                    # supp_set = {"imgs": X, "labels": Y}
                    # supp_set = {"imgs": self.XSupp, "labels": self.YSupp} 
                    
                    test_accs_std, test_loss, test_outputs, dists = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                                            device=device, model_type=model_type, supp_set=None, classes_per_set=classes_per_set, samples_per_class=samples_per_class, metrics=metrics, finetune=True)
                    test_accs = test_accs_std[0]
                    # Only this distances
                    # For all ft_epoch_nums
                    # if ft_epoch_num == 2 and not Constants["DEACTIVATE_WANDB"] and (exp == "exp3" or exp == "exp6") and model_type == "PrototypicalNetwork":
                    # if ft_epoch_num == 2 and not Constants["DEACTIVATE_WANDB"] and case_loss_curves_exp3:
                    if not Constants["DEACTIVATE_WANDB"] and case_loss_curves_exp3:
                        path_distances = Const_c.get_distances_path(Constants=Constants, sufix_path="_after_ft_distances", boots_iter=boots_iter)
                        # if not os.path.exists(os.path.dirname(path_distances)):
                        if not os.path.exists(path_distances):
                            os.makedirs(os.path.dirname(path_distances), exist_ok=True)
                            FewShotTrain.calculate_distances(encoder, dists, test_accs_std, list(Constants["TGT_DATASETS"].keys())[0], model_type, device, batch_size, classes_per_set, samples_per_class, boots_iter=boots_iter, end_of_epoch=end_of_epoch)


                    # print("TEST ACCS", test_accs)
                    if not Constants["DEACTIVATE_WANDB"]:
                        wandb.log({"ft_eval_acc": test_accs, "ft_eval_loss": test_loss})

                    if end_of_epoch:
                        if test_accs > metrics['best_accuracy']:
                            # print(
                            #     f"Test accuracy improved from {best_accuracy:.2f} to {test_accs:.2f}"
                            # )

                            metrics['best_accuracy'] = test_accs
                            best_epoch = i
                            # metrics['best_class_rep'] = classification_report(y_true=Y_eval.cpu().tolist(), y_pred=test_outputs.cpu(), output_dict=True)

                            # If not exists (For example if using no pretrained weights)
                            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                            try:
                                new_path = Const_c.get_id_extensions(Constants, prev_str=checkpoint_path)
                                FewShotTrain.store_encoder(encoder, optimizer=optimizer, new_path=new_path, model_type=model_type)
                            except:
                                # Assuming the name is too long, lets try with a dictionary
                                coded_path = Const_c.add_to_dictionary_of_files(Const_c.get_id_extensions(Constants, prev_str=checkpoint_path)) + ".pt"
                                FewShotTrain.store_encoder(encoder, optimizer=optimizer, new_path=coded_path, model_type=model_type)
                            epochs_no_improve = 0

                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= metrics['PATIENCE']:
                                print("Early stopping triggered")
                                break


                    encoder.train()

        return metrics['best_accuracy'], best_epoch, optimizer, None


    # Calculate distances from the encoder trained only on few-shot of the other src datasets
    def calculate_distances(encoder, dists_base, accs_base, ds_name, model_type, device, batch_size, classes_per_set, samples_per_class, boots_iter, end_of_epoch): 
        path_distances = Const_c.get_distances_path(Constants=Constants, sufix_path="_after_ft_distances", boots_iter=boots_iter)
        os.makedirs(os.path.dirname(path_distances), exist_ok=True)
        if end_of_epoch:
            np.testing.assert_equal(len(dists_base) > 0, True)
            FewShotTrain.store_distances(dists_base, accs= accs_base, full_path= path_distances)
        
        all_src_ds = Constants["ALL_TST_DATASETS"]
        for src_ds in all_src_ds:
            # try:
            if Constants["NUM_SRC_CLASSES_DATASETS"][src_ds] < Constants["LIMIT_N_WAY_TEST"]:
                continue

            # if src_ds == ds_name:
            #     # print("Skipping distances for the same dataset as target:", src_ds)
            #     continue
            # if ds_name.startswith("omniglot_SOTA_") and src_ds.startswith("omniglot_SOTA_"):
            #     continue
            # if ds_name.startswith("miniImageNet_SOTA_") and src_ds.startswith("miniImageNet_SOTA_"):
            #     continue

            ## Load only the training src dataset
            pretrained_sources = True if Constants["NoSrcDataset"] else pretrained_sources

            # If no validation src paramenter, Xval is empty
            ## Get just the test partition!!
            data_dict = load_supervised_data(ds_name=src_ds, min_occurence=50, all_datasets=False, pretrained_sources=pretrained_sources, boots_iter=boots_iter)

            PARAMS = get_params_for_loading_datasets(src_datasets="NO_SRC_DATASET", ds_name=src_ds, samples_per_class=samples_per_class, run=boots_iter, Constants=Constants)
            XTrain, YTrain, XTest, YTest, XSupp, YSupp, XVal, YVal = split_train_test(
                PARAMS=PARAMS,
                X=data_dict["X_tgt"], 
                Y=data_dict["Y_tgt"],
            )
            
            X_obj_ds, Y_obj_ds = XTest, YTest

            debug_images_prefix = False
            if Constants["DEBUG_IMAGES"] and end_of_epoch:
                debug_images_prefix = Constants["MODEL_TYPE"] + "/trained_" + ds_name + "_over_" + src_ds + "/"

            encoder.eval()
            test_accs_std, test_loss, test_outputs, dists_src = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=torch.from_numpy(X_obj_ds), Y=torch.from_numpy(Y_obj_ds), X_train=None, Y_train=None,
                                    device=device, model_type=model_type, supp_set=None, classes_per_set=classes_per_set, samples_per_class=samples_per_class, metrics={}, finetune=True, debug_images=debug_images_prefix)
            test_accs = test_accs_std[0]
            # Store also accs
            path_distances_src = Const_c.get_distances_path(Constants=Constants, sufix_path="ft_distances_over_" + src_ds + "as_tgt_ds", boots_iter=boots_iter)
            if end_of_epoch:
                FewShotTrain.store_distances(dists_src, accs= test_accs_std, full_path=path_distances_src )
            # except: 
            #     print("Error calculating distances for dataset:", src_ds)
            wandb.log({"test_acc- " + src_ds + "_trained_on- " + ds_name: test_accs, "test_loss- " + src_ds + "_trained_on- " + ds_name: test_loss })


    def store_distances(dists, accs, full_path):

        # Path not exists
        if not os.path.exists(full_path):    
            print("\n*********\n*********\n*********\n*********Saving distances to", full_path)    
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # Save the distances to a file
            np.save(full_path, dists)
            # Save accs and std to file
            np.save(full_path.replace(".npy", "_accs.npy"), accs)
            print(f"Distances saved to {full_path}")


        ## Prev version more complete comparing pret and ft
        # dists_path = Const_c.get_distances_path(Constants=Constants, sufix_path=sufix_path)
        # # Path not exists
        # if not os.path.exists(dists_path):        
        #     os.makedirs(os.path.dirname(dists_path), exist_ok=True)
        #     # Save the distances to a file
        #     np.save(dists_path, dists)
        #     print(f"Distances saved to {dists_path}")


    def store_encoder(encoder, optimizer, new_path, model_type):
        if model_type == "RelationNetwork":
            torch.save({
                'feature_encoder': encoder.feature_encoder.state_dict(),
                'relation_network': encoder.relation_network.state_dict(),
                'feature_encoder_optimizer_state_dict': encoder.feature_encoder_optim.state_dict(),
                'relation_network_optimizer_state_dict': encoder.relation_network_optim.state_dict(),
            }, new_path )

        else:
            ## If fine tune, evaluating with this is cheating (if test or tgt data, of course)
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, new_path )


    def adjust_learning_rate( optimizer):
        """Updates the learning rate given the learning rate decay.
        The routine has been implemented according to the original Lua SGD optimizer
        """
        for i, group in enumerate(optimizer.param_groups):
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = group['lr'] / (1 + group['step'] * group['weight_decay'])


    def include_exc(exc, index, only_nk, samples_per_class):
        ## Make sure the predicted class is in support set
        # Different sample
        if not only_nk:
            index_red = index[(index != exc).nonzero(as_tuple=True)[0]]
            assert len(index_red) < len(index)
            index = index_red
        # else only possible to use n*k samples to train, must repeat
        # if only_nk:
        #     index_ch = np.random.choice(index, samples_per_class, replace=True)
        # else:
        index_ch = np.random.choice(index, samples_per_class, replace=False)
        
        # if set == "eval":
        #     index_ch = index[np.arange(0, samples_per_class)]
        
        return list(index_ch)



    def get_support_set_index( Y, exc, classes_per_set, samples_per_class, set="train", only_nk=False ):
        all_index = []

        # Assumed to use all classes
        all_classes = np.unique(Y)
            
        np.testing.assert_equal(len(all_classes) >= classes_per_set, True)
        np.testing.assert_equal(exc < len(Y), True)

        # if set == "train":
        # np.random.shuffle(all_classes)

        subset_classes = [x for x in all_classes if x != Y[exc]]

        subset_classes = random.sample(subset_classes, classes_per_set - 1)
        subset_classes.append(Y[exc].item())

        if Constants["SHUFFLE_SUPP_SET"]:
            random.shuffle(subset_classes)
        else:
            subset_classes.sort()


        for c in subset_classes:
            index = (Y == c).nonzero(as_tuple=True)[0]
                
            if Y[exc].item() == c:
                index_ch = FewShotTrain.include_exc(exc, index, only_nk, samples_per_class)
            else:
                # # Not the tgt image
                # index = [x for x in index if x != exc]
                # if only_nk:
                #     index_ch = np.random.choice(index, samples_per_class, replace=True)
                # else:
                index_ch = np.random.choice(index, samples_per_class, replace=False)

                # if set == "eval":
                #     index_ch = index[np.arange(0, samples_per_class)]
                
            all_index = all_index + list(index_ch)
        if Constants["SHUFFLE_SUPP_SET"]:
            np.random.shuffle(all_index) 

        np.testing.assert_equal(len(all_index), classes_per_set*samples_per_class)

        return all_index

    # def get_only_a_subsample_set( X, Y, batch_size, classes_per_set, samples_per_class=1):
    #     index_batch = np.random.choice(X.shape[0], batch_size, replace=False)
    #     # Only choose support label here 
    #     imgs, labels, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    #     return imgs[0], labels[0], x_target[0], y_target[0]
        
    # Convert every episode classes to a transcription of size "num_classes_per_set"
    def codify_subset_classes( supp, exc, max_c, model_type):

        return supp, exc

        # new_vec = np.append(supp, exc)
        # classes = {}
        
        # for e in range(len(new_vec)):
        #     class_ = new_vec[e]
        #     if class_ not in classes:
        #         classes[class_] = len(classes)
        #     new_vec[e] = classes[class_]

        # try:
        #     assert len(classes) == max_c 
        # except:
        #     breakpoint()
        # supp_v, query_v = new_vec[:len(supp)], new_vec[len(supp):]
        
        # return supp_v, query_v

    # OnlyNK applies for the special FineTuning phase after pretraining
    def get_subsamples_sets( X, Y, index_batch, classes_per_set, metrics, model_type="", 
                            samples_per_class=1, set="train", only_nk=False, change_classes_alias=False):

        batch_size = len(index_batch) # The remainder could be lower

        ########
        ## Testing best approach. All_training data means all extracted for batch permuted
        condition = "supp_from_train_set" # "all_training_data" #  prefixed_support_set 
        if set == "eval" and Constants["USE_ORIGINAL_FIXED_SUPP_SET"]:
            condition = "prefixed_support_set"
        ########   


        # if len(X.shape) == 2:
        #     X = X.unsqueeze(0)

        if model_type == "PrototypicalNetwork":
            target_x = np.zeros((batch_size, classes_per_set, X.shape[1], X.shape[2], X.shape[3]), np.float32)
            target_y = np.zeros((batch_size, classes_per_set), np.int32)
        elif model_type == "MatchingNetwork":
            target_x = np.zeros((batch_size, X.shape[1], X.shape[2], X.shape[3]), np.float32)
            target_y = np.zeros((batch_size, 1), np.int32)
        elif model_type == "RelationNetwork":
            target_x = np.zeros((batch_size, classes_per_set, X.shape[1], X.shape[2], X.shape[3]), np.float32)
            target_y = np.zeros((batch_size, classes_per_set), np.int32)

        # if condition == "prefixed_support_set":

        #     ## Get sets
        #     total_data = classes_per_set * samples_per_class

        #     assert total_data == len(metrics['YSupp'])
        #     support_set_x = np.zeros((batch_size, total_data,
        #                                 X.shape[1], X.shape[2], X.shape[3]), np.float32)
        #     support_set_y = np.zeros((batch_size, total_data), np.int32)

        #     for b in range(batch_size):
        #         exc = index_batch[b]

        #         if model_type == "PrototypicalNetwork":
        #             exc, y_supp = get_multiple_querys(exc, Y, metrics['YSupp'], only_nk)
        #         elif model_type == "MatchingNetwork":
        #             exc, y_supp = get_one_query(exc, Y, metrics['YSupp'], only_nk)
        #         elif model_type == "RelationNetwork":
        #             exc, y_supp = get_multiple_querys(exc, Y, metrics['YSupp'], only_nk)

        #         # change Y labels to equal size of num_classes 
        #         if Y[exc].item() not in y_supp:
        #             breakpoint()
        #         supp_y, exc_y = FewShotTrain.codify_subset_classes(y_supp, Y[exc], classes_per_set, model_type)
        #         ############ DEBUG ##################
        #         # index_debug_fixed = np.arange(0,samples_per_class)
        #         # supp_y, exc_y = FewShotTrain.codify_subset_classes(Y[index_debug_fixed], Y[exc], classes_per_set)
        #         #####################################

        #         supp_x, supp_y = metrics['XSupp'], supp_y
        #         if Constants["SHUFFLE_SUPP_SET"]:
        #             p = np.random.permutation(len(supp_y))
        #             supp_x, supp_y = metrics['XSupp'][p], supp_y[p]

        #         support_set_x[b] = supp_x
        #         support_set_y[b] = supp_y

        #         target_x[b] = X[exc] 
        #         target_y[b] = exc_y


        if condition == "supp_from_train_set":
            ## Get sets
            # max_c = min(classes_per_set, len(np.unique(Y)))
            max_c = classes_per_set
            total_data = max_c * samples_per_class
            support_set_x = np.zeros((batch_size, total_data,
                                        X.shape[1], X.shape[2], X.shape[3]), np.float32)
            support_set_y = np.zeros((batch_size, total_data), np.int32)

            # Any augmentation?
            augment_query_case, augment_supp_case = False, False
            if set == "train" and not only_nk:
                augment_query_case = True if (Constants["SIMCLR"] == "query") else False
                augment_supp_case  = True if (Constants["SIMCLR"] == "supportSet") else False

            for b in range(batch_size):
                exc = index_batch[b]
                
                supp_index = FewShotTrain.get_support_set_index(Y, exc, max_c, samples_per_class, only_nk=only_nk)

                if model_type == "PrototypicalNetwork":
                    out_1, out_2 = get_multiple_querys(exc, Y, X, supp_index, set=set, only_nk=only_nk)
                elif model_type == "MatchingNetwork":
                    out_1, out_2 = get_one_query(exc, Y, X, supp_index, set=set, only_nk=only_nk)
                elif model_type == "RelationNetwork":
                    out_1, out_2 = get_multiple_querys(exc, Y, X, supp_index, set=set, only_nk=only_nk)

                # # change Y labels to equal size of num_classes  
                # supp_y, exc_y = FewShotTrain.codify_subset_classes(Y[supp_index], Y[exc], max_c, model_type)

                supp_y = Y[supp_index]
                if not augment_query_case:
                    if not augment_supp_case:
                        exc_y = Y[out_1]
                        exc_x = X[out_1]                         
                    # Augment support set
                    else:
                        exc_y = out_2["query_class"]
                        exc_x = out_2["query_obj"]
                # Augment only query
                else:
                    exc_y = Y[out_1]
                    exc_x = out_2

                supp_x, supp_y = X[supp_index], supp_y
                if Constants["SHUFFLE_SUPP_SET"]:
                    p = np.random.permutation(len(supp_y))
                    supp_x, supp_y = X[supp_index][p], supp_y[p]

                if augment_supp_case:
                    supp_x = out_1

                support_set_x[b] = supp_x # Random subset to train
                support_set_y[b] = supp_y # np.expand_dims(y_temp[:], axis=1)
                target_x[b] = exc_x 
                target_y[b] = exc_y



        elif condition == "all_training_data":
            ## Get sets
            support_set_x = np.zeros((batch_size, X.shape[0] - 1,
                                        X.shape[1], X.shape[2], X.shape[3]), np.float32)

            support_set_y = np.zeros((batch_size, X.shape[0] - 1), np.int32)

            for i in range(batch_size):
                # classes_idx = np.arange(X.shape[0])
                # choose_classes = np.random.choice(classes_idx, size=classes_per_set, replace=False)

                choose_classes = np.arange(X.shape[0]) # Use all examples
                if Constants["SHUFFLE_SUPP_SET"]:
                    np.random.shuffle(choose_classes)
                choose_to_predict = [choose_classes[-1]]
                choose_classes = choose_classes[:-1]

                support_set_x[i] = X[choose_classes] # Random subset to train
                support_set_y[i] = Y[choose_classes] # np.expand_dims(y_temp[:], axis=1)
                
                target_x[i] = X[choose_to_predict] 
                target_y[i] = Y[choose_to_predict] 


        if Constants["change_classes_alias"]:
            for b in range(batch_size):
                sub_support_set_x, sub_support_set_y = support_set_x[b], support_set_y[b]
                max_s = np.max(sub_support_set_y)
 
                ## DOING ONLY CHANGE HERE
                classes = np.unique(sub_support_set_y)
                # Change each class to a new one
                for i in range(len(classes)):
                    sub_support_set_y[sub_support_set_y == classes[i]] = i
                    if len(target_y[b]) > 1:
                        ## Prototypical case, multiple queries
                        for j in range(len(target_y[b])):
                            if classes[i] == target_y[b][j]:
                                target_y[b][j] = i
                    # MatchingNetwork case, only one query per episode
                    else:
                        if classes[i] == target_y[b]:
                            target_y[b] = i
                p     = np.random.permutation(len(sub_support_set_y))
                p_tgt = np.random.permutation(len(target_y[b]))
                support_set_x[b], support_set_y[b] = sub_support_set_x[p], sub_support_set_y[p]
                target_x[b], target_y[b] = target_x[b][p_tgt], target_y[b][p_tgt]

        return support_set_x, support_set_y, target_x, target_y


# N_query for each class needed at PrototypicalNetworks
def get_multiple_querys(exc, Y, X, supp_index, set, only_nk=False):
    new_exc, new_exc_obj = [], []
    # Only classes present in supp
    supp_classes, counts = np.unique(Y[supp_index], return_counts=True)

    # if not only_nk:
    #     supp_classes = np.unique(Y[supp_index])
    # else:
    #     supp_classes = np.unique(supp_index)

    if Constants["SIMCLR"] == "supportSet" and set == "train" and not only_nk:
        new_supp_xs, query = augment_support_only(supp_index, supp_classes, X, Y, multiple_query=True)

        return new_supp_xs, query

    for class_ in supp_classes:

        # For exp augmentation on only query. This is train but not on the support set on the special FineTuning phase
        if Constants["SIMCLR"] == "query" and set == "train" and not only_nk:
            # Choose one samples of class class_. (Supp_index are index because inside here is not only_nk)
            class_x_samples = np.array(supp_index[(Y[supp_index] == class_).nonzero(as_tuple=True)[0]])
            if class_x_samples.size > 1:
                sample_index_x = int(class_x_samples[np.random.randint(0, class_x_samples.size)].item())
            else:
                sample_index_x = int(class_x_samples.item())

            img_to_augment = X[sample_index_x]
            new_exc.append( sample_index_x )
            new_exc_obj.append( apply_simclr_augmentation(img_to_augment) )
        
        
        else:
            index = (Y == class_).nonzero(as_tuple=True)[0]

            if exc in index:
                new_exc.append(exc)
            else:
                if not only_nk:
                    sub_index = [i for i in index if i not in supp_index]
                    if len(sub_index) > 1:
                        index = sub_index

                new_exc.append(np.random.choice(index, Constants["n_query"], replace=False)[0])

    if len(new_exc_obj) != 0:
        return new_exc, torch.stack(new_exc_obj)
    return new_exc, None

# Just one query from same class than Ysupp for MatchingNetwork
def get_one_query(exc, Y, X, supp_index, set, only_nk=False):
    # Only classes present in supp
    supp_classes, counts = np.unique(Y[supp_index], return_counts=True)
    spc = counts[0]

    # For exp augmentation on only query. This is train but not on the support set on the special FineTuning phase
    if Constants["SIMCLR"] == "query" and set == "train" and not only_nk:
        sample_index_x = supp_index[np.random.randint(0, len(supp_index))]
        img_to_augment = X[sample_index_x]
        new_exc_obj = apply_simclr_augmentation(img_to_augment)
        return sample_index_x, new_exc_obj
    # Choose only 1 sample per class, after that augment those k times. Also not preserving query
    elif Constants["SIMCLR"] == "supportSet" and set == "train" and not only_nk:
        new_supp_xs, query = augment_support_only(supp_index, supp_classes, X, Y, multiple_query=False)
        return new_supp_xs, query

    # Add a query from the Y set    
    else:
        # Choose a random int from the class
        random_int = np.random.randint(0, len(supp_classes))

        # Set of Y samples from same supp classes, of the obj class
        index = (Y == supp_classes[random_int]).nonzero(as_tuple=True)[0]
        # Check if exc is already one of the searched classes
        if exc in index:
            new_exc = exc
        else:
            # In the standard case, we do not want to include one of the samples in the support set as query
            if not only_nk:
                sub_index = [i for i in index if i not in supp_index]
                if len(sub_index) > 1:
                    index = sub_index
            # try:        
            new_exc = np.random.choice(index, Constants["n_query"], replace=False)[0]
            # except:
            #     breakpoint()
        return new_exc, None


def augment_support_only(supp_index, supp_classes, X, Y, multiple_query=False):
    num_classes = len(supp_classes)
    # extract only n random samples, the rest will not be used 
    sample_indexes_x = np.array(supp_index)[np.random.randint(0, len(supp_index), num_classes)]
    query_obj, query_class = torch.tensor([]), torch.tensor([])
    if not multiple_query:
        sample_index_x_q   = sample_indexes_x[np.random.randint(0, len(sample_indexes_x))]
        query_obj, query_class  = X[sample_index_x_q], Y[sample_index_x_q]

    new_supp_xs = torch.tensor([])
    dictionary_classes = {}

    # Times to augment new x images. Iterate through real array
    for s_index in supp_index:
        real_class  = Y[s_index].item()
        if real_class not in dictionary_classes:
            dictionary_classes[real_class] = len(dictionary_classes)
            index_q = sample_indexes_x[dictionary_classes[real_class]]
            # Store the query only once per class, as it is not augmented
            if multiple_query:
                # Tricking to reuse the Y vector
                query_obj   = torch.cat((query_obj,   X[index_q].unsqueeze(0)), dim=0)
                query_class = torch.cat((query_class, Y[s_index].unsqueeze(0)), dim=0)
        real_sample_to_augment = X[sample_indexes_x[dictionary_classes[real_class]]]
        
        # Use the real images only for the query
        # if s_index in sample_indexes_x:
        #     # If the sample is already in the support set, do not augment it
        #     new_supp_xs = torch.cat((new_supp_xs, X[s_index].unsqueeze(0)), dim=0)
        #     continue 
        

        new_supp_xs = torch.cat((new_supp_xs, apply_simclr_augmentation(real_sample_to_augment, class_=real_class).unsqueeze(0)), dim=0)
    return new_supp_xs, {"query_obj": query_obj, "query_class": query_class}

def extend_dimenstions_bs(arr, batch_size):
    arr_expanded = np.expand_dims(arr, axis=0)

    # Duplicamos la información x veces
    arr_duplicated = np.repeat(arr_expanded, batch_size, axis=0)

    return arr_duplicated
