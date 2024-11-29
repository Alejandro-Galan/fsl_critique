import tqdm, torch, wandb, os, random
import numpy as np
from torch.autograd import Variable

from my_utils.constants import Constants
from network.loss import prototypical_loss
from models.PrototypicalNetwork import PrototypicalNetwork


class FewShotTrain():

    def train_few_shot_net(batch_size, encoder, X, Y, device, classes_per_set, samples_per_class, X_eval, Y_eval, checkpoint_path, 
                           model_type="Original", optimizer = None, episodes=1000, metrics=None):

        encoder.train()
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)

        
        best_epoch = 0.0
        epochs_no_improve = 0
        
        # total_train_batches = (X.shape[0] + batch_size - 1) // batch_size
        # total_train_batches = X.shape[0] // batch_size
        total_train_batches = episodes // batch_size # As it is random, not fixed

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                # * number of samples
                # end_i = (i + 1) * batch_size # No need to cut, is not sequential
                # if end_i > X.shape[0]: 
                #     end_i = X.shape[0]
                # index_batch = indexes[i*batch_size:end_i]
                index_batch = np.random.choice(X.shape[0], batch_size, replace=False)
                x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, metrics=metrics, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
                
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
                    acc, c_loss, outs = encoder(support_set_images=x_support_set.cuda(), support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda())
                elif model_type == "PrototypicalNetwork":
                    all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                    acc, c_loss = prototypical_loss(all_outputs, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)

                wandb.log({"tr_acc": acc, "tr_loss": c_loss})

                # optimize process
                optimizer.zero_grad()
                c_loss.backward()
                optimizer.step()

                FewShotTrain.adjust_learning_rate(optimizer)

                total_c_loss += c_loss.item()
                total_accuracy += acc.item()
                iter_out = "tr_loss: {}, tr_accuracy: {}, lr: {}".format(total_c_loss/(i+1), acc.item(), optimizer.param_groups[0]['lr'])
                pbar.set_description(iter_out)
                pbar.update(1)
                # self.total_train_iter+=1


                # limit = 20 if total_train_batches > 20 else total_train_batches
                limit = total_train_batches // Constants.LIMIT_VALIDATION_SRC  # Twice # One per epoch
                if Constants.VALIDATION_SRC and (i + 1) % limit == 0:
                # if False:
                    encoder.eval()
                    # supp_set = {"imgs": X, "labels": Y}
                    # supp_set = {"imgs": self.XSupp, "labels": self.YSupp} ## TODO which supp to choose
                    
                    test_accs, test_loss, test_outputs = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                                            device=device, model_type=model_type, supp_set=None, classes_per_set=classes_per_set, samples_per_class=samples_per_class, metrics=metrics)

                    # print("TEST ACCS", test_accs)
                    wandb.log({"tr_eval_acc": test_accs, "tr_eval_loss": test_loss})

                    if test_accs > metrics['best_accuracy']:
                        # print(
                        #     f"Test accuracy improved from {best_accuracy:.2f} to {test_accs:.2f}"
                        # )

                        metrics['best_accuracy'] = test_accs
                        best_epoch = i
                        # metrics['best_class_rep'] = classification_report(y_true=Y_eval.cpu().tolist(), y_pred=test_outputs.cpu(), output_dict=True)

                        ## If fine tune, evaluating with this is cheating
                        # torch.save({
                        #     'model_state_dict': encoder.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict(),
                        # }, checkpoint_path.replace("encoder.pt", "_trained_model.pt") )

                        epochs_no_improve = 0

                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= metrics['PATIENCE']:
                            print("Early stopping triggered")
                            break


                    encoder.train()


        # Save only in last iteration
        new_path = checkpoint_path.replace("encoder.pt", "_trained_model.pt")
        os.makedirs(os.path.dirname(new_path), exist_ok=True )

        torch.save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },  new_path)


        return metrics['best_accuracy'], best_epoch, optimizer, None

        # total_c_loss = total_c_loss / total_train_batches
        # total_accuracy = total_accuracy / total_train_batches
        # return total_c_loss, total_accuracy, optimizer


    def eval_few_shot_net(batch_size, encoder, X, Y, X_train, Y_train, device, classes_per_set, samples_per_class, metrics, model_type="Original", supp_set=None, finetune=False):
        All_acc, All_out = [], []

        encoder.eval()
        with torch.no_grad():
            indexes = np.arange(X.shape[0])
            np.random.shuffle(indexes)

            # offset_limit = (X.shape[0] + batch_size - 1)
            offset_limit = X.shape[0] # No need to pick the extra
            total_test_batches = offset_limit // batch_size



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
                    end_ind = ind + batch_size
                    if end_ind <= X.shape[0]:
                        # end_ind = min(ind + batch_size, X.shape[0])

                        ### Iterating for each batch. Obtain that target and support from test set
                        index_batch = np.random.choice(X.shape[0], batch_size, replace=False)    
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

                        if model_type == "MatchingNetwork":
                            acc, loss, output = encoder(support_set_images=x_support_set.to(device), 
                                                    support_set_labels_one_hot=y_support_set_one_hot.to(device).float(), 
                                                    target_image=x_target.to(device), target_label=y_target.to(device))
                        elif model_type == "PrototypicalNetwork":
                            output, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                            acc, c_loss = prototypical_loss(output, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)
        

                        All_acc.append(acc.cpu().item())
                        if output.dim() == 0:
                            All_out.append(output.unsqueeze(0) )
                        else:                              
                            All_out.append(output )
                
                    if (ind / batch_size) % 5 == 0:
                        try:
                            if finetune:
                                iter_out = "Finetune: Test Accuracy: " + str(np.mean(All_acc) )
                            else:
                                iter_out = "SrcTrain: Test Accuracy: " + str(np.mean(All_acc) )
                        except:
                            breakpoint()
                        pbar.set_description(iter_out)
                        pbar.update(5)
                # Asegúrate de actualizar la barra de progreso al final
                pbar.update(total_test_batches - pbar.n)

        return np.mean(All_acc), None, torch.cat(All_out)


    def finetune_few_shot_net(batch_size, encoder, X, Y, device, classes_per_set, samples_per_class, X_eval, Y_eval, checkpoint_path, 
                           model_type="", optimizer = None, episodes=1000, metrics=None):

        encoder.train()
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)

        
        best_epoch = 0.0
        epochs_no_improve = 0
        
        # total_train_batches = (X.shape[0] + batch_size - 1) // batch_size
        # total_train_batches = X.shape[0] // batch_size
        total_train_batches = episodes // batch_size # As it is random, not fixed

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                # * number of samples
                # end_i = (i + 1) * batch_size # No need to cut, is not sequential
                # if end_i > X.shape[0]: 
                #     end_i = X.shape[0]
                # index_batch = indexes[i*batch_size:end_i]
                index_batch = np.random.choice(X.shape[0], batch_size, replace=False)
                x_support_set, y_support_set, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, metrics=metrics, 
                                                                                                    classes_per_set=classes_per_set, samples_per_class=samples_per_class,
                                                                                                    only_nk=True)
                
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
                    acc, c_loss, outs = encoder(support_set_images=x_support_set.cuda(), support_set_labels_one_hot=y_support_set_one_hot.cuda(), target_image=x_target.cuda(), target_label=y_target.cuda())
                elif model_type == "PrototypicalNetwork":
                    all_outputs, inputs_y = PrototypicalNetwork.get_outputs(x_target, y_target, x_support_set, y_support_set, encoder)
                    acc, c_loss = prototypical_loss(all_outputs, target=inputs_y, n_support=samples_per_class, samples_per_class=samples_per_class, batch_size=batch_size)

                wandb.log({"ft_acc": acc, "ft_loss": c_loss})

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
                if (i + 1) % limit == 0:
                # if True:
                    encoder.eval()
                    # supp_set = {"imgs": X, "labels": Y}
                    # supp_set = {"imgs": self.XSupp, "labels": self.YSupp} ## TODO which supp to choose
                    
                    test_accs, test_loss, test_outputs = FewShotTrain.eval_few_shot_net(batch_size=batch_size, encoder=encoder, X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                                            device=device, model_type=model_type, supp_set=None, classes_per_set=classes_per_set, samples_per_class=samples_per_class, metrics=metrics, finetune=True)

                    # print("TEST ACCS", test_accs)
                    wandb.log({"ft_eval_acc": test_accs, "ft_eval_loss": test_loss})

                    if test_accs > metrics['best_accuracy']:
                        # print(
                        #     f"Test accuracy improved from {best_accuracy:.2f} to {test_accs:.2f}"
                        # )

                        metrics['best_accuracy'] = test_accs
                        best_epoch = i
                        # metrics['best_class_rep'] = classification_report(y_true=Y_eval.cpu().tolist(), y_pred=test_outputs.cpu(), output_dict=True)

                        torch.save({
                            'model_state_dict': encoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_path.replace("encoder.pt", "_trained_finetuned_model.pt") )

                        epochs_no_improve = 0

                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= metrics['PATIENCE']:
                            print("Early stopping triggered")
                            break


                    encoder.train()

        return metrics['best_accuracy'], best_epoch, optimizer, None



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
        # if Constants.LIMIT_N_WAY:
        #     ind_exc = np.where(Y[exc].item() == all_classes)[0][0]
        #     n_all_classes = [ind_exc]
        #     all_classes = np.delete(all_classes, ind_exc)
        #     np.random.shuffle(all_classes)

        #     new_classes = random.sample(sorted(all_classes), classes_per_set - 1)
        #     n_all_classes.extend(new_classes)

        #     all_classes = n_all_classes
            
        # if set == "train":
        # np.random.shuffle(all_classes)

        subset_classes = [Y[exc]]
        subset_classes = [x for x in all_classes if x != Y[exc]]
        subset_classes = random.sample(subset_classes, classes_per_set - 1)
        subset_classes.append(Y[exc])
        if Constants.SHUFFLE_SUPP_SET:
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
                try:
                    index_ch = np.random.choice(index, samples_per_class, replace=False)
                except:
                    breakpoint()

                # if set == "eval":
                #     index_ch = index[np.arange(0, samples_per_class)]
                
            all_index = all_index + list(index_ch)
        if Constants.SHUFFLE_SUPP_SET:
            np.random.shuffle(all_index) # TODO LSTM importa?


        return all_index

    # def get_only_a_subsample_set( X, Y, batch_size, classes_per_set, samples_per_class=1):
    #     index_batch = np.random.choice(X.shape[0], batch_size, replace=False)
    #     # Only choose support label here 
    #     imgs, labels, x_target, y_target = FewShotTrain.get_subsamples_sets(X, Y, index_batch, model_type=model_type, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    #     return imgs[0], labels[0], x_target[0], y_target[0]
        
    # Convert every episode classes to a transcription of size "num_classes_per_set"
    def codify_subset_classes( supp, exc, max_c):

        new_vec = np.append(supp, exc)
        classes = {}
        
        for e in range(len(new_vec)):
            class_ = new_vec[e]
            if class_ not in classes:
                classes[class_] = len(classes)
            new_vec[e] = classes[class_]

        assert len(classes) <= max_c

        supp_v, query_v = new_vec[:len(supp)], new_vec[len(supp):]
        
        return supp_v, query_v

    def get_subsamples_sets( X, Y, index_batch, classes_per_set, metrics, model_type="", samples_per_class=1, set="train", only_nk=False):

        batch_size = len(index_batch) # The remainder could be lower

        ########
        ## Testing best approach. All_training data means all extracted for batch permuted
        condition = "supp_from_train_set" # "all_training_data" #  prefixed_support_set 
        if set == "eval" and Constants.USE_ORIGINAL_FIXED_SUPP_SET:
            condition = "prefixed_support_set"
        ########   


        # if len(X.shape) == 2:
        #     X = X.unsqueeze(0)

        if model_type == "PrototypicalNetwork":
            target_x = np.zeros((batch_size, classes_per_set, X.shape[1], X.shape[2], X.shape[3]), np.float32)
            target_y = np.zeros((batch_size, classes_per_set), np.int32)
        else:
            target_x = np.zeros((batch_size, X.shape[1], X.shape[2], X.shape[3]), np.float32)
            target_y = np.zeros((batch_size, 1), np.int32)

        if condition == "prefixed_support_set":

            ## Get sets
            total_data = classes_per_set * samples_per_class

            assert total_data == len(metrics['YSupp'])
            support_set_x = np.zeros((batch_size, total_data,
                                        X.shape[1], X.shape[2], X.shape[3]), np.float32)
            support_set_y = np.zeros((batch_size, total_data), np.int32)

            for b in range(batch_size):
                exc = index_batch[b]

                if model_type == "PrototypicalNetwork":
                    exc = get_multiple_querys(exc, Y, metrics['YSupp'], only_nk)

                # change Y labels to equal size of num_classes 
                supp_y, exc_y = FewShotTrain.codify_subset_classes(metrics['YSupp'], Y[exc], classes_per_set)
                ############ DEBUG ##################
                # index_debug_fixed = np.arange(0,samples_per_class)
                # supp_y, exc_y = FewShotTrain.codify_subset_classes(Y[index_debug_fixed], Y[exc], classes_per_set)
                #####################################

                supp_x, supp_y = metrics['XSupp'], supp_y
                if Constants.SHUFFLE_SUPP_SET:
                    p = np.random.permutation(len(supp_y))
                    supp_x, supp_y = metrics['XSupp'][p], supp_y[p]

                support_set_x[b] = supp_x
                support_set_y[b] = supp_y

                target_x[b] = X[exc] 
                target_y[b] = exc_y



        elif condition == "supp_from_train_set":
            ## Get sets
            # max_c = min(classes_per_set, len(np.unique(Y)))
            max_c = classes_per_set
            total_data = max_c * samples_per_class
            support_set_x = np.zeros((batch_size, total_data,
                                        X.shape[1], X.shape[2], X.shape[3]), np.float32)
            support_set_y = np.zeros((batch_size, total_data), np.int32)

            for b in range(batch_size):
                exc = index_batch[b]

                supp_index = FewShotTrain.get_support_set_index(Y, exc, max_c, samples_per_class, only_nk=only_nk)
                if model_type == "PrototypicalNetwork":
                    exc = get_multiple_querys(exc, Y, supp_index)

                # change Y labels to equal size of num_classes 
                supp_y, exc_y = FewShotTrain.codify_subset_classes(Y[supp_index], Y[exc], max_c)

                supp_x, supp_y = X[supp_index], supp_y
                if Constants.SHUFFLE_SUPP_SET:
                    p = np.random.permutation(len(supp_y))
                    supp_x, supp_y = X[supp_index][p], supp_y[p]


                support_set_x[b] = supp_x # Random subset to train
                support_set_y[b] = supp_y # np.expand_dims(y_temp[:], axis=1)
                
                target_x[b] = X[exc] 
                target_y[b] = exc_y


        elif condition == "all_training_data":
            ## Get sets
            support_set_x = np.zeros((batch_size, X.shape[0] - 1,
                                        X.shape[1], X.shape[2], X.shape[3]), np.float32)

            support_set_y = np.zeros((batch_size, X.shape[0] - 1), np.int32)

            for i in range(batch_size):
                # classes_idx = np.arange(X.shape[0])
                # TODO: Check if trains better with a subset than with all available
                # choose_classes = np.random.choice(classes_idx, size=classes_per_set, replace=False)

                choose_classes = np.arange(X.shape[0]) # Use all examples
                if Constants.SHUFFLE_SUPP_SET:
                    np.random.shuffle(choose_classes)
                choose_to_predict = [choose_classes[-1]]
                choose_classes = choose_classes[:-1]

                support_set_x[i] = X[choose_classes] # Random subset to train
                support_set_y[i] = Y[choose_classes] # np.expand_dims(y_temp[:], axis=1)
                
                target_x[i] = X[choose_to_predict] 
                target_y[i] = Y[choose_to_predict] 

        return support_set_x, support_set_y, target_x, target_y


# N_query for each class needed at PrototypicalNetworks
def get_multiple_querys(exc, Y, supp_index, only_nk=False):
    new_exc = []
    # Only classes present in supp
    if not only_nk:
        supp_classes = np.unique(Y[supp_index])
    else:
        supp_classes = np.unique(supp_index)

    for class_ in supp_classes:
        index = (Y == class_).nonzero(as_tuple=True)[0]

        if exc in index:
            new_exc.append(exc)
        else:
            if not only_nk:
                sub_index = [i for i in index if i not in supp_index]
                if len(sub_index) > 1:
                    index = sub_index
            
            new_exc.append(np.random.choice(index, Constants.n_query, replace=False)[0])
    return new_exc

def extend_dimenstions_bs(arr, batch_size):
    arr_expanded = np.expand_dims(arr, axis=0)

    # Duplicamos la información x veces
    arr_duplicated = np.repeat(arr_expanded, batch_size, axis=0)

    return arr_duplicated
