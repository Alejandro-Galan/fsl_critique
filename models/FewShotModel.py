"""
Few-Shot training, evaluation, and fine-tuning logic.

All public methods live on FewShotTrain as static methods to preserve the
original call-sites in train_fs.py.
"""

import importlib
import os
import random
import sys

import numpy as np
import torch
import tqdm
from torch.autograd import Variable

from datasets.loader import apply_simclr_augmentation, get_params_for_loading_datasets, load_supervised_data
from network.loss import prototypical_loss
from models.PrototypicalNetwork import PrototypicalNetwork
from utils import constants
from utils.constants import Const_c
from utils.train_utils import debug_images_and_dists, split_train_test

importlib.reload(constants)

# ---------------------------------------------------------------------------
# Global experiment constants
# ---------------------------------------------------------------------------
exp       = str(sys.argv[2])
full_name = str(sys.argv[3])

Constants_c = Const_c(exp, full_name)
Constants   = Constants_c.Constants


# ===========================================================================
# Main class
# ===========================================================================

class FewShotTrain:
    """Static container for few-shot train / eval / finetune routines."""

    # -----------------------------------------------------------------------
    # Pre-training
    # -----------------------------------------------------------------------

    def train_few_shot_net(
        batch_size, encoder, X, Y, device, classes_per_set, samples_per_class,
        X_eval, Y_eval, checkpoint_path,
        model_type="", optimizer=None, metrics=None,
        scheduler=None, X_val=None, Y_val=None,
    ):
        encoder.train()
        total_c_loss   = 0.0
        total_accuracy = 0.0
        best_epoch     = 0.0
        epochs_no_improve = 0

        new_path = checkpoint_path + "_trained_model.pt"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        total_batches  = Constants["EPISODES"] // batch_size
        limit_val_tgt  = total_batches // Constants["LIMIT_VALIDATION_SRC_TGT"]
        limit_val_src  = total_batches // Constants["LIMIT_VALIDATION_SRC_SRC"]

        with tqdm.tqdm(total=total_batches) as pbar:
            for i in range(total_batches):
                index_batch = np.random.choice(X.shape[0], batch_size, replace=False)

                # ---- forward pass ----
                acc, c_loss, optimizer = _run_train_step(
                    encoder, X, Y, index_batch,
                    batch_size, classes_per_set, samples_per_class,
                    model_type, metrics, optimizer, scheduler,
                )

                total_c_loss   += c_loss.item()
                total_accuracy += acc.item()

                desc = f"tr_loss: {total_c_loss/(i+1):.4f}  acc: {acc.item():.4f}  mean: {total_accuracy/(i+1):.4f}"
                if model_type != "RelationNetwork":
                    desc += f"  lr: {optimizer.param_groups[0]['lr']:.6f}"
                pbar.set_description(desc)
                pbar.update(1)

                # ---- source validation ----
                if Constants["VALIDATION_SRC_SRC_DATA"] and (i + 1) % limit_val_src == 0:
                    encoder.eval()
                    val_accs, _, _, _ = FewShotTrain.eval_few_shot_net(
                        batch_size=batch_size, encoder=encoder,
                        X=X_val, Y=Y_val, X_train=X, Y_train=Y,
                        device=device, model_type=model_type,
                        supp_set=None, classes_per_set=classes_per_set,
                        samples_per_class=samples_per_class,
                        metrics=metrics, set="Val",
                    )
                    val_acc = val_accs[0]
                    if val_acc > metrics["best_accuracy"]:
                        metrics["best_accuracy"] = val_acc
                        best_epoch = i
                        epochs_no_improve = 0
                        FewShotTrain.store_encoder(encoder, optimizer, new_path, model_type)
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= metrics["PATIENCE"]:
                            print("Early stopping triggered")
                            break

                encoder.train()

                # ---- target validation (observation only) ----
                if Constants["VALIDATION_SRC_TGT_DATA"] and (i + 1) % limit_val_tgt == 0:
                    tgt_accs, _, _, _ = FewShotTrain.eval_few_shot_net(
                        batch_size=batch_size, encoder=encoder,
                        X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                        device=device, model_type=model_type,
                        supp_set=None, classes_per_set=classes_per_set,
                        samples_per_class=samples_per_class, metrics=metrics,
                    )
                    encoder.train()

        # Save final checkpoint if not using source-validation-based saving
        if not Constants["VALIDATION_SRC_SRC_DATA"]:
            FewShotTrain.store_encoder(encoder, optimizer, new_path, model_type)

        return metrics["best_accuracy"], best_epoch, optimizer, None


    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    def eval_few_shot_net(
        batch_size, encoder, X, Y, X_train, Y_train, device,
        classes_per_set, samples_per_class, metrics,
        model_type="Original", supp_set=None,
        finetune=False, set="Test", debug_images=False,
    ):
        all_acc, all_out, all_dists = [], [], []
        total_batches = Constants["TEST_EPISODES"] // batch_size
        indexes       = np.arange(X.shape[0])

        encoder.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=total_batches) as pbar:
                for ind in range(total_batches):
                    ind     = ind % len(indexes)
                    end_ind = ind + batch_size
                    if end_ind > len(indexes):
                        continue

                    index_batch = indexes[ind:end_ind]
                    acc, loss, output, dists = _run_eval_step(
                        encoder, X, Y, index_batch,
                        batch_size, classes_per_set, samples_per_class,
                        model_type, metrics, device, debug_images,
                    )

                    # Accumulate distances for exp3
                    if exp == "exp3" and model_type in ("PrototypicalNetwork", "RelationNetwork"):
                        d = dists.cpu().numpy()
                        all_dists = d if len(all_dists) == 0 else np.concatenate((all_dists, d), axis=0)

                    all_acc.append(acc.cpu().item())
                    if output is not None:
                        all_out.append(output.unsqueeze(0) if output.dim() == 0 else output)

                    if (ind // batch_size) % 5 == 0:
                        prefix = "Finetune" if finetune else ">"
                        pbar.set_description(f"{prefix}: {set} Acc {np.mean(all_acc):.4f}")
                        pbar.update(5)

                pbar.update(total_batches - pbar.n)

        mean_std = [np.mean(all_acc), np.std(all_acc)]
        if not all_out:
            return mean_std, None, None, all_dists
        return mean_std, None, torch.cat(all_out), all_dists


    # -----------------------------------------------------------------------
    # Fine-tuning
    # -----------------------------------------------------------------------

    def finetune_few_shot_net(
        batch_size, encoder, X, Y, device, classes_per_set, samples_per_class,
        X_eval, Y_eval, checkpoint_path, boots_iter,
        model_type="", optimizer=None, metrics=None,
        ft_epoch_num=0, reuse_logs=True,
    ):
        total_c_loss      = 0.0
        total_accuracy    = 0.0
        best_epoch        = 0.0
        epochs_no_improve = 0
        total_batches     = Constants["FINE_TUNING_EPISODES"] // batch_size

        encoder.train()

        with tqdm.tqdm(total=total_batches) as pbar:
            for i in range(total_batches):
                index_batch = _sample_ft_batch(X, Y, batch_size)

                acc, c_loss, optimizer = _run_finetune_step(
                    encoder, X, Y, index_batch,
                    batch_size, classes_per_set, samples_per_class,
                    model_type, metrics, optimizer,
                )

                total_c_loss   += c_loss.item()
                total_accuracy += acc.item()
                pbar.set_description(
                    f"ft_loss: {total_c_loss/(i+1):.4f}  acc: {acc.item():.4f}"
                    f"  lr: {optimizer.param_groups[0]['lr']:.6f}"
                )
                pbar.update(1)

                end_of_epoch         = (i + 1) % total_batches == 0
                is_exp3_loss_case    = (
                    ft_epoch_num == 2 and exp == "exp3"
                    and model_type in ("PrototypicalNetwork", "RelationNetwork")
                )
                if not Constants["EXHAUSTIVE_LOSS_CURVES"]:
                    is_exp3_loss_case = is_exp3_loss_case and end_of_epoch

                if end_of_epoch or is_exp3_loss_case:
                    encoder.eval()
                    test_accs_std, _, _, dists = FewShotTrain.eval_few_shot_net(
                        batch_size=batch_size, encoder=encoder,
                        X=X_eval, Y=Y_eval, X_train=X, Y_train=Y,
                        device=device, model_type=model_type, supp_set=None,
                        classes_per_set=classes_per_set,
                        samples_per_class=samples_per_class,
                        metrics=metrics, finetune=True,
                    )
                    test_accs = test_accs_std[0]

                    if is_exp3_loss_case:
                        path_dists = Const_c.get_distances_path(
                            Constants=Constants, sufix_path="_after_ft_distances", boots_iter=boots_iter
                        )
                        if not os.path.exists(path_dists):
                            os.makedirs(os.path.dirname(path_dists), exist_ok=True)
                            FewShotTrain.calculate_distances(
                                encoder, dists, test_accs_std,
                                list(Constants["TGT_DATASETS"].keys())[0],
                                model_type, device, batch_size,
                                classes_per_set, samples_per_class,
                                boots_iter=boots_iter, end_of_epoch=end_of_epoch,
                            )

                    if end_of_epoch:
                        if test_accs > metrics["best_accuracy"]:
                            metrics["best_accuracy"] = test_accs
                            best_epoch = i
                            epochs_no_improve = 0
                            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                            try:
                                save_path = Const_c.get_id_extensions(Constants, prev_str=checkpoint_path)
                                FewShotTrain.store_encoder(encoder, optimizer, save_path, model_type)
                            except Exception:
                                coded = Const_c.add_to_dictionary_of_files(
                                    Const_c.get_id_extensions(Constants, prev_str=checkpoint_path)
                                ) + ".pt"
                                FewShotTrain.store_encoder(encoder, optimizer, coded, model_type)
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= metrics["PATIENCE"]:
                                print("Early stopping triggered")
                                break

                    encoder.train()

        return metrics["best_accuracy"], best_epoch, optimizer, None


    # -----------------------------------------------------------------------
    # Distance analysis (exp3)
    # -----------------------------------------------------------------------

    def calculate_distances(encoder, dists_base, accs_base, ds_name, model_type,
                             device, batch_size, classes_per_set, samples_per_class,
                             boots_iter, end_of_epoch):
        path_dists = Const_c.get_distances_path(
            Constants=Constants, sufix_path="_after_ft_distances", boots_iter=boots_iter
        )
        os.makedirs(os.path.dirname(path_dists), exist_ok=True)

        if end_of_epoch:
            np.testing.assert_equal(len(dists_base) > 0, True)
            FewShotTrain.store_distances(dists_base, accs=accs_base, full_path=path_dists)

        for src_ds in Constants["ALL_TST_DATASETS"]:
            if Constants["NUM_SRC_CLASSES_DATASETS"][src_ds] < Constants["LIMIT_N_WAY_TEST"]:
                continue

            pretrained_sources = bool(Constants["NoSrcDataset"])
            data = load_supervised_data(
                ds_name=src_ds, min_occurence=50,
                all_datasets=False, pretrained_sources=pretrained_sources, boots_iter=boots_iter,
            )
            PARAMS = get_params_for_loading_datasets(
                "NO_SRC_DATASET", src_ds, samples_per_class, boots_iter, Constants=Constants
            )
            _, _, XTest, YTest, _, _, _, _ = split_train_test(
                PARAMS=PARAMS, X=data["X_tgt"], Y=data["Y_tgt"]
            )

            debug_prefix = False
            if Constants["DEBUG_IMAGES"] and end_of_epoch:
                debug_prefix = f"{Constants['MODEL_TYPE']}/trained_{ds_name}_over_{src_ds}/"

            encoder.eval()
            accs_std, _, _, dists_src = FewShotTrain.eval_few_shot_net(
                batch_size=batch_size, encoder=encoder,
                X=torch.from_numpy(XTest), Y=torch.from_numpy(YTest),
                X_train=None, Y_train=None,
                device=device, model_type=model_type, supp_set=None,
                classes_per_set=classes_per_set,
                samples_per_class=samples_per_class,
                metrics={}, finetune=True, debug_images=debug_prefix,
            )
            test_acc = accs_std[0]

            if end_of_epoch:
                path_src = Const_c.get_distances_path(
                    Constants=Constants,
                    sufix_path=f"ft_distances_over_{src_ds}as_tgt_ds",
                    boots_iter=boots_iter,
                )
                FewShotTrain.store_distances(dists_src, accs=accs_std, full_path=path_src)

            print(f"test_acc-{src_ds}_trained_on-{ds_name}: {test_acc}")


    # -----------------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------------

    def store_distances(dists, accs, full_path):
        if os.path.exists(full_path):
            return
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, dists)
        np.save(full_path.replace(".npy", "_accs.npy"), accs)
        print(f"Distances saved to {full_path}")


    def store_encoder(encoder, optimizer, new_path, model_type):
        if model_type == "RelationNetwork":
            torch.save({
                "feature_encoder":                    encoder.feature_encoder.state_dict(),
                "relation_network":                   encoder.relation_network.state_dict(),
                "feature_encoder_optimizer_state_dict": encoder.feature_encoder_optim.state_dict(),
                "relation_network_optimizer_state_dict": encoder.relation_network_optim.state_dict(),
            }, new_path)
        else:
            torch.save({
                "model_state_dict":     encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, new_path)


    # -----------------------------------------------------------------------
    # Learning-rate decay
    # -----------------------------------------------------------------------

    def adjust_learning_rate(optimizer):
        """Decay LR following the original Lua SGD implementation."""
        for group in optimizer.param_groups:
            group.setdefault("step", 0)
            group["step"] += 1
            group["lr"] = group["lr"] / (1 + group["step"] * group["weight_decay"])


    # -----------------------------------------------------------------------
    # Support-set construction
    # -----------------------------------------------------------------------

    def include_exc(exc, index, only_nk, samples_per_class):
        """Return `samples_per_class` indices for the query class."""
        if not only_nk:
            index = index[(index != exc).nonzero(as_tuple=True)[0]]
        return list(np.random.choice(index, samples_per_class, replace=False))


    def get_support_set_index(Y, exc, classes_per_set, samples_per_class, set="train", only_nk=False):
        """Sample class indices for a support set, always including the query class."""
        all_classes = np.unique(Y)
        np.testing.assert_equal(len(all_classes) >= classes_per_set, True)
        np.testing.assert_equal(exc < len(Y), True)

        other_classes = [c for c in all_classes if c != Y[exc]]
        chosen        = random.sample(other_classes, classes_per_set - 1) + [Y[exc].item()]

        if Constants["SHUFFLE_SUPP_SET"]:
            random.shuffle(chosen)
        else:
            chosen.sort()

        all_index = []
        for c in chosen:
            index = (Y == c).nonzero(as_tuple=True)[0]
            if Y[exc].item() == c:
                all_index += FewShotTrain.include_exc(exc, index, only_nk, samples_per_class)
            else:
                all_index += list(np.random.choice(index, samples_per_class, replace=False))

        if Constants["SHUFFLE_SUPP_SET"]:
            np.random.shuffle(all_index)

        np.testing.assert_equal(len(all_index), classes_per_set * samples_per_class)
        return all_index


    def codify_subset_classes(supp, exc, max_c, model_type):
        """Identity mapping (class re-coding disabled)."""
        return supp, exc


    def get_subsamples_sets(X, Y, index_batch, classes_per_set, metrics,
                            model_type="", samples_per_class=1,
                            set="train", only_nk=False, change_classes_alias=False):
        """Build batched (support_x, support_y, target_x, target_y) tensors."""
        batch_size = len(index_batch)
        condition  = "supp_from_train_set"
        if set == "eval" and Constants["USE_ORIGINAL_FIXED_SUPP_SET"]:
            condition = "prefixed_support_set"

        target_x, target_y = _init_target_arrays(X, batch_size, classes_per_set, model_type)
        support_set_x, support_set_y = None, None

        if condition == "supp_from_train_set":
            support_set_x, support_set_y, target_x, target_y = _build_support_from_train(
                X, Y, index_batch, batch_size, classes_per_set, samples_per_class,
                model_type, metrics, set, only_nk, target_x, target_y,
            )

        elif condition == "all_training_data":
            support_set_x, support_set_y, target_x, target_y = _build_support_all_data(
                X, Y, batch_size, target_x, target_y,
            )

        if Constants["change_classes_alias"]:
            support_set_x, support_set_y, target_x, target_y = _remap_class_aliases(
                support_set_x, support_set_y, target_x, target_y, batch_size
            )

        return support_set_x, support_set_y, target_x, target_y


# ===========================================================================
# Module-level query helpers
# ===========================================================================

def get_multiple_querys(exc, Y, X, supp_index, set, only_nk=False):
    """Return query indices/tensors for PrototypicalNetwork and RelationNetwork."""
    supp_classes, _ = np.unique(Y[supp_index], return_counts=True)

    if Constants["SIMCLR"] == "supportSet" and set == "train" and not only_nk:
        return augment_support_only(supp_index, supp_classes, X, Y, multiple_query=True)

    new_exc, new_exc_obj = [], []
    for class_ in supp_classes:
        if Constants["SIMCLR"] == "query" and set == "train" and not only_nk:
            samples = np.array(supp_index[(Y[supp_index] == class_).nonzero(as_tuple=True)[0]])
            idx     = int(samples[np.random.randint(0, samples.size)].item()) if samples.size > 1 else int(samples.item())
            new_exc.append(idx)
            new_exc_obj.append(apply_simclr_augmentation(X[idx]))
        else:
            index = (Y == class_).nonzero(as_tuple=True)[0]
            if exc in index:
                new_exc.append(exc)
            else:
                if not only_nk:
                    sub = [i for i in index if i not in supp_index]
                    if len(sub) > 1:
                        index = sub
                new_exc.append(np.random.choice(index, Constants["n_query"], replace=False)[0])

    if new_exc_obj:
        return new_exc, torch.stack(new_exc_obj)
    return new_exc, None


def get_one_query(exc, Y, X, supp_index, set, only_nk=False):
    """Return a single query index/tensor for MatchingNetwork."""
    supp_classes, counts = np.unique(Y[supp_index], return_counts=True)

    if Constants["SIMCLR"] == "query" and set == "train" and not only_nk:
        idx = supp_index[np.random.randint(0, len(supp_index))]
        return idx, apply_simclr_augmentation(X[idx])

    if Constants["SIMCLR"] == "supportSet" and set == "train" and not only_nk:
        return augment_support_only(supp_index, supp_classes, X, Y, multiple_query=False)

    random_class = supp_classes[np.random.randint(0, len(supp_classes))]
    index        = (Y == random_class).nonzero(as_tuple=True)[0]
    if exc in index:
        return exc, None

    if not only_nk:
        sub = [i for i in index if i not in supp_index]
        if len(sub) > 1:
            index = sub
    return np.random.choice(index, Constants["n_query"], replace=False)[0], None


def augment_support_only(supp_index, supp_classes, X, Y, multiple_query=False):
    """Augment support samples; optionally build a query from the same samples."""
    sample_indexes = np.array(supp_index)[np.random.randint(0, len(supp_index), len(supp_classes))]
    query_obj, query_class = torch.tensor([]), torch.tensor([])

    if not multiple_query:
        q_idx = sample_indexes[np.random.randint(0, len(sample_indexes))]
        query_obj, query_class = X[q_idx], Y[q_idx]

    new_supp_xs      = torch.tensor([])
    class_to_aug_idx = {}

    for s_index in supp_index:
        real_class = Y[s_index].item()
        if real_class not in class_to_aug_idx:
            class_to_aug_idx[real_class] = len(class_to_aug_idx)
            q_idx = sample_indexes[class_to_aug_idx[real_class]]
            if multiple_query:
                query_obj   = torch.cat((query_obj,   X[q_idx].unsqueeze(0)), dim=0)
                query_class = torch.cat((query_class, Y[s_index].unsqueeze(0)), dim=0)

        aug_sample = X[sample_indexes[class_to_aug_idx[real_class]]]
        new_supp_xs = torch.cat(
            (new_supp_xs, apply_simclr_augmentation(aug_sample, class_=real_class).unsqueeze(0)),
            dim=0,
        )

    return new_supp_xs, {"query_obj": query_obj, "query_class": query_class}


# ===========================================================================
# Private step helpers
# ===========================================================================

def _to_variable(arr, requires_grad=False, dtype="float"):
    t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    t = t.float() if dtype == "float" else t.long()
    return Variable(t) if not requires_grad else Variable(t, requires_grad=False)


def _build_one_hot(y_support, batch_size, classes_per_set):
    y_support = torch.unsqueeze(y_support, 2)
    seq_len   = y_support.size(1)
    one_hot   = torch.FloatTensor(batch_size, seq_len, classes_per_set).zero_()
    one_hot.scatter_(2, y_support.data, 1)
    return Variable(one_hot)


def _forward_pass(encoder, x_supp, y_supp_hot, x_tgt, y_tgt, model_type,
                  samples_per_class, classes_per_set, batch_size, train=True,
                  device=None, y_supp_labels=None):
    if model_type == "MatchingNetwork":
        return encoder(
            support_set_images=x_supp.cuda(),
            support_set_labels_one_hot=y_supp_hot.cuda(),
            target_image=x_tgt.cuda(),
            target_label=y_tgt.cuda(),
        )[:2] + (None, None)

    if model_type == "PrototypicalNetwork":
        outputs, inputs_y = PrototypicalNetwork.get_outputs(x_tgt, y_tgt, x_supp, y_supp_hot, encoder)
        acc, c_loss, dists = prototypical_loss(
            outputs, target=inputs_y,
            n_support=samples_per_class,
            samples_per_class=samples_per_class,
            batch_size=batch_size,
        )
        return acc, c_loss, outputs, dists

    if model_type == "RelationNetwork":
        # RelationNetwork needs integer support labels to build class prototypes.
        # MatchingNetwork receives one-hot labels, but passing one-hot labels here
        # makes RelationNetwork index the support tensor with a flattened one-hot
        # mask and triggers CUDA "index out of bounds" errors.
        support_labels = y_supp_labels if y_supp_labels is not None else y_supp_hot.argmax(dim=-1)
        acc, c_loss, extra, dists = encoder(
            x_supp.cuda(), support_labels.cuda(),
            x_tgt.cuda(), y_tgt.cuda(),
            train=train,
            SAMPLE_NUM_PER_CLASS=samples_per_class,
            CLASS_NUM=classes_per_set,
        )
        return acc, c_loss, extra, torch.Tensor(np.array(dists))

    raise NotImplementedError(f"Unknown model_type: {model_type}")


def _run_train_step(encoder, X, Y, index_batch, batch_size, classes_per_set,
                    samples_per_class, model_type, metrics, optimizer, scheduler):
    x_supp, y_supp, x_tgt, y_tgt = FewShotTrain.get_subsamples_sets(
        X, Y, index_batch, model_type=model_type, metrics=metrics,
        classes_per_set=classes_per_set, samples_per_class=samples_per_class,
    )
    np.testing.assert_equal(x_supp.shape[1], classes_per_set * samples_per_class)

    x_supp    = _to_variable(x_supp, dtype="float")
    y_supp    = _to_variable(y_supp, requires_grad=False, dtype="long")
    x_tgt     = _to_variable(x_tgt,  dtype="float")
    y_tgt     = _to_variable(y_tgt,  requires_grad=False, dtype="long").squeeze()
    y_supp_oh = _build_one_hot(y_supp, batch_size, classes_per_set)

    acc, c_loss, _, _ = _forward_pass(
        encoder, x_supp, y_supp_oh, x_tgt, y_tgt,
        model_type, samples_per_class, classes_per_set, batch_size, train=True,
        y_supp_labels=y_supp,
    )

    if model_type != "RelationNetwork":
        optimizer.zero_grad()
        c_loss.backward()
        optimizer.step()
        for _ in range(batch_size):
            scheduler.step()
        FewShotTrain.adjust_learning_rate(optimizer)

    return acc, c_loss, optimizer


def _run_eval_step(encoder, X, Y, index_batch, batch_size, classes_per_set,
                   samples_per_class, model_type, metrics, device, debug_images):
    x_supp, y_supp, x_tgt, y_tgt = FewShotTrain.get_subsamples_sets(
        X, Y, index_batch, model_type=model_type, metrics=metrics,
        classes_per_set=classes_per_set, samples_per_class=samples_per_class,
        set="eval", only_nk=True,
    )

    x_supp    = _to_variable(x_supp, dtype="float")
    y_supp    = _to_variable(y_supp, requires_grad=False, dtype="long")
    x_tgt     = _to_variable(x_tgt,  dtype="float")
    y_tgt     = _to_variable(y_tgt,  requires_grad=False, dtype="long").squeeze()
    y_supp_oh = _build_one_hot(y_supp, batch_size, classes_per_set)

    acc, c_loss, output, dists = _forward_pass(
        encoder, x_supp, y_supp_oh, x_tgt, y_tgt,
        model_type, samples_per_class, classes_per_set, batch_size, train=False, device=device,
        y_supp_labels=y_supp,
    )

    if debug_images and dists is not None:
        debug_images_and_dists(x_tgt, x_supp, dists, debug_images=debug_images)

    return acc, c_loss, output, dists


def _run_finetune_step(encoder, X, Y, index_batch, batch_size, classes_per_set,
                       samples_per_class, model_type, metrics, optimizer):
    x_supp, y_supp, x_tgt, y_tgt = FewShotTrain.get_subsamples_sets(
        X, Y, index_batch, model_type=model_type, metrics=metrics,
        classes_per_set=classes_per_set, samples_per_class=samples_per_class,
        only_nk=True, change_classes_alias=True,
    )

    x_supp    = _to_variable(x_supp, dtype="float")
    y_supp    = _to_variable(y_supp, requires_grad=False, dtype="long")
    x_tgt     = _to_variable(x_tgt,  dtype="float")
    y_tgt     = _to_variable(y_tgt,  requires_grad=False, dtype="long").squeeze()
    y_supp_oh = _build_one_hot(y_supp, batch_size, classes_per_set)

    acc, c_loss, _, _ = _forward_pass(
        encoder, x_supp, y_supp_oh, x_tgt, y_tgt,
        model_type, samples_per_class, classes_per_set, batch_size, train=True,
        y_supp_labels=y_supp,
    )

    if model_type != "RelationNetwork":
        optimizer.zero_grad()
        c_loss.backward()
        optimizer.step()
        FewShotTrain.adjust_learning_rate(optimizer)

    return acc, c_loss, optimizer


def _sample_ft_batch(X, Y, batch_size):
    """Sample a random index batch for fine-tuning, with replacement if needed."""
    replace = X.shape[0] < batch_size
    return np.random.choice(X.shape[0], batch_size, replace=replace)


# ---------------------------------------------------------------------------
# Support-set construction helpers
# ---------------------------------------------------------------------------

def _init_target_arrays(X, batch_size, classes_per_set, model_type):
    c, h, w = X.shape[1], X.shape[2], X.shape[3]
    if model_type in ("PrototypicalNetwork", "RelationNetwork"):
        return (
            np.zeros((batch_size, classes_per_set, c, h, w), np.float32),
            np.zeros((batch_size, classes_per_set), np.int32),
        )
    # MatchingNetwork
    return (
        np.zeros((batch_size, c, h, w), np.float32),
        np.zeros((batch_size, 1), np.int32),
    )


def _build_support_from_train(X, Y, index_batch, batch_size, classes_per_set,
                               samples_per_class, model_type, metrics,
                               set, only_nk, target_x, target_y):
    max_c   = classes_per_set
    c, h, w = X.shape[1], X.shape[2], X.shape[3]
    support_set_x = np.zeros((batch_size, max_c * samples_per_class, c, h, w), np.float32)
    support_set_y = np.zeros((batch_size, max_c * samples_per_class), np.int32)

    augment_query = set == "train" and not only_nk and Constants["SIMCLR"] == "query"
    augment_supp  = set == "train" and not only_nk and Constants["SIMCLR"] == "supportSet"

    for b in range(batch_size):
        exc        = index_batch[b]
        supp_index = FewShotTrain.get_support_set_index(Y, exc, max_c, samples_per_class, only_nk=only_nk)

        if model_type == "PrototypicalNetwork":
            out_1, out_2 = get_multiple_querys(exc, Y, X, supp_index, set=set, only_nk=only_nk)
        elif model_type == "MatchingNetwork":
            out_1, out_2 = get_one_query(exc, Y, X, supp_index, set=set, only_nk=only_nk)
        elif model_type == "RelationNetwork":
            out_1, out_2 = get_multiple_querys(exc, Y, X, supp_index, set=set, only_nk=only_nk)
        else:
            raise NotImplementedError(model_type)

        supp_y = Y[supp_index]
        if augment_query:
            exc_y = Y[out_1]
            exc_x = out_2
        elif augment_supp:
            exc_y = out_2["query_class"]
            exc_x = out_2["query_obj"]
        else:
            exc_y = Y[out_1]
            exc_x = X[out_1]

        supp_x = X[supp_index]
        if Constants["SHUFFLE_SUPP_SET"]:
            p = np.random.permutation(len(supp_y))
            supp_x, supp_y = supp_x[p], supp_y[p]
        if augment_supp:
            supp_x = out_1

        support_set_x[b] = supp_x
        support_set_y[b] = supp_y
        target_x[b]      = exc_x
        target_y[b]      = exc_y

    return support_set_x, support_set_y, target_x, target_y


def _build_support_all_data(X, Y, batch_size, target_x, target_y):
    c, h, w = X.shape[1], X.shape[2], X.shape[3]
    n       = X.shape[0]
    support_set_x = np.zeros((batch_size, n - 1, c, h, w), np.float32)
    support_set_y = np.zeros((batch_size, n - 1), np.int32)

    for i in range(batch_size):
        idx = np.arange(n)
        if Constants["SHUFFLE_SUPP_SET"]:
            np.random.shuffle(idx)
        support_set_x[i] = X[idx[:-1]]
        support_set_y[i] = Y[idx[:-1]]
        target_x[i]      = X[idx[-1]]
        target_y[i]      = Y[idx[-1]]

    return support_set_x, support_set_y, target_x, target_y


def _remap_class_aliases(support_set_x, support_set_y, target_x, target_y, batch_size):
    """Re-index class labels to a contiguous range [0, N) per episode."""
    for b in range(batch_size):
        classes   = np.unique(support_set_y[b])
        mapping   = {c: i for i, c in enumerate(classes)}
        sub_sy    = support_set_y[b].copy()
        sub_ty    = target_y[b].copy()

        for old, new in mapping.items():
            sub_sy[support_set_y[b] == old] = new
            if len(sub_ty) > 1:
                sub_ty[target_y[b] == old] = new
            elif target_y[b] == old:
                sub_ty = np.array([new])

        p     = np.random.permutation(len(sub_sy))
        p_tgt = np.random.permutation(len(sub_ty))
        support_set_x[b] = support_set_x[b][p]
        support_set_y[b] = sub_sy[p]
        target_x[b]      = target_x[b][p_tgt]
        target_y[b]      = sub_ty[p_tgt]

    return support_set_x, support_set_y, target_x, target_y
