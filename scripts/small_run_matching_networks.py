import sys

sys.path.append("./")

from scripts.config_small import DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS
from train_fs import run_bootstrap as train

import wandb
from my_utils.constants import Constants




def run_approach_experiments(ds_pretrain_hparams: dict, ds_test_hparams: dict):
    for ds_name, ds_pretrain_config in ds_pretrain_hparams.items():

        for spc in Constants.SAMPLES_PER_CLASS:

            for model_type in Constants.MODEL_TYPE:
                if not Constants.DEACTIVATE_WANDB:
                    wandb.init(
                        project="small-run-matching-networks-SSL-symbols", 
                        group=Constants.GROUP_EXPERIMENT,
                        tags=[str(spc), ds_name, model_type],

                        name=Constants.Experiment + "_" + str(spc) + "_samples_" + ds_name + "_" + model_type,
                        config={"batch_size": DS_PRETRAIN_HPARAMS[ds_name]['batch_size'],
                                "epochs": DS_PRETRAIN_HPARAMS[ds_name]['epochs']} 
                    )

                train(
                    ds_name=ds_name,
                    samples_per_class=spc,
                    model_type=model_type,
                    epochs=ds_pretrain_config["epochs"],
                    episodes=ds_pretrain_config["episodes"],
                    batch_size=ds_pretrain_config["batch_size"],
                    num_runs=Constants.BOOTSTRAP_ITERS,
                )

                if not Constants.DEACTIVATE_WANDB:
                    wandb.finish()
    



if __name__ == "__main__":

    if not Constants.DEACTIVATE_WANDB:
        wandb.login()


    run_approach_experiments(DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS)
      
    
