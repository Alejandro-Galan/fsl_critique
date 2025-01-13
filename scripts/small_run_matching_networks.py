import sys, wandb, os

sys.path.append("./")

from train_fs import run_bootstrap as train

import wandb, time
from my_utils.constants import Const_c
# Initialize reading the json constants file for each experiment
exp = int(sys.argv[1])
Constants_c = Const_c(exp)
Constants = Constants_c.Constants



def run_approach_experiments(exp_name: str = "", group_exp_name: str = ""):
    counter = 0
    for ds_name in Constants["TGT_DATASETS"]:
        if counter > 0:
            print("HERE IS THE ERROR")
            exit()
        counter += 1
        for spc in Constants["SAMPLES_PER_CLASS"]:

            for model_type in Constants["MODEL_TYPE"]:
                if not Constants["DEACTIVATE_WANDB"]:

                    timestamp = str(time.time()).replace(".", "")
                    exp_name_sufix = "_tgt_ds--" + ds_name + "--spc_" + str(spc) + "--model--" + model_type  
                    full_name = "logs_csv/last_exec_" + group_exp_name + "/" + exp_name + exp_name_sufix + "_.csv"
                    print("FULL NAME:", full_name)

                    if not Constants["OVERWRITE_LOGS"] and os.path.exists(full_name):
                        print("Skipping experiment already on logs", exp_name + exp_name_sufix)
                        continue

                    run = wandb.init(
                        project="small-run-matching-networks-SSL-symbols", 
                        group=Constants["GROUP_EXPERIMENT"],
                        tags=[str(spc), ds_name, model_type],

                        name=Constants["Experiment"] + "_" + str(spc) + "_samples_" + ds_name + "_" + model_type,
                        config={"batch_size": Constants["BATCH_SIZE"],
                                "epochs": Constants['EPOCHS']} 
                    )

                train(
                    ds_name=ds_name,
                    samples_per_class=spc,
                    model_type=model_type,
                    epochs=Constants['EPOCHS'],
                    episodes=Constants["EPISODES"],
                    batch_size=Constants["BATCH_SIZE"],
                    num_runs=Constants["BOOTSTRAP_ITERS"],
                )

                if not Constants["DEACTIVATE_WANDB"]:
                    wandb.finish()
                    api = wandb.Api()
                    run = api.run("grifa/small-run-matching-networks-SSL-symbols/" + run.id)

                    history = run.history()  

                    config = run.config
                    for key, value in config.items():
                        history[key] = value
                    # It was only on the first one
                    history['bootstrap_iter'] = history['bootstrap_iter'].ffill()
                    history['exp_name'] = exp_name.split("--")[0]
                    new_columns = {"tgt_dataset": ds_name, "samples_per_class": spc, "model_type": model_type,
                                   "src_datasets": "__".join(Constants["DATASETS_NAMES"])}
                    for key, value in new_columns.items():
                        history[key] = value
                    
                    history.to_csv(full_name, index=False)
                    # history.to_csv("logs_csv/last_exec/" + exp_name + exp_name_sufix + "__" + timestamp + "_.csv", index=False)

    

def main(exp_name, group_exp_name):
    
    if not Constants["DEACTIVATE_WANDB"]:
        wandb.login()


    run_approach_experiments(exp_name=exp_name, group_exp_name=group_exp_name)
    

if __name__ == "__main__":
    main()
      
    
