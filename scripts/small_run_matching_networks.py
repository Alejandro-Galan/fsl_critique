import sys, wandb, os, importlib

sys.path.append("./")

from train_fs import run_bootstrap as train

import wandb, time
from my_utils import constants
importlib.reload(constants)
from my_utils.constants import Const_c

# Initialize reading the json constants file for each experiment
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants


def run_approach_experiments(exp_name: str = "", group_exp_name: str = ""):
    counter = 0
    for ds_name in Constants["TGT_DATASETS"]:
        if counter > 0:
            print("HERE IS THE ERROR")
            exit()
        counter += 1

        # Iteration done in "run_multiple_experiments" now #for spc in Constants["SAMPLES_PER_CLASS"]:
            # Iteration done in "run_multiple_experiments" now #for model_type in Constants["MODEL_TYPE"]:
        
        if not Constants["DEACTIVATE_WANDB"]:

            timestamp = str(time.time()).replace(".", "")
            exp_name_sufix = "_tgt_ds--" + ds_name + "--spc_" + str(Constants['SAMPLES_PER_CLASS']) + "--model--" + Constants['MODEL_TYPE']  
            full_name = "logs_csv/last_exec_" + group_exp_name + "/" + exp_name + exp_name_sufix + "_.csv"
            print("FULL NAME:", full_name)


            string_id_base, string_id_ft = Const_c.get_experiment_id(Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1) ## Assure the last iteration planned is done
            path_weights_base = "WEIGHTS/" + string_id_base 
            finetune_path = "WEIGHTS/" + string_id_ft + "--_trained_finetuned_model.pt" 



            if not Constants["OVERWRITE_LOGS"]:
                if os.path.exists(full_name):
                    if os.path.exists(path_weights_base + "_trained_model.pt"):
                        if not os.path.exists(finetune_path) or not Const_c.all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1):
                            print("Althogh existing, Model stored but finetuning not found. Not skipped")
                        # If exists and all boots iter were executed
                        else:
                            print("Skipping experiment already on logs", exp_name + exp_name_sufix)
                            continue
                    else:
                        print("Weights not found")

            run = wandb.init(
                project="small-run-matching-networks-SSL-symbols", 
                group=Constants["GROUP_EXPERIMENT"],
                tags=[str(Constants['SAMPLES_PER_CLASS']), ds_name, Constants['MODEL_TYPE']],

                name=Constants["Experiment"] + "_" + str(Constants['SAMPLES_PER_CLASS']) + "_samples_" + ds_name + "_" + Constants['MODEL_TYPE'],
                config={"batch_size": Constants["BATCH_SIZE"],
                        "epochs": Constants['EPOCHS']} 
            )

        train(
            ds_name=ds_name,
            samples_per_class=Constants['SAMPLES_PER_CLASS'],
            model_type=Constants['MODEL_TYPE'],
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
            if not 'bootstrap_iter' in history: # The whole iteration could not have existed
                continue
            config = run.config
            for key, value in config.items():
                history[key] = value
            # It was only on the first one
            history['bootstrap_iter'] = history['bootstrap_iter'].ffill()
            history['exp_name'] = exp_name.split("--")[0]
            new_columns = {"tgt_dataset": ds_name, "samples_per_class": Constants['SAMPLES_PER_CLASS'], "model_type": Constants['MODEL_TYPE'],
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
    main(str(sys.argv[1]), sys.argv[2])
      
    
