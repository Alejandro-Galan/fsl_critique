import sys, wandb, os, importlib

sys.path.append("./")

from train_fs import run_bootstrap as train

import wandb, time, math
from utils import constants
importlib.reload(constants)
from utils.constants import Const_c
import pandas as pd




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
            full_name = Const_c.get_logs_csv_path(exp, exp_name, ds_name, Constants)
            print("FULL NAME:", full_name)


            string_id_base, string_id_ft = Const_c.get_experiment_id(Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1) ## Assure the last iteration planned is done
            path_weights_base = "WEIGHTS/" + string_id_base 
            finetune_path = Const_c.get_id_extensions(PARAMS=Constants, prev_str="WEIGHTS/" + string_id_ft)



            if not Constants["OVERWRITE_LOGS"]:
                if os.path.exists(full_name):
                    if Const_c.all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter=Constants["BOOTSTRAP_ITERS"], string_id_base=string_id_base):
                        print("Skipping whole exp, all", Constants["BOOTSTRAP_ITERS"], "boots iters already done on logs")
                        continue
                    
                    # ## Only the logs matter
                    # if os.path.exists(path_weights_base + "_trained_model.pt"):
                    #     if not os.path.exists(finetune_path) or not Const_c.all_boots_iter_done(exp, exp_name, ds_name, Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1):
                    #         print("Althogh existing, Model stored but finetuning not found. Not skipped")
                    #     # If exists and all boots iter were executed
                    #     else:
                    #         print("Skipping experiment already on logs", exp_name)
                    #         continue
                    # else:
                    #     print("Weights not found")


            # if "exp6" in exp:
            #     breakpoint()
            ### May be best to run it for each run_bootstrap
            # run = wandb.init(
            #     project="small-run-matching-networks-SSL-symbols", 
            #     group=Constants["GROUP_EXPERIMENT"],
            #     tags=[str(Constants['SAMPLES_PER_CLASS']), ds_name, Constants['MODEL_TYPE']],

            #     name=Constants["Experiment"] + "_" + str(Constants['SAMPLES_PER_CLASS']) + "_samples_" + ds_name + "_" + Constants['MODEL_TYPE'],
            #     config={"batch_size": Constants["BATCH_SIZE"],
            #             "epochs": Constants['EPOCHS']} 
            # )

        accumulated_history = train(
            ds_name=ds_name,
            samples_per_class=Constants['SAMPLES_PER_CLASS'],
            model_type=Constants['MODEL_TYPE'],
            epochs=Constants['EPOCHS'],
            episodes=Constants["EPISODES"],
            batch_size=Constants["BATCH_SIZE"],
            num_runs=Constants["BOOTSTRAP_ITERS"],
        )

        if not Constants["DEACTIVATE_WANDB"]:
            
            ## In case of missing bootstrap_iters
            # size_offset = math.ceil(len(accumulated_history['bootstrap_iter']) / Constants["BOOTSTRAP_ITERS"])

            # last_real_value = float('nan')
            # # Order hystory by _step
            # copy_boots_iter = accumulated_history['bootstrap_iter'].copy()
            # for i, val in enumerate(accumulated_history['bootstrap_iter']):
            #     if pd.isna(val):
            #         if pd.isna(last_real_value):
            #             copy_boots_iter[i] = i // size_offset
            #         else:
            #             copy_boots_iter[i] = last_real_value
            #     else:
            #         last_real_value = val
            # accumulated_history['bootstrap_iter'] = copy_boots_iter
            
            # history['bootstrap_iter'] = [
            #     i // size_offset if pd.isna(val) else val
            #     for i, val in enumerate(history['bootstrap_iter'])
            # ]
            # print("AFTER History to", history['bootstrap_iter'])

            

            ####### No fill in that case
            ###history['bootstrap_iter'] = history['bootstrap_iter'].ffill()


            
            accumulated_history['exp_name'] = exp_name.split("--")[0]
            new_columns = {"tgt_dataset": ds_name, "samples_per_class": Constants['SAMPLES_PER_CLASS'], "model_type": Constants['MODEL_TYPE'],
                            "src_datasets": "__".join(Constants["DATASETS_NAMES"])}
            for key, value in new_columns.items():
                accumulated_history[key] = value

            os.makedirs(os.path.dirname(full_name), exist_ok=True)
            accumulated_history.to_csv(full_name, index=False)

    
def main(exp_name, group_exp_name):
    
    if not Constants["DEACTIVATE_WANDB"]:
        wandb.login()


    run_approach_experiments(exp_name=exp_name, group_exp_name=group_exp_name)
    

if __name__ == "__main__":
    main(str(sys.argv[1]), sys.argv[2])
      
    
