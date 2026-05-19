import importlib
import os
import sys

sys.path.append("./")

from train_fs import run_bootstrap as train
from utils import constants
from utils.constants import Const_c

importlib.reload(constants)

# Initialize reading the json constants file for each experiment.
exp = str(sys.argv[2])
full_name = str(sys.argv[3])
Constants_c = Const_c(exp, full_name)
Constants = Constants_c.Constants


def run_approach_experiments(exp_name: str = "", group_exp_name: str = ""):
    for index, ds_name in enumerate(Constants["TGT_DATASETS"]):
        if index > 0:
            raise RuntimeError("Only one target dataset is expected per generated experiment.")

        full_name = Const_c.get_logs_csv_path(exp, exp_name, ds_name, Constants)
        string_id_base, _ = Const_c.get_experiment_id(
            Constants, boots_iter=Constants["BOOTSTRAP_ITERS"] - 1
        )

        if not Constants["OVERWRITE_LOGS"] and os.path.exists(full_name):
            if Const_c.all_boots_iter_done(
                exp,
                exp_name,
                ds_name,
                Constants,
                boots_iter=Constants["BOOTSTRAP_ITERS"],
                string_id_base=string_id_base,
            ):
                print("Skipping whole experiment: all bootstrap iterations already exist in logs")
                continue

        accumulated_history = train(
            ds_name=ds_name,
            samples_per_class=Constants["SAMPLES_PER_CLASS"],
            model_type=Constants["MODEL_TYPE"],
            epochs=Constants["EPOCHS"],
            episodes=Constants["EPISODES"],
            batch_size=Constants["BATCH_SIZE"],
            num_runs=Constants["BOOTSTRAP_ITERS"],
        )

        if accumulated_history is not None and not accumulated_history.empty:
            accumulated_history["exp_name"] = exp_name.split("--")[0]
            accumulated_history["group_exp_name"] = group_exp_name
            os.makedirs(os.path.dirname(full_name), exist_ok=True)
            accumulated_history.to_csv(full_name, index=False)


def main(exp_name, group_exp_name):
    run_approach_experiments(exp_name=exp_name, group_exp_name=group_exp_name)


if __name__ == "__main__":
    main(str(sys.argv[1]), sys.argv[2])
