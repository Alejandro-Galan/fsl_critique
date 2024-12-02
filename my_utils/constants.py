### Constants

class Constants():

    ##### Main parameters
    #### Use Source dataset/s?
    ALL_DATASETS = True
    # Use all classes or only a fixed limit
    LIMIT_N_WAY = 5 #20 # None
    DATASETS_NAMES = ["b-59-850", "Egyptian", "TKH", "Greek"]
    TGT_DATASETS = {"Greek": {}, "b-59-850": {}, "Egyptian": {}, "TKH": {}}

    INPUT_SIZE = (28, 28) #(84, 84) #(40, 40) # Size the symbol images are resized
    MODEL_TYPE = ["MatchingNetwork"] #["PrototypicalNetwork"] #["MatchingNetwork"]

    EPOCHS = 1 #500 #20 #500 #150
    EPISODES = 1000 #1000 #1000
    BATCH_SIZE = 4 #1 #16
    SAMPLES_PER_CLASS = [1, 5, 10] #[10] #, 15, 20, 25, 30]
    
    BOOTSTRAP_ITERS = 1 # TODO DEBUG, put 10 
    lr = 1e-3 #1e-4 #1e-3

    FineTuning = True #True
    GROUP_EXPERIMENT = "testing_experiments" # "all_experiments" 

    USE_ORIGINAL_FIXED_SUPP_SET = True
    #####



    DEACTIVATE_WANDB = True


    SOTA_DATASETS = ["omniglot", "miniImageNet", "omniglot_SOTA_testSet", "miniImageNet_SOTA_testSet"]
    mini_dataset = False
    if mini_dataset:
        DATASETS_NAMES = ["Greek"] #, "miniImageNet"]
        # TGT_DATASETS = {"miniImageNet_SOTA_testSet": {}} #, "miniImageNet": {}}
        TGT_DATASETS = {"omniglot_SOTA_testSet": {}} #, "miniImageNet": {}}
        
        # DATASETS_NAMES = ["b-59-850"]
        # TGT_DATASETS = {"Egyptian": {}}


    

    #### Debug faster settings
    SMALL_TEST_SET = True # TODO Debug/Test porposes, faster training
    VALIDATION_SRC = True #False
    LIMIT_VALIDATION_SRC = 1

    #### Training conf
    PATIENCE = 10

    SHUFFLE_SUPP_SET = False

    epochsFineTuning = 20
    if mini_dataset:
        epochsFineTuning = 3 #10
    lrFineTuning = 1e-4



    #### Experiments:
    Experiment = "FineTuning" if FineTuning else "NoFT"

    if LIMIT_N_WAY:
        Experiment += f"_{LIMIT_N_WAY}way"
    else:
        Experiment += "_AllWay"

    Experiment += "_InputS_" + str(INPUT_SIZE[0]) + "x" + str(INPUT_SIZE[1])


    #### Prototypical
    n_query = 1



    greek_comparative = False
    if greek_comparative:
        Experiment = "GreekComp"

        DATASETS_NAMES = ["TKH"]
        TGT_DATASETS = {"Greek": {}}

        MODEL_TYPE = ["PrototypicalNetwork", "MatchingNetwork"]
        SAMPLES_PER_CLASS = [1, 5, 10]
        EPISODES = 100 #250 #1000
        epochsFineTuning = 5 #5   

        VALIDATION_SRC = True #True
        LIMIT_VALIDATION_SRC = 1 #5
        SMALL_TEST_SET = True

        FineTuning = True
        BATCH_SIZE = 4
