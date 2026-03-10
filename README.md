<p align="center">
  <a href="https://praig.ua.es/"><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Few-Shot Symbol Classification via Self-Supervised Learning and Nearest Neighbor</h1>

<h4 align="center">Full text available <a href="https://doi.org/10.1016/j.patrec.2023.01.014" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>


## About

## Load data example:

If you want to replicate just the load of the data, check "calculate_distances" function in "models/FewShotModels.py"

In that function, all the datasets are loaded without partition sets. 

```
  ## Load only the training src dataset
  pretrained_sources = True if Constants["NoSrcDataset"] else pretrained_sources

  # If no validation src paramenter, Xval is empty
  data_dict = load_supervised_data(ds_name=ds_name, min_occurence=50, all_datasets=False, pretrained_sources=pretrained_sources, boots_iter=0)
  
  
  XTrain, YTrain = data_dict["X_tgt"], data_dict["Y_tgt"]
```
"XTrain" and "YTrain" correspond to the whole loaded dataset. 


## How To Use

All the needed packages can be found at "docker/Dockerfile"


To execute an experiment, please execute this line:

```
python3 ./scripts/auto_paralel_exps.sh <experiment_number> <simultaneous_executions>
```

##### Correspondences between experiments and its id
| Experiment | Paper name                                    |   |
|------------|-----------------------------------------------|---|
| 1          | Out of domain pre-trained, Supervised         |   |
| 3          | No pre-train use (Strict Baseline)            |   |
| 4          | In domain pre-train, Supervised               |   |
| 10         | Clustering out of domain, Unsupervised        |   |
| 13         | Data Augmentation out of domain, Unsupervised |   |

There are also a few relevant scripts. 

#### Extract the logs into simpler tables:

```
python3 logs_csv/filter_logs_csv.py
```

#### Compare datasets by predictions and generated embeddings:

```
python3 scripts/complementary_comp/main_complementary_comparison_methods.py
```

## Reproducibility
If you want to try our code and check that your results seem correct, compare to the following table for the supervised ood-pt experiment.

|  Target Dataset, Source Dataset, N-way, K-shot  |   with IDA |   no IDA |
|                            |   ft_eval_acc |   before_ft_eval_acc |
|:---------------------------|--------------:|---------------------:|
| ('Greek', 'CAP.', 5, 1)    |      0.708311 |             0.666275 |
| ('Greek', 'CAP.', 5, 5)    |      0.897089 |             0.859725 |
| ('Greek', 'CAP.', 5, 10)   |      0.916079 |             0.888032 |
| ('Greek', 'CAP.', 20, 1)   |      0.570962 |             0.487583 |
| ('Greek', 'CAP.', 20, 5)   |      0.805387 |             0.711182 |
| ('Greek', 'CAP.', 20, 10)  |      0.841094 |             0.76587  |
| ('Greek', 'CIF.', 5, 1)    |      0.727379 |             0.693075 |
| ('Greek', 'CIF.', 5, 5)    |      0.870487 |             0.761471 |
| ('Greek', 'CIF.', 5, 10)   |      0.909753 |             0.828277 |
| ('Greek', 'CIF.', 20, 1)   |      0.499615 |             0.404211 |
| ('Greek', 'CIF.', 20, 5)   |      0.798306 |             0.604976 |
| ('Greek', 'CIF.', 20, 10)  |      0.846634 |             0.72314  |
| ('Greek', 'Egypt.', 5, 1)  |      0.749464 |             0.729729 |
| ('Greek', 'Egypt.', 5, 5)  |      0.895914 |             0.875775 |
| ('Greek', 'Egypt.', 5, 10) |      0.925757 |             0.914675 |
| ('Greek', 'Omni.', 5, 1)   |      0.715863 |             0.733225 |
| ('Greek', 'Omni.', 5, 5)   |      0.873262 |             0.829213 |
| ('Greek', 'Omni.', 5, 10)  |      0.91117  |             0.87255  |
| ('Greek', 'Omni.', 20, 1)  |      0.52964  |             0.388698 |
| ('Greek', 'Omni.', 20, 5)  |      0.805355 |             0.623918 |
| ('Greek', 'Omni.', 20, 10) |      0.848142 |             0.693969 |
| ('Greek', 'TKH', 5, 1)     |      0.722943 |             0.714182 |
| ('Greek', 'TKH', 5, 5)     |      0.904182 |             0.896889 |
| ('Greek', 'TKH', 5, 10)    |      0.908575 |             0.889307 |
| ('Greek', 'TKH', 20, 1)    |      0.618159 |             0.592111 |
| ('Greek', 'TKH', 20, 5)    |      0.821157 |             0.783617 |
| ('Greek', 'TKH', 20, 10)   |      0.850301 |             0.810196 |
| ('Greek', 'mINet', 5, 1)   |      0.680679 |             0.607483 |
| ('Greek', 'mINet', 5, 5)   |      0.872879 |             0.782667 |
| ('Greek', 'mINet', 5, 10)  |      0.910207 |             0.799383 |
| ('Greek', 'mINet', 20, 1)  |      0.46506  |             0.326046 |
| ('Greek', 'mINet', 20, 5)  |      0.782393 |             0.561639 |
| ('Greek', 'mINet', 20, 10) |      0.840324 |             0.709633 |
| ('Greek', 'oMNIST', 5, 1)  |      0.57418  |             0.404997 |
| ('Greek', 'oMNIST', 5, 5)  |      0.865617 |             0.756603 |
| ('Greek', 'oMNIST', 5, 10) |      0.896217 |             0.767237 |




## Citations


## License
This work is under a [MIT](LICENSE) license.
