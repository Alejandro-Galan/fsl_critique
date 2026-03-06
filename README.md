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

|  Dataset, N-way, K-shot  |   wit IDA |   no IDA |
|:------------------|--------------:|---------------------:|
| ('CAP.', 5, 1)    |     0.58575   |            0.52608   |
| ('CAP.', 5, 5)    |     0.849527  |            0.692157  |
| ('CAP.', 5, 10)   |     0.915925  |            0.765815  |
| ('Greek', 5, 1)   |     0.694944  |            0.645408  |
| ('Greek', 5, 5)   |     0.883273  |            0.824545  |
| ('Greek', 5, 10)  |     0.9111    |            0.854696  |
| ('Greek', 20, 1)  |     0.54979   |            0.459218  |
| ('Greek', 20, 5)  |     0.805293  |            0.675706  |
| ('Greek', 20, 10) |     0.845389  |            0.751328  |
| ('Omni.', 5, 1)   |     0.656689  |            0.621083  |
| ('Omni.', 5, 5)   |     0.884959  |            0.842189  |
| ('Omni.', 5, 10)  |     0.956662  |            0.900265  |
| ('Omni.', 20, 1)  |     0.483369  |            0.444266  |
| ('Omni.', 20, 5)  |     0.81198   |            0.709403  |
| ('Omni.', 20, 10) |     0.934959  |            0.824712  |




## Citations


## License
This work is under a [MIT](LICENSE) license.
