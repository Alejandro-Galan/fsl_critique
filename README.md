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


To execute one experiment, please execute this line:

```
./scripts/execute_experiment.sh <experiment_number>
```

The hiperparameters can be changed in that script and in 
```
hyperparameters_experiments.json
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

| Target Dataset, Source Dataset, N-way, K-shot |   ('with IDA', 'MN.') |   ('with IDA', 'PN.') |   ('with IDA', 'RN') |   ('no IDA', 'MN') |   ('no IDA', 'PN') |   ('no IDA', 'RN') |
|:---------------------------|-------------------------------------:|-----------------------------------------:|-------------------------------------:|--------------------------------------------:|------------------------------------------------:|--------------------------------------------:|
| ('Greek', 'CAP.', 5, 1)    |                             0.62335  |                                 0.73532  |                             0.78075  |                                    0.55935  |                                        0.70845  |                                    0.747212 |
| ('Greek', 'CAP.', 5, 5)    |                             0.89225  |                                 0.9092   |                             0.888    |                                    0.8525   |                                        0.87122  |                                    0.854387 |
| ('Greek', 'CAP.', 5, 10)   |                             0.93185  |                                 0.9308   |                             0.877962 |                                    0.9148   |                                        0.90695  |                                    0.830925 |
| ('Greek', 'CAP.', 20, 1)   |                             0.52445  |                                 0.574798 |                             0.624306 |                                    0.39275  |                                        0.518155 |                                    0.567909 |
| ('Greek', 'CAP.', 20, 5)   |                             0.7992   |                                 0.814008 |                             0.802344 |                                    0.67385  |                                        0.759123 |                                    0.697922 |
| ('Greek', 'CAP.', 20, 10)  |                             0.8361   |                                 0.849375 |                             0.839056 |                                    0.76185  |                                        0.810091 |                                    0.726675 |
| ('Greek', 'CIF.', 5, 1)    |                             0.675    |                                 0.76975  |                             0.737387 |                                    0.612    |                                        0.752513 |                                    0.714712 |
| ('Greek', 'CIF.', 5, 5)    |                             0.881063 |                                 0.862975 |                             0.867425 |                                    0.83125  |                                        0.672387 |                                    0.780775 |
| ('Greek', 'CIF.', 5, 10)   |                             0.90905  |                                 0.92675  |                             0.89346  |                                    0.71315  |                                        0.90416  |                                    0.86752  |
| ('Greek', 'CIF.', 20, 1)   |                             0.44725  |                                 0.481045 |                             0.57055  |                                    0.1937   |                                        0.44462  |                                    0.574313 |
| ('Greek', 'CIF.', 20, 5)   |                             0.76975  |                                 0.809235 |                             0.815933 |                                    0.32705  |                                        0.723305 |                                    0.764573 |
| ('Greek', 'CIF.', 20, 10)  |                             0.8191   |                                 0.864188 |                             0.856615 |                                    0.5748   |                                        0.805725 |                                    0.788895 |
| ('Greek', 'Egypt.', 5, 1)  |                             0.7004   |                                 0.73952  |                             0.823225 |                                    0.6659   |                                        0.73031  |                                    0.808787 |
| ('Greek', 'Egypt.', 5, 5)  |                             0.8878   |                                 0.90417  |                             0.895737 |                                    0.86745  |                                        0.89952  |                                    0.8565   |
| ('Greek', 'Egypt.', 5, 10) |                             0.93425  |                                 0.92535  |                             0.91565  |                                    0.91915  |                                        0.92504  |                                    0.896125 |
| ('Greek', 'Omni.', 5, 1)   |                             0.694813 |                                 0.765013 |                             0.687763 |                                    0.73825  |                                        0.764763 |                                    0.696662 |
| ('Greek', 'Omni.', 5, 5)   |                             0.88725  |                                 0.868787 |                             0.86375  |                                    0.890375 |                                        0.769513 |                                    0.82775  |
| ('Greek', 'Omni.', 5, 10)  |                             0.91675  |                                 0.92282  |                             0.89394  |                                    0.8457   |                                        0.8842   |                                    0.88775  |
| ('Greek', 'Omni.', 20, 1)  |                             0.495688 |                                 0.563778 |                             0.529453 |                                    0.2345   |                                        0.516444 |                                    0.41515  |
| ('Greek', 'Omni.', 20, 5)  |                             0.79525  |                                 0.803416 |                             0.8174   |                                    0.481875 |                                        0.736962 |                                    0.652916 |
| ('Greek', 'Omni.', 20, 10) |                             0.83555  |                                 0.8508   |                             0.858075 |                                    0.58835  |                                        0.747197 |                                    0.74636  |
| ('Greek', 'TKH', 5, 1)     |                             0.6629   |                                 0.76707  |                             0.742838 |                                    0.65235  |                                        0.76536  |                                    0.7275   |
| ('Greek', 'TKH', 5, 5)     |                             0.8854   |                                 0.92114  |                             0.906462 |                                    0.8813   |                                        0.91558  |                                    0.893012 |
| ('Greek', 'TKH', 5, 10)    |                             0.9277   |                                 0.93249  |                             0.854775 |                                    0.9117   |                                        0.92769  |                                    0.813338 |
| ('Greek', 'TKH', 20, 1)    |                             0.556    |                                 0.657585 |                             0.646575 |                                    0.523    |                                        0.627942 |                                    0.633709 |
| ('Greek', 'TKH', 20, 5)    |                             0.81075  |                                 0.829242 |                             0.824059 |                                    0.7563   |                                        0.81791  |                                    0.774897 |
| ('Greek', 'TKH', 20, 10)   |                             0.84685  |                                 0.862022 |                             0.842894 |                                    0.81625  |                                        0.847444 |                                    0.765381 |
| ('Greek', 'mINet', 5, 1)   |                             0.679937 |                                 0.742    |                             0.6201   |                                    0.62625  |                                        0.708638 |                                    0.487563 |
| ('Greek', 'mINet', 5, 5)   |                             0.887375 |                                 0.865363 |                             0.8659   |                                    0.849063 |                                        0.7203   |                                    0.778637 |
| ('Greek', 'mINet', 5, 10)  |                             0.90155  |                                 0.92894  |                             0.90013  |                                    0.7401   |                                        0.91346  |                                    0.74459  |
| ('Greek', 'mINet', 20, 1)  |                             0.4471   |                                 0.43775  |                             0.51033  |                                    0.17505  |                                        0.314288 |                                    0.4888   |
| ('Greek', 'mINet', 20, 5)  |                             0.7625   |                                 0.777877 |                             0.806803 |                                    0.42495  |                                        0.549283 |                                    0.710685 |
| ('Greek', 'mINet', 20, 10) |                             0.8162   |                                 0.862275 |                             0.842498 |                                    0.58925  |                                        0.789238 |                                    0.75041  |
| ('Greek', 'oMNIST', 5, 1)  |                             0.52165  |                                 0.68189  |                             0.519    |                                    0.4553   |                                        0.56977  |                                    0.18992  |
| ('Greek', 'oMNIST', 5, 5)  |                             0.863    |                                 0.89312  |                             0.84073  |                                    0.75035  |                                        0.79988  |                                    0.71958  |
| ('Greek', 'oMNIST', 5, 10) |                             0.8972   |                                 0.91202  |                             0.87943  |                                    0.67265  |                                        0.84817  |                                    0.78089  |




## Citations


## License
This work is under a [MIT](LICENSE) license.
