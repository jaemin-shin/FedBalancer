# FedBalancer

- A systematic federated learning with sample selection to optimize the time-to-accuracy performance
- Includes an adaptive deadline control scheme that predicts the optimal deadline for each round with varying client data

This repository contains the code and experiments for the paper:

>  Conditionally accepted to [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
>
> [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

## System Requirements

The system is written and evaluated based on  ```Python 3.6.9```, with ```tensorflow 1.14.0```.

Please use ```virtualenv``` to run a self-contained setup:

```
$ git clone https://github.com/jaemin-shin/FedBalancer.git
$ virtualenv ./fedbalancer-venv -p python3.6
$ source ./fedbalancer-venv/bin/activate
$ pip install -r requirements.txt
```

## Disclaimer (IMPORTANT)

We evaluated based on five datasets: ```FEMNIST```, ```Celeba```, ```Reddit```, ```Shakespeare```, ```UCI-HAR```.

Currently, this repository only supports experiments with ```UCI-HAR```.

Handling other datasets will be added soon.

## How to run the experiments

### Running the main experiment of the paper in Section 4.2 and 4.3 one by one

1. Go to directory of dataset `data/har` for instructions on generating the benchmark dataset
2. Run (please refer to the list of config files in ```configs/har```, it contains all the config files for the experiment)
```
$ cd models/

# FedAvg + 1T experiment in Section 4.2 and 4.3 with random seed 0
$ python main.py --config=configs/har/har_fedavg_1T_seed0.cfg

# FedBalancer experiment in Section 4.2 and 4.3 with random seed 0
$ python main.py --config=configs/har/har_fedbalancer_seed0.cfg
```

### Running the main experiment of the paper in Section 4.2 and 4.3 at ONCE

1. Go to directory of dataset `data/har` for instructions on generating the benchmark dataset
2. Configure your python file
3. Run (IMPORTANT NOTE: before you run the experiment, please refer to the python file that runs all the experiments in `paper_experiments`. You need to assign which GPU you will assign at each experiment, and you may need to run experiments partially as running all the experiments may exceed the VRAM of your GPU)
```
$ cd models/

# Run baseline experiments
$ python ../paper_experiments/experiment_run_har_baselines.py

# Run fedbalancer experiments
$ python ../paper_experiments/experiment_run_har_fedbalancer.py
```

<h3 id="config">Config File</h3>
To simplify the command line arguments, we move most of the parameters to a <span id="jump">config file</span>. Below is a detailed example.

```bash
## whether to consider heterogeneity
behav_hete False # bool, whether to simulate state(behavior) heterogeneity
hard_hete False # bool, whether to simulate hardware heterogeneity, which contains differential on-device training time and network speed


## no training mode to tune system configurations
no_training False # bool, whether to run in no_training mode, skip training process if True


## ML related configurations
dataset femnist # dataset to use
model cnn # file that defines the DNN model
learning_rate 0.01 # learning-rate of DNN
batch_size 10 # batch-size for training 


## system configurations, refer to https://arxiv.org/abs/1812.02903 for more details
num_rounds 500 # number of FL rounds to run
clients_per_round 100 # expected clients in each round
min_selected 60 # min selected clients number in each round, fail if not satisfied
max_sample 340 #  max number of samples to use in each selected client
eval_every 5 # evaluate every # rounds, -1 for not evaluate
num_epochs 5 # number of training epochs (E) for each client in each round
seed 0 # basic random seed
round_ddl 270 0 # μ and σ for deadline, which follows a normal distribution
update_frac 0.8  # min update fraction in each round, round fails when fraction of clients that successfully upload their is not less than "update_frac"
max_client_num -1 # max number of clients in the simulation process, -1 for infinite


### ----- NOTE! below are advanced configurations. 
### ----- Strongly recommend: specify these configurations only after reading the source code. 
### ----- Configuration items of [aggregate_algorithm, fedprox*, structure_k, qffl*] are mutually-exclusive 

## basic algorithm
aggregate_algorithm SucFedAvg # choose in [SucFedAvg, FedAvg], please refer to models/server.py for more details. In the configuration file, SucFedAvg refers to the "FedAvg" algorithm described in https://arxiv.org/pdf/1602.05629.pdf

## compression algorithm
# compress_algo grad_drop # gradiant compress algorithm, choose in [grad_drop, sign_sgd], not use if commented
# structure_k 100
## the k for structured update, not use if commented, please refer to the arxiv for more 

## advanced aggregation algorithms
# fedprox True # whether to apply fedprox and params needed, please refer to the sysml'20 (https://arxiv.org/pdf/1812.06127.pdf) for more details
# fedprox_mu 0.5
# fedprox_active_frac 0.8

# qffl True # whether to apply qffl(q-fedavg) and params needed, please refer to the ICLR'20 (https://arxiv.org/pdf/1905.10497.pdf) for more
# qffl_q 5
```

## How to parse the results after the experiment


## Benchmark Datasets

#### FEMNIST

- **Overview:** Image Dataset
- **Details:** 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with option to make them all 128 by 128 pixels), 3500 users
- **Task:** Image Classification



#### Celeba

- **Overview:** Image Dataset based on the [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **Details:** 9343 users (we exclude celebrities with less than 5 images)
- **Task:** Image Classification (Smiling vs. Not smiling)



#### Reddit

- **Overview:** We preprocess the Reddit data released by [pushshift.io](https://files.pushshift.io/reddit/) corresponding to December 2017.
- **Details:** 1,660,820 users with a total of 56,587,343 comments. 
- **Task:** Next-word Prediction.


#### Shakespeare


#### UCI-HAR



## Notes

- This repository is based on [Chengxu Yang et al.'s work, FLASH](https://github.com/PKU-Chengxu/FLASH), which is a heterogeneity-aware benchmarking framework for FL based on [Sebastian Calas et al.'s work, LEAF](https://leaf.cmu.edu/).
