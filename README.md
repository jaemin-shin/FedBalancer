# FedBalancer

- A systematic federated learning with sample selection to optimize the time-to-accuracy performance
- Includes an adaptive deadline control scheme that predicts the optimal deadline for each round with varying client data

This repository contains the code of the simulation experiments (Section 4.1~4.5) of the paper:

> [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
>
> [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

For the testbed experiment on Android devices in our paper (Section 4.6), please refer to the following repository: [flower-FedBalancer-testbed](https://github.com/jaemin-shin/flower-FedBalancer-testbed).

## System Requirements

The system is written and evaluated based on  ```Python 3.9.12```, with ```PyTorch 1.10.1+cu111```, running on ```Ubuntu 20.04``` server with eight ```NVIDIA TITAN RTX``` GPUs.

As an alternative setup, you can use general Ubuntu servers with NVIDIA GPUs.

Note that you could also run our system and experiments on CPUs, which could be slower.

The experimental results on different setup and different GPUs may differ, but the results will derive same conclusions that we stated in our paper.

## System Installation

Please use ```conda``` to run a self-contained setup:

### Install miniconda on your system

```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ conda update conda
$ conda update --all
```

### Setup a conda env and install pip inside conda env

```
$ git clone https://github.com/jaemin-shin/FedBalancer.git
$ cd FedBalancer
$ conda env create -f environment.yml --name fb-torch-conda
$ conda activate fb-torch-conda
$ conda install pip
```

### Deactivate and re-activate the conda env to activate pip inside conda env
```
$ while [ ! -z $CONDA_PREFIX ]; do conda deactivate; done
$ conda activate fb-torch-conda
```

### Install other dependencies

```
$ pip install torch
$ pip install timeout_decorator
$ pip install matplotlib
```

### Installing GLIBCXX_3.4.29 (only on Ubuntu)

```
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
$ sudo apt-get update
$ sudo apt-get install gcc-4.9
$ sudo apt-get upgrade libstdc++6
$ sudo apt-get dist-upgrade
```

## Datasets

This repository contains experiment on a widely used dataset for the FL benchmark, ```FEMNIST```.

## How to Run the Experiments

### Running the main experiment of the paper in Section 4.2 and 4.3 one by one

1. Go to directory of FEMNIST dataset in `data/femnist` for instructions on generating the benchmark dataset
2. Run (please refer to the list of config files in the subdirectory in ```configs/femnist/```, it contains all the config files for the experiment)
```
$ cd models/

# FedAvg + 1T experiment in Section 4.2 and 4.3 with random seed 0

$ python main.py --config=configs/femnist/femnist_fedavg_1T_seed0.cfg

# FedBalancer experiment in Section 4.2 and 4.3 with random seed 0

$ python main.py --config=configs/femnist/femnist_fedbalancer_seed0.cfg
```

Add ```CUDA_VISIBLE_DEVICES={GPU_ID} ``` before the command to run the experiment on the specific GPUs. If you set ```GPU_ID``` as ```-1```, the experiment runs on cpus.

Note that you could change the parameters of fedbalancer config file to test another parameters. 

<h3 id="config">Config File</h3>
To simplify the command line arguments, we move most of the parameters to a <span id="jump">config file</span>. Below is a detailed example.

```bash
## whether to consider heterogeneity
behav_hete False # bool, whether to simulate state(behavior) heterogeneity -> fixed to False in our experiments
hard_hete True # bool, whether to simulate hardware heterogeneity, which contains differential on-device training time and network speed -> fixed to True in our experiments

## ML related configurations
dataset femnist # dataset to use
model cnn # file that defines the model (in this example, it is a CNN model defined under models/femnist/)
learning_rate 0.0003 # learning-rate of LR
batch_size 10 # batch-size for training 

## system configurations, refer to https://arxiv.org/abs/1812.02903 for more details
num_rounds 6000 # number of FL rounds to run
clients_per_round 5 # expected clients in each round
min_selected 1 # min selected clients number in each round, fail if not satisfied -> fixed to 1 in our experiments
max_sample 2147483647 #  max number of samples to use in each selected client -> fixed to large int in our experiments
eval_every 5 # evaluate every # rounds, -1 for not evaluate
num_epochs 5 # number of training epochs (E) for each client in each round
seed 0 # basic random seed
update_frac 0.01  # min update fraction in each round, round fails when fraction of clients that successfully upload their is not less than "update_frac" -> fixed to 0.01 in our experiments

aggregate_algorithm SucFedAvg # fixed to SucFedAvg in our experiments, SucFedAvg refers to the "FedAvg" algorithm described in https://arxiv.org/pdf/1602.05629.pdf 

### ----- NOTE! below are configurations for our baselines.

## parameters to configure deadlines
# ddl_baseline_fixed True # True if experiment uses fixed deadline
# ddl_baseline_fixed_value_multiplied_at_mean 1.0 # In our experiments, the deadline is sampled as mean of clients' round completion times. This parameter indicates the multiplication factor at the sampled deadline. If 1.0, the mean is used, and this becomes the 1T experiment in our paper. If 2.0, 2.0 times mean is used, and this becomes the 2T experiment in our paper.
# ddl_baseline_smartpc False # True if using SmartPC (SPC) or Wait-For-All (WFA) as a deadline configuration method.
# ddl_baseline_smartpc_percentage 0.8 # Specifies the portion of clients that will successfully send the result at a round. If 0.8, this indicates SmartPC (SPC) experiment in our paper. If 1.0, the round waits for every clients to end, and this is Wait-For-All (WFA) experiment in our paper.

## fedprox parameters
# fedprox True # whether to apply fedprox and params needed, please refer to T. Li et al., MLSys'20 (https://arxiv.org/pdf/1812.06127.pdf) for more details
# fedprox_mu 0.0

## sample selection baseline parameters
# ss_baseline True # whether to apply sample selection baseline from our paper, please refer to configs/femnist/femnist_sampleselection_baseline.cfg when running this experiment

## Other paraemeters
global_final_time 300000 # the experiment terminates if the time in the experiment exceeds the global_final_time
# global_final_test_accuracy 0.9 # the experiment terminates if the test accuracy exceeds the global_final_test_accuracy

### ----- NOTE! below are configurations for FedBalancer.
### ----- Strongly recommend: please refer to our paper when configuring below parameters.
fedbalancer True
fb_w 20
fb_p 1.0
fb_simple_control_lt True # whether to control the loss threshold
fb_simple_control_ddl True # whether to control the deadline
fb_simple_control_lt_stepsize 0.05 # ltr in our paper
fb_simple_control_ddl_stepsize 0.05 # ddlr in our paper
fb_client_selection True # if True, fedbalancer performs client selection based on Oort, as written in Section 3.2.3 in our paper. We recommend to set this as True.
fb_inference_pipelining True # if True, fedbalancer clients only performs full-data inference once, when they are first selected for a round. If False, fedbalancer clients performs full-data inference at every selected round to get up-to-date sample-level loss. We recommend to set this as True.
noise_factor 0.5 # noise factor for differential privacy of FedBalancer

# oortbalancer # this option allows us to perform OortBalancer as described in Section 3.4. This should not be used with fedbalancer True option.
```

## How to Parse the Results After the Experiment
- Please refer to the jupyter notebook ipynb scripts in ```results_parsing```

## Notes

- This repository is based on [Chengxu Yang et al.'s work, FLASH](https://github.com/PKU-Chengxu/FLASH), which is a heterogeneity-aware benchmarking framework for FL based on [Sebastian Calas et al.'s work, LEAF](https://leaf.cmu.edu/). We follow the license of LEAF according to the LICENSE.md file.

## Citation

If you publish work that uses this repository, please cite FedBalancer as follows:

```bibtex
@inproceedings{10.1145/3498361.3538917,
author = {Shin, Jaemin and Li, Yuanchun and Liu, Yunxin and Lee, Sung-Ju},
title = {FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients},
year = {2022},
isbn = {9781450391856},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3498361.3538917},
doi = {10.1145/3498361.3538917},
booktitle = {Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services},
pages = {436–449},
numpages = {14},
location = {Portland, Oregon},
series = {MobiSys '22}
}
```
