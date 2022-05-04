# FedBalancer

- A systematic federated learning with sample selection to optimize the time-to-accuracy performance
- Includes an adaptive deadline control scheme that predicts the optimal deadline for each round with varying client data

This repository contains the code and experiments for the paper:

>  Conditionally accepted to [MobiSys'22](https://www.sigmobile.org/mobisys/2022/)
>
> [FedBalancer: Data and Pace Control for Efficient Federated Learning on Heterogeneous Clients](https://arxiv.org/abs/2201.01601)

## System Requirements

The system is written and evaluated based on  ```Python 3.6.9```, with ```tensorflow 1.15.4```, running on ```Ubuntu 18.04``` server with eight ```NVIDIA TITAN Xp``` GPUs.

As an alternative setup, you can use general Ubuntu servers with NVIDIA GPUs.

Note that you could also run our system and experiemtns on CPUs, which could be a bit slower. How to run the experiments on CPUs are instructed below.

The experimental results on different setup and different GPUs may differ, but the results will derive same conclusions that we stated in our paper.

## System Installation

Please use ```conda``` to run a self-contained setup:

### Update your NVIDIA driver
The following is to update your driver. If you already have installed the required drivers, this step could be skipped. Be sure to have NVIDIA drivers up-to-date.

After running the commands below, run "nvidia-smi" to confirm your update and check that it is on the 11.1 (or newer) CUDA runtime.

```
$ sudo apt-get dist-upgrade
$ sudo shudown -r now
$ sudo apt-get install dkms build-essential
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get install nvidia-driver-455
$ sudo shutdown -r now
```

### Install miniconda on your system

```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ conda update conda
$ conda update --all
```

### Setup a conda env to install NVIDIA's build of TensorFlow 1.15

```
$ conda create --name fb-conda python=3.6
$ conda activate fb-conda
$ conda install pip
```

### Create a local index for the "wheel" and supporting dependencies

```pip``` will be used for the installing required packages, but the NVIDIA package index is not available on PyPI.org. Please run the following command to set up the index (you should be in activated fb-conda env)

```
$ pip install --user nvidia-pyindex
```

Then, add ```export PATH=$PATH:$HOME/.local/bin``` to your .bashrc file to let new index be recognized by the system. Please re-activate your bash by running the following:

```
$ source ~/.bashrc
```

If you are running other kind of shell like ```zsh```, the above commands should be changed accordingly.

### Install the NVIDIA TensorFlow Build

Run the following while being in the activated fb-conda env.

```
$ conda install -c conda-forge openmpi
```

Add the following to the .bashrc file ```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/fb-conda/lib/``` and re-activate the bash by running the following:

```
$ source ~/.bashrc
```

Then, with the fb-conda env being activated, use the following command to install the ```NVIDIA Tensorflow 1.15``` with ```pip```.

```
$ pip install --user nvidia-tensorflow[horovod]
```

### Install other dependencies

To install other dependencies, please run the following:

```
$ git clone https://github.com/jaemin-shin/FedBalancer.git
$ cd FedBalancer
$ pip install -r requirements.txt
```

<!-- ## Important Note

We evaluated based on five datasets: ```FEMNIST```, ```Celeba```, ```Reddit```, ```Shakespeare```, ```UCI-HAR```.

Currently, this repository only supports experiments with ```Celeba``` and ```UCI-HAR```.

Handling other datasets will be added soon. -->

## How to Run the Experiments

### Running the main experiment of the paper in Section 4.2 and 4.3 one by one

1. Go to directory of respective dataset in `data/` for instructions on generating the benchmark dataset
2. Run (please refer to the list of config files in the subdirectory with respective dataset name in ```configs/```, it contains all the config files for the experiment)
```
$ cd models/

# FedAvg + 1T experiment in Section 4.2 and 4.3 with random seed 0
# candidate {dataset_name}: femnist/celeba/reddit/shakespeare/har

$ python main.py --config=configs/{dataset_name}/{dataset_name}_fedavg_1T_seed0.cfg
# example: python main.py --config=configs/har/har_fedavg_1T_seed0.cfg

# FedBalancer experiment in Section 4.2 and 4.3 with random seed 0
# candidate {dataset_name}: femnist/celeba/reddit/shakespeare/har

$ python main.py --config=configs/{dataset_name}/{dataset_name}_fedbalancer_seed0.cfg
# example: python main.py --config=configs/har/har_fedbalancer_seed0.cfg
```

Add ```CUDA_VISIBLE_DEVICES={GPU_ID} ``` before the command to run the experiment on the specific GPUs. If you set ```GPU_ID``` as ```-1```, the experiment runs on cpus.

### Running the main experiment of the paper in Section 4.2 and 4.3 at ONCE

1. Go to directory of respective dataset in `data/` for instructions on generating the benchmark dataset
2. Configure your python file (IMPORTANT NOTE: before you run the experiment, please refer to the python file that runs all the experiments in `paper_experiments`. You need to assign which GPU you will assign at each experiment, and you may need to run experiments partially as running all the experiments may exceed the VRAM of your GPU; check the available RAM if you are running the experiments on CPU.)
3. Run
```
$ cd models/

# Run baseline experiments
$ python ../paper_experiments/experiment_run_{dataset_name}_baselines.py

# Run fedbalancer experiments
$ python ../paper_experiments/experiment_run_{dataset_name}_fedbalancer.py

# TO RUN WITH CPU: Run baseline experiments
$ python ../paper_experiments/experiment_run_{dataset_name}_baselines_cpu.py

# TO RUN WITH CPU: Run fedbalancer experiments
$ python ../paper_experiments/experiment_run_{dataset_name}_fedbalancer_cpu.py
```

<h3 id="config">Config File</h3>
To simplify the command line arguments, we move most of the parameters to a <span id="jump">config file</span>. Below is a detailed example.

```bash
## whether to consider heterogeneity
behav_hete False # bool, whether to simulate state(behavior) heterogeneity -> fixed to False in our experiments
hard_hete False # bool, whether to simulate hardware heterogeneity, which contains differential on-device training time and network speed -> fixed to True in our experiments

## ML related configurations
dataset har # dataset to use
model lr # file that defines the model (in this example, it is LogisticRegression)
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

aggregate_algorithm SucFedAvg # choose in [SucFedAvg, FedAvg], please refer to models/server.py for more details. In the configuration file, SucFedAvg refers to the "FedAvg" algorithm described in https://arxiv.org/pdf/1602.05629.pdf -> this is fixed to SucFedAvg in our experiments

### ----- NOTE! below are advanced configurations. 
### ----- Strongly recommend: specify these configurations only after reading the source code. 

# compress_algo grad_drop # gradiant compress algorithm, choose in [grad_drop, sign_sgd], not use if commented
# structure_k 100
## the k for structured update, not use if commented, please refer to the arxiv for more 

# qffl True # whether to apply qffl(q-fedavg) and params needed, please refer to the ICLR'20 (https://arxiv.org/pdf/1905.10497.pdf) for more
# qffl_q 5

## parameters related to Oort client selection method, please refer to the OSDI'21 (https://www.usenix.org/conference/osdi21/presentation/lai) for more
realoort True # whether to apply oort or not
# oort_pacer True # whether to apply oort-pacer or not
# oort_pacer_delta 10
# oort_blacklist True # whether to apply oort's client blacklisting or not
# oort_blacklist_rounds

### ----- NOTE! below are configurations for our baselines.
### ----- Strongly recommend: specify these configurations only after reading the source code. 

## parameters to configure deadlines
# ddl_baseline_fixed True # True if experiment uses fixed deadline
# ddl_baseline_fixed_value_multiplied_at_mean 1.0 # In our experiments, the deadline is sampled as mean of clients' round completion times. This parameter indicates the multiplication factor at the sampled deadline. If 1.0, the mean is just used, and this becomes the 1T experiment. If 2.0, 2.0 x mean is used, and this becomes the 2T experiment.
# ddl_baseline_smartpc False # True if SmartPC or Wait-For-All experiment.
# ddl_baseline_smartpc_percentage 0.8 # Specifies the portion of clients that will successfully send the result at a round. If 0.8, this indicates SmartPC experiment. If 1.0, the round waits for every clients to end, and this is Wait-For-All experiment.

## fedprox parameters
# fedprox True # whether to apply fedprox and params needed, please refer to the sysml'20 (https://arxiv.org/pdf/1812.06127.pdf) for more details
# fedprox_mu 0.5

## Other paraemeters
global_final_time 500000 # the experiment terminates if the time in the experiment exceeds the global_final_time
# global_final_test_accuracy 0.9 # the experiment terminates if the test accuracy exceeds the global_final_test_accuracy
# output_path /path/to/your/preferred/directory # path to save experiment output files -- attended clients and clients info

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

# realoortbalancer # this option allows us to perform OortBalancer as described in Section 3.4. This should not be used with fedbalancer True option.
```

## How to Parse the Results After the Experiment
- Please refer to the jupyter notebook ipynb scripts in ```results_parsing```
- Current repository only contains jupyter notebook for ```Celeba``` and ```UCI-HAR``` dataset. Other datasets will be added shortly.
### Celeba Results
- FedAvg+1T: 0.97 ± 0.04 Speedup, 0.847 ± 0.010 Accuracy
- FedAvg+2T: 0.63 ± 0.07 Speedup, 0.825 ± 0.014 Accuracy
- FedAvg+SPC: 0.76 ± 0.17 Speedup, 0.834 ± 0.017 Accuracy
- FedAvg+WFA: 0.55 ± 0.07 Speedup, 0.816 ± 0.016 Accuracy
- Prox+1T: 1.12 ± 0.05 Speedup, 0.854 ± 0.013 Accuracy
- Prox+2T: 0.69 ± 0.06 Speedup, 0.832 ± 0.016 Accuracy
- FedBalancer: 1.53 ± 0.16 Speedup, 0.849 ± 0.005 Accuracy
### UCI-HAR Results
- FedAvg+1T: 0.61 ± 0.34 Speedup, 0.876 ± 0.011 Accuracy
- FedAvg+2T: 0.83 ± 0.24 Speedup, 0.894 ± 0.002 Accuracy
- FedAvg+SPC: 0.69 ± 0.12 Speedup, 0.880 ± 0.005 Accuracy
- FedAvg+WFA: 0.83 ± 0.24 Speedup, 0.894 ± 0.002 Accuracy
- Prox+1T: 0.93 ± 0.15 Speedup, 0.905 ± 0.008 Accuracy
- Prox+2T: 0.83 ± 0.24 Speedup, 0.894 ± 0.002 Accuracy
- FedBalancer: 1.37 ± 0.06 Speedup, 0.914 ± 0.005 Accuracy

## Notes

- This repository is based on [Chengxu Yang et al.'s work, FLASH](https://github.com/PKU-Chengxu/FLASH), which is a heterogeneity-aware benchmarking framework for FL based on [Sebastian Calas et al.'s work, LEAF](https://leaf.cmu.edu/). We follow the license of LEAF according to the LICENSE.md file.
