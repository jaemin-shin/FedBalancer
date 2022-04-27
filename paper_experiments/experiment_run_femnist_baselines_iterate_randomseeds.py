import os

config_lines = [
    'behav_hete False ',
    'hard_hete True',
    'no_training False ',
    'realworld False',
    'dataset femnist',
    'model cnn',
    'num_rounds 6000',
    'max_client_num -1 ',
    'learning_rate 0.001 ',
    'eval_every 5 ',
    'clients_per_round 100 ',
    'min_selected 1 ',
    'max_sample 2147483647',
    'batch_size 10 ',
    # 'seed 0',
    'round_ddl 60 0',
    'update_frac 0.01',
    'aggregate_algorithm SucFedAvg ',
    'num_epochs 5',
    'ddl_baseline_fixed True',
    'ddl_baseline_fixed_value_multiplied_at_mean 1.0',
    'global_deadline_time 200000'
]


baseline_lines = [
(['fedavg_ddl_fixed_1_0'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0']),
(['fedavg_ddl_fixed_2_0'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 2.0']),
(['fedavg_ddl_smartpc'], ['ddl_baseline_smartpc True', 'ddl_baseline_smartpc_percentage 0.8']),
(['fedavg_ddl_waitforall'], ['ddl_baseline_smartpc True', 'ddl_baseline_smartpc_percentage 1.0']),
(['fedprox_mu_0_0_ddl_fixed_1_0'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0', 'fedprox True', 'fedprox_mu 0.0']),
(['fedprox_mu_0_0_ddl_fixed_2_0'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 2.0', 'fedprox True', 'fedprox_mu 0.0'])
]

process_count = 0
gpu_id = {}
gpu_id[0] = 0
gpu_id[1] = 0
gpu_id[2] = 1
gpu_id[3] = 1
gpu_id[4] = 2
gpu_id[5] = 2
gpu_id[6] = 3
gpu_id[7] = 3
gpu_id[8] = 4
gpu_id[9] = 4
gpu_id[10] = 5
gpu_id[11] = 5


for seed in range(1,3):
    for baseline in baseline_lines:
        new_config_file = open('configs/paper_experiments/211102_baselines/femnist/femnist_'+baseline[0][0]+'_rs'+str(seed)+'.cfg', 'w')
        for line in config_lines:
            new_config_file.write(line+'\n')
        for line in baseline[1]:
            new_config_file.write(line+'\n')
        new_config_file.write('seed '+str(seed)+'\n')
        new_config_file.close()
        os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/211102_baselines/femnist/femnist_'+baseline[0][0]+'_rs'+str(seed)+'.cfg &')
        process_count += 1