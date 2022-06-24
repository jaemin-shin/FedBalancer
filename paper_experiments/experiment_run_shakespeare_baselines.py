import os

config_lines = [
'behav_hete False',
'hard_hete True',
'no_training False',
'realworld False',
'dataset shakespeare',
'model stacked_lstm',
'num_rounds 750',
'max_client_num -1',
'learning_rate 0.8',
'eval_every 5',
'clients_per_round 10',
'min_selected 1',
'max_sample 2147483647',
'batch_size 100',
'update_frac 0.01',
'aggregate_algorithm SucFedAvg',
'num_epochs 5',
'output_path ../models/configs/shakespeare/',
'global_final_time 2000000']

baseline_lines = [
(['fedavg_1T'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0']),
(['fedavg_2T'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 2.0']),
(['fedavg_SPC'], ['ddl_baseline_smartpc True', 'ddl_baseline_smartpc_percentage 0.8']),
(['fedavg_WFA'], ['ddl_baseline_smartpc True', 'ddl_baseline_smartpc_percentage 1.0']),
(['fedprox_mu_0_0_1T'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0', 'fedprox True', 'fedprox_mu 0.0']),
(['fedprox_mu_0_0_2T'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 2.0', 'fedprox True', 'fedprox_mu 0.0']),
(['sampleselection_baseline'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0', 'fedprox True', 'fedprox_mu 0.0', 'ss_baseline True'])
]

process_count = 0
gpu_id = {}
gpu_id[0] = 0
gpu_id[1] = 1
gpu_id[2] = 3
gpu_id[3] = 4
gpu_id[4] = 5
gpu_id[5] = 6
gpu_id[6] = 6
gpu_id[7] = 3
gpu_id[8] = 4
gpu_id[9] = 4
gpu_id[10] = 5
gpu_id[11] = 5
gpu_id[12] = 6
gpu_id[13] = 6
gpu_id[14] = 0
gpu_id[15] = 1
gpu_id[16] = 2
gpu_id[17] = 3


for seed in range(0,3,1):
    for exp in baseline_lines:
        new_config_file = open('../models/configs/shakespeare/shakespeare_'+exp[0][0]+'_seed'+str(seed)+'.cfg', 'w')
        for line in config_lines:
            new_config_file.write(line+'\n')
        for line in exp[1]:
            new_config_file.write(line+'\n')
        new_config_file.write('seed '+str(seed)+'\n')
        new_config_file.close()
        os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python ../models/main.py --config=../models/configs/shakespeare/shakespeare_'+exp[0][0]+'_seed'+str(seed)+'.cfg &')
        process_count += 1