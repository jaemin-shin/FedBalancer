import os

config_lines = [
'behav_hete False',
'hard_hete True',
'no_training False',
'realworld False',
'dataset shakespeare',
'model stacked_lstm',
'num_rounds 7500',
'max_client_num -1',
'learning_rate 0.8',
'eval_every 5',
'clients_per_round 10',
'min_selected 1',
'max_sample 2147483647',
'batch_size 100',
# 'seed 0',
'round_ddl 60 0',
'update_frac 0.01',
'aggregate_algorithm SucFedAvg',
'num_epochs 5',
'global_deadline_time 4000000']


baseline_lines = [
(['fedavg_ddl_fixed_1_0'], ['ddl_baseline_fixed True', 'ddl_baseline_fixed_value_multiplied_at_mean 1.0']),
]

process_count = 0
gpu_id = {}
gpu_id[0] = 7


for seed in range(1,2):
    for baseline in baseline_lines:
        new_config_file = open('configs/paper_experiments/211102_baselines/shakespeare/shakespeare_'+baseline[0][0]+'_rs'+str(seed)+'_untilobj.cfg', 'w')
        for line in config_lines:
            new_config_file.write(line+'\n')
        for line in baseline[1]:
            new_config_file.write(line+'\n')
        new_config_file.write('seed '+str(seed)+'\n')
        new_config_file.close()
        os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/211102_baselines/shakespeare/shakespeare_'+baseline[0][0]+'_rs'+str(seed)+'_untilobj.cfg &')
        process_count += 1