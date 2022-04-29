import os

config_lines = [
'behav_hete False',
'hard_hete True',
'no_training False',
'realworld False',
'dataset har',
'model lr',
'num_rounds 6000',
'max_client_num -1',
'learning_rate 0.0003',
'eval_every 5',
'clients_per_round 5',
'min_selected 1',
'max_sample 2147483647',
'batch_size 10',
'update_frac 0.01',
'aggregate_algorithm SucFedAvg',
'num_epochs 5',
'output_path ../models/configs/har/',
'fedbalancer True',
'fb_simple_control_lt True',
'fb_simple_control_ddl True',
'fb_inference_pipelining True',
'fb_client_selection True',
'global_final_time 500000']


fedbalancer_lines = [
(['fedbalancer'], ['fb_w 20', 'fb_simple_control_lt_stepsize 0.05', 'fb_simple_control_ddl_stepsize 0.05', 'fb_p 1.0'])
]

process_count = 0

process_count = 0

for seed in range(3):
    for exp in fedbalancer_lines:
        new_config_file = open('../models/configs/har/har_'+exp[0][0]+'_seed'+str(seed)+'.cfg', 'w')
        for line in config_lines:
            new_config_file.write(line+'\n')
        for line in exp[1]:
            new_config_file.write(line+'\n')
        new_config_file.write('seed '+str(seed)+'\n')
        new_config_file.close()
        os.system('CUDA_VISIBLE_DEVICES=-1 python ../models/main.py --config=../models/configs/har/har_'+exp[0][0]+'_seed'+str(seed)+'.cfg &')
        process_count += 1