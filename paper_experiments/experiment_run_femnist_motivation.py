import os

config_lines = [
    'behav_hete False',
    'hard_hete True',
    'no_training False',
    'realworld False',
    'dataset femnist',
    'model cnn',
    'num_rounds 10000',
    'max_client_num -1',
    'learning_rate 0.001',
    'eval_every 5',
    'clients_per_round 100',
    'min_selected 1',
    'max_sample 2147483647',
    'batch_size 10',
    'seed 0',
    'round_ddl 60 0',
    'update_frac 0.01',
    'aggregate_algorithm SucFedAvg',
    'num_epochs 5',
    'ddl_baseline_fixed True',
    'global_deadline_time 1000000'
]


baseline_lines = [
    (['fedavg_ddl_fixed_0_25'], ['ddl_baseline_fixed_value_multiplied_at_mean 0.25'], 0),
    (['fedavg_ddl_fixed_0_75'], ['ddl_baseline_fixed_value_multiplied_at_mean 0.75'], 0),
    (['fedavg_ddl_fixed_1_25'], ['ddl_baseline_fixed_value_multiplied_at_mean 1.25'], 0),
    (['fedavg_ddl_fixed_1_5'], ['ddl_baseline_fixed_value_multiplied_at_mean 1.5'], 0),
    (['fedavg_ddl_fixed_1_75'], ['ddl_baseline_fixed_value_multiplied_at_mean 1.75'], 1),
    (['fedavg_ddl_fixed_2_5'], ['ddl_baseline_fixed_value_multiplied_at_mean 2.5'], 1),
    (['fedavg_ddl_fixed_3_0'], ['ddl_baseline_fixed_value_multiplied_at_mean 3.0'], 1),
    (['fedavg_ddl_fixed_3_5'], ['ddl_baseline_fixed_value_multiplied_at_mean 3.5'], 1),
    (['fedavg_ddl_fixed_4_0'], ['ddl_baseline_fixed_value_multiplied_at_mean 4.0'], 1),
]

process_count = 0

for baseline in baseline_lines:
    new_config_file = open('configs/paper_experiments/211102_baselines/femnist/femnist_'+baseline[0][0]+'.cfg', 'w')
    for line in config_lines:
        new_config_file.write(line+'\n')
    for line in baseline[1]:
        new_config_file.write(line+'\n')
    new_config_file.close()
    os.system('CUDA_VISIBLE_DEVICES='+str(baseline[2])+' python main.py --config=configs/paper_experiments/211102_baselines/femnist/femnist_'+baseline[0][0]+'.cfg &')
    process_count += 1

# action_steps = [5, 10, 20]
# lr_stepsizes = [0.01, 0.05, 0.1, 0.2, 0.25]
# ddl_stepsizes = [0.01, 0.05, 0.1, 0.2, 0.25]
# fb_ps = [0.0, 0.25]

# process_count = 0

# for action_step in action_steps:
#     for lr_stepsize in lr_stepsizes:
#         for ddl_stepsize in ddl_stepsizes:
#             for fb_p in fb_ps:
#                 str_as = str(action_step)
#                 str_lr = '_'.join(str(lr_stepsize).split('.'))
#                 str_ddl = '_'.join(str(ddl_stepsize).split('.'))
#                 str_fb_p = '_'.join(str(fb_p).split('.'))
#                 new_config_file = open('configs/paper_experiments/210726/har_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'.cfg', 'w')
#                 for line in lines:
#                     tmp = line.strip().split(' ')
#                     if 'fedbalancer_action_step' in tmp:
#                         tmp[1] = str(action_step)
#                     elif 'fb_simple_control_lt_stepsize' in tmp:
#                         tmp[1] = str(lr_stepsize)
#                     elif 'fb_simple_control_ddl_stepsize' in tmp:
#                         tmp[1] = str(ddl_stepsize)
#                     elif 'fb_p' in tmp:
#                         tmp[1] = str(fb_p)
#                     new_config_file.write(' '.join(tmp)+'\n')
#                 new_config_file.close()
#                 os.system('CUDA_VISIBLE_DEVICES='+str(7-(process_count//30))+' python main.py --config=configs/paper_experiments/210726/har_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'.cfg &')
#                 process_count += 1