import os

example_file = open('configs/paper_experiments/211102_fedbalancer/har_raw/har_raw_fb_p0_0_as20_lss0_05_dss0_05_fixed13.cfg')

lines = example_file.readlines()


action_steps = [20, 5]
lr_stepsizes = [0.01, 0.05, 0.1]
ddl_stepsizes = [0.05, 0.1, 0.25]
# fb_ps = [0.0, 0.25]
fb_ps = [0.0, 0.25]

process_count = 0

process_count_start = 0
#process_count_end = 20
process_count_end = 36

postfix = 'fixed13'

gpu_id = list(range(36))
gpu_id[0] = 0
gpu_id[1] = 0
gpu_id[2] = 0
gpu_id[3] = 0
gpu_id[4] = 0
gpu_id[5] = 0
gpu_id[6] = 0
gpu_id[7] = 1
gpu_id[8] = 1
gpu_id[9] = 1
gpu_id[10] = 1
gpu_id[11] = 1
gpu_id[12] = 1
gpu_id[13] = 1
gpu_id[14] = 5
gpu_id[15] = 5
gpu_id[16] = 5
gpu_id[17] = 5
gpu_id[18] = 5
gpu_id[19] = 5
gpu_id[20] = 5
gpu_id[21] = 6
gpu_id[22] = 6
gpu_id[23] = 6
gpu_id[24] = 6
gpu_id[25] = 6
gpu_id[26] = 6
gpu_id[27] = 6
gpu_id[28] = 7
gpu_id[29] = 7
gpu_id[30] = 7
gpu_id[31] = 7
gpu_id[32] = 7
gpu_id[33] = 7
gpu_id[34] = 7
gpu_id[35] = 7
for action_step in action_steps:
    for fb_p in fb_ps:
        for lr_stepsize in lr_stepsizes:
            for ddl_stepsize in ddl_stepsizes:
                str_as = str(action_step)
                str_lr = '_'.join(str(lr_stepsize).split('.'))
                str_ddl = '_'.join(str(ddl_stepsize).split('.'))
                str_fb_p = '_'.join(str(fb_p).split('.'))
                if os.path.exists('configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log'):
                    file = open('configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log')
                    orig_file_lines = file.readlines()
                    is_final_appeared = False
                    for line in orig_file_lines:
                        orig_file_tmp = line.split(' ')
                        if 'FINAL' in orig_file_tmp:
                            is_final_appeared = True
                    
                    if is_final_appeared:
                        print('configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log is done!')
                        continue
                    else:
                        if process_count >= process_count_start and process_count < process_count_end:
                            new_config_file = open('configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg', 'w')
                            for line in lines:
                                tmp = line.strip().split(' ')
                                if 'fedbalancer_action_step' in tmp:
                                    tmp[1] = str(action_step)
                                elif 'fb_simple_control_lt_stepsize' in tmp:
                                    tmp[1] = str(lr_stepsize)
                                elif 'fb_simple_control_ddl_stepsize' in tmp:
                                    tmp[1] = str(ddl_stepsize)
                                elif 'fb_p' in tmp:
                                    tmp[1] = str(fb_p)
                                new_config_file.write(' '.join(tmp)+'\n')
                            new_config_file.close()
                            # os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/211102_fedbalancer/celeba/celeba_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_mu_0_1.cfg &')
                            
                            # os.system('CUDA_VISIBLE_DEVICES='+str(7-(process_count//5))+' python main.py --config=configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                            os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                        
                            process_count += 1
                else:
                    if process_count >= process_count_start and process_count < process_count_end:
                        new_config_file = open('configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg', 'w')
                        for line in lines:
                            tmp = line.strip().split(' ')
                            if 'fedbalancer_action_step' in tmp:
                                tmp[1] = str(action_step)
                            elif 'fb_simple_control_lt_stepsize' in tmp:
                                tmp[1] = str(lr_stepsize)
                            elif 'fb_simple_control_ddl_stepsize' in tmp:
                                tmp[1] = str(ddl_stepsize)
                            elif 'fb_p' in tmp:
                                tmp[1] = str(fb_p)
                            new_config_file.write(' '.join(tmp)+'\n')
                        new_config_file.close()
                        # os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/211102_fedbalancer/celeba/celeba_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_mu_0_1.cfg &')
                        
                        # os.system('CUDA_VISIBLE_DEVICES='+str(7-(process_count//5))+' python main.py --config=configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                        os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/220228_fedbalancer/har_raw/har_raw_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                    
                        process_count += 1