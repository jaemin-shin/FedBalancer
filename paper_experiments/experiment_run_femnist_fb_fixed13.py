import os

example_file = open('configs/paper_experiments/210726/femnist_fb_p_0_0_efficiency_deadline_problemfixed_with_oort_realoortversion_stepsize_0_02_ddl_high_fe_fixed5.cfg')

lines = example_file.readlines()


action_steps = [20, 5]
lr_stepsizes = [0.01, 0.05, 0.1]
ddl_stepsizes = [0.05, 0.1, 0.25]
# fb_ps = [0.0, 0.25]
fb_ps = [0.0, 0.25]

process_count = 0

process_count_start = 0
#process_count_end = 20
process_count_end = 9

postfix = 'fixed13'

for action_step in action_steps:
    for fb_p in fb_ps:
        for lr_stepsize in lr_stepsizes:
            for ddl_stepsize in ddl_stepsizes:
                str_as = str(action_step)
                str_lr = '_'.join(str(lr_stepsize).split('.'))
                str_ddl = '_'.join(str(ddl_stepsize).split('.'))
                str_fb_p = '_'.join(str(fb_p).split('.'))
                if os.path.exists('configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log'):
                    file = open('configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log')
                    orig_file_lines = file.readlines()
                    is_final_appeared = False
                    for line in orig_file_lines:
                        orig_file_tmp = line.split(' ')
                        if 'FINAL' in orig_file_tmp:
                            is_final_appeared = True
                    
                    if is_final_appeared:
                        print('configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg.log is done!')
                        continue
                    else:
                        if process_count >= process_count_start and process_count < process_count_end:
                            new_config_file = open('configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg', 'w')
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
                            os.system('CUDA_VISIBLE_DEVICES='+str(7-(process_count//2))+' python main.py --config=configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                        
                            process_count += 1
                else:
                    if process_count >= process_count_start and process_count < process_count_end:
                        new_config_file = open('configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg', 'w')
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
                        os.system('CUDA_VISIBLE_DEVICES='+str(7-(process_count//2))+' python main.py --config=configs/paper_experiments/220228_fedbalancer/femnist/femnist_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'_'+postfix+'.cfg &')
                    
                        process_count += 1