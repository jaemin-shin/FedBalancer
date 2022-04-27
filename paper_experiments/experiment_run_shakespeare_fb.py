import os

example_file = open('configs/paper_experiments/210726/shakespeare_fb_p_0_0_ltstepsize_0_01.cfg')

lines = example_file.readlines()


action_steps = [20]
lr_stepsizes = [0.05]
ddl_stepsizes = [0.1]
fb_ps = [0.0]

process_count = 0

for action_step in action_steps:
    for lr_stepsize in lr_stepsizes:
        for ddl_stepsize in ddl_stepsizes:
            for fb_p in fb_ps:
                str_as = str(action_step)
                str_lr = '_'.join(str(lr_stepsize).split('.'))
                str_ddl = '_'.join(str(ddl_stepsize).split('.'))
                str_fb_p = '_'.join(str(fb_p).split('.'))
                new_config_file = open('configs/paper_experiments/211102_fedbalancer/shakespeare/shakespeare_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'.cfg', 'w')
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
                os.system('CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/paper_experiments/211102_fedbalancer/shakespeare/shakespeare_fb_p'+str_fb_p+'_as'+str_as+'_lss'+str_lr+'_dss'+str_ddl+'.cfg &')
                process_count += 1