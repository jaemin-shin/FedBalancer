import os

example_file = open('configs/paper_experiments/210726/shakespeare_fb_p_0_0_ltstepsize_0_01_fixed5.cfg')

lines = example_file.readlines()

experiments = [
    ['fb_p0_0_as20_lss0_05_dss0_05', (0.0, 20, 0.05, 0.05)],
    ['fb_p0_0_as20_lss0_1_dss0_05', (0.0, 20, 0.1, 0.05)],
    ['fb_p0_0_as5_lss0_01_dss0_1', (0.0, 5, 0.01, 0.1)],
    ['fb_p0_25_as5_lss0_05_dss0_1', (0.25, 5, 0.05, 0.1)]
]

gpu_id = {}
gpu_id[0] = 0
gpu_id[1] = 1
gpu_id[2] = 2
gpu_id[3] = 3
gpu_id[4] = 4
gpu_id[5] = 5
gpu_id[6] = 6
gpu_id[7] = 7

process_count = 0

for exp in experiments:
    for seed in range(1,3):
        new_config_file = open('configs/paper_experiments/220228_fedbalancer/shakespeare/shakespeare_'+exp[0]+'_fixed13_rs'+str(seed)+'.cfg', 'w')
        for line in lines:
            tmp = line.strip().split(' ')
            if 'fedbalancer_action_step' in tmp:
                tmp[1] = str(exp[1][1])
            elif 'fb_simple_control_lt_stepsize' in tmp:
                tmp[1] = str(exp[1][2])
            elif 'fb_simple_control_ddl_stepsize' in tmp:
                tmp[1] = str(exp[1][3])
            elif 'fb_p' in tmp:
                tmp[1] = str(exp[1][0])
            elif 'seed' in tmp:
                tmp[1] = str(seed)
            new_config_file.write(' '.join(tmp)+'\n')
        new_config_file.close()
        os.system('CUDA_VISIBLE_DEVICES='+str(gpu_id[process_count])+' python main.py --config=configs/paper_experiments/220228_fedbalancer/shakespeare/shakespeare_'+exp[0]+'_fixed13_rs'+str(seed)+'.cfg &')

        process_count += 1