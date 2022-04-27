import os

filenames = [
    'femnist_fedavg_ddl_fixed_2_0_untilobj.cfg',
    'femnist_fedavg_ddl_smartpc_untilobj.cfg',
    'femnist_fedavg_ddl_waitforall_untilobj.cfg',
    'femnist_fedprox_mu_0_0_ddl_fixed_2_0_untilobj.cfg',
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES=2 python main.py --config=configs/paper_experiments/211102_baselines/femnist/'+file+' &')
    process_count += 1