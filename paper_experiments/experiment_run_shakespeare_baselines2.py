import os

filenames = [
    ('shakespeare_batchexp_fedavg_ddl_fixed_1_0_untilobj.cfg',3)
    #('shakespeare_fedprox_mu_0_0_ddl_fixed_2_0.cfg',4)
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_baselines/shakespeare/'+file[0]+' &')
    process_count += 1