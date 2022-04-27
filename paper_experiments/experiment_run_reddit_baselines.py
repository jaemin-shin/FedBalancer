import os

filenames = [
    'reddit_fedavg_ddl_fixed_1_0_untilobj.cfg'
    # 'reddit_fedavg_ddl_fixed_2_0_untilobj.cfg',
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES=3 python main.py --config=configs/paper_experiments/211102_baselines/reddit/'+file+' &')
    process_count += 1