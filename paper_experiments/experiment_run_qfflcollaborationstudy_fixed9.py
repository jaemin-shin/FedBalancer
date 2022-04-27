import os

filenames = [
    ('femnist_qffl_fb_f_fixed9.cfg', 6),
    ('femnist_qffl_fb_s_fixed9.cfg', 7),
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_qfflcollaboration/femnist/'+file[0]+' &')
    process_count += 1

filenames = [
    ('reddit_qffl_fb_f_fixed9.cfg', 6),
    ('reddit_qffl_fb_s_fixed9.cfg', 6),
    ('reddit_qffl_fedavg.cfg', 7),
    ('reddit_qffl_fedprox_mu_0_0.cfg', 7),
]


for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_qfflcollaboration/reddit/'+file[0]+' &')
    process_count += 1