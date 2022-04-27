import os

filenames = [
    ('femnist_structure_100_fb_f_fixed9.cfg', 0),
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_compressioncollaboration/femnist/'+file[0]+' &')
    process_count += 1

filenames = [
    ('reddit_structure_100_fb_f_fixed9.cfg', 6),
    ('reddit_structure_100_fedavg.cfg', 0),
    ('reddit_structure_100_fedprox_mu_0_0.cfg', 6),
]


for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_compressioncollaboration/reddit/'+file[0]+' &')
    process_count += 1