import os

filenames = [
    ('femnist_fb_wo_cs_fixed9.cfg', 0),
    #('femnist_fb_wo_dc_fixed9.cfg', 0),
    #('femnist_fb_wo_ds_fixed9.cfg', 0),
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_ablationstudy/femnist/'+file[0]+' &')
    process_count += 1

filenames = [
    ('reddit_fb_wo_cs_fixed9.cfg', 1),
    #('reddit_fb_wo_dc_fixed9.cfg', 1),
    #('reddit_fb_wo_ds_fixed9.cfg', 1),
]


for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_ablationstudy/reddit/'+file[0]+' &')
    process_count += 1