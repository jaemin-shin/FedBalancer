import os

filenames = [
    ('femnist_fb_wo_cs_dc_ds.cfg', 3),
    ('femnist_fb_wo_cs_dc.cfg', 3),
    ('femnist_fb_wo_cs_ds.cfg', 4),
    ('femnist_fb_wo_cs.cfg', 4),
    ('femnist_fb_wo_dc_ds.cfg', 5),
    ('femnist_fb_wo_dc.cfg', 5),
    ('femnist_fb_wo_ds.cfg', 2),
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_ablationstudy/femnist/'+file[0]+' &')
    process_count += 1