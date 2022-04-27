import os

filenames = [
    'reddit_fb_wo_cs_dc_ds.cfg',
    'reddit_fb_wo_cs_dc.cfg',
    'reddit_fb_wo_cs.cfg',
    'reddit_fb.cfg'
]
process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES=7 python main.py --config=configs/paper_experiments/211102_ablationstudy/reddit/'+file+' &')
    process_count += 1