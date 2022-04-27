import os

filenames = [
    #('femnist_realoortbalancer_fb_old_noip_fixed9.cfg', 2),
    #('femnist_realoortbalancer_fb_old_ip_fixed9.cfg', 2),
    ('femnist_realoortbalancer_fb_s_noip_fixed9.cfg', 2),
    ('femnist_realoortbalancer_fb_s_ip_fixed9.cfg', 3),
    ('femnist_realoortbalancer_fb_f_noip_fixed9.cfg', 5),
    ('femnist_realoortbalancer_fb_f_ip_fixed9.cfg', 5),
]

process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_oortcollaboration/femnist/'+file[0]+' &')
    process_count += 1

filenames = [
    #('reddit_realoortbalancer_fb_old_noip_fixed9.cfg', 3),
    #('reddit_realoortbalancer_fb_old_ip_fixed9.cfg', 3),
    ('reddit_realoortbalancer_fb_s_noip_fixed9.cfg', 3),
    ('reddit_realoortbalancer_fb_s_ip_fixed9.cfg', 4),
    ('reddit_realoortbalancer_fb_f_noip_fixed9.cfg', 5),
    ('reddit_realoortbalancer_fb_f_ip_fixed9.cfg', 5),
    #('reddit_realoort_fedprox_mu_0_0.cfg', 4),
    #('reddit_realoort.cfg', 4),
]


for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_oortcollaboration/reddit/'+file[0]+' &')
    process_count += 1