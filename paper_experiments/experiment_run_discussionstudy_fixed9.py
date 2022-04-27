import os

# filenames = [
#     #('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_noip.cfg', 2),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_1.cfg', 1),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_2.cfg', 2),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_5.cfg', 3),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf1_0.cfg', 4),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf2_0.cfg', 5),
#     ('femnist_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf5_0.cfg', 7),
# ]


# for file in filenames:
#     os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_fedbalancer/femnist/'+file[0]+' &')

filenames = [
    #('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_noip.cfg', 1),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_1.cfg', 3),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_2.cfg', 3),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf0_5.cfg', 3),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf1_0.cfg', 7),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf2_0.cfg', 7),
    ('reddit_fb_p0_0_as20_lss0_05_dss0_05_fixed9_nf5_0.cfg', 7),
]


for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(file[1])+' python main.py --config=configs/paper_experiments/211102_fedbalancer/reddit/'+file[0]+' &')