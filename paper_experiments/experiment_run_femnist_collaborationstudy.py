import os

filenames = [
    # '211102_oortcollaboration/femnist/femnist_realoort_fedprox_mu_0_0.cfg',
    # '211102_oortcollaboration/femnist/femnist_realoort.cfg',
    # '211102_oortcollaboration/femnist/femnist_realoortbalancer_fb_f.cfg',
    # '211102_qfflcollaboration/femnist/femnist_qffl_fedprox_mu_0_0_flash.cfg',
    # '211102_qfflcollaboration/femnist/femnist_qffl_fedavg_flash.cfg',
    # '211102_qfflcollaboration/femnist/femnist_qffl_fb_f_flash.cfg',
    '211102_qfflcollaboration/femnist/femnist_qffl_fb_f_flash_real.cfg',
    # '211102_compressioncollaboration/femnist/femnist_gdrop_fedprox_mu_0_0_flash.cfg',
    # '211102_compressioncollaboration/femnist/femnist_gdrop_fedavg_flash.cfg',
    # '211102_compressioncollaboration/femnist/femnist_gdrop_fb_f_flash.cfg',
    # '211102_compressioncollaboration/femnist/femnist_signsgd_fedprox_mu_0_0.cfg',
    # '211102_compressioncollaboration/femnist/femnist_signsgd_fedavg.cfg',
    # '211102_compressioncollaboration/femnist/femnist_signsgd_fb_f.cfg',
    # '211102_compressioncollaboration/femnist/femnist_structure_100_fedprox_mu_0_0.cfg',
    # '211102_compressioncollaboration/femnist/femnist_structure_100_fedavg.cfg',
    # '211102_compressioncollaboration/femnist/femnist_structure_100_fb_f.cfg',
]
process_count = 0

for file in filenames:
    os.system('CUDA_VISIBLE_DEVICES='+str(7 - process_count // 4)+' python main.py --config=configs/paper_experiments/'+file+' &')
    process_count += 1