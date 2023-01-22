# simulate device type
# current classify as big/middle/small device
# device can also be 
from utils.logger import Logger
from utils.device_util.device_util import Device_Util
import numpy as np
import json
import random

# -1 - self define device, 0 - small, 1 - mid, 2 - big

L = Logger()
logger = L.get_logger()

class Device():
        
    du = Device_Util()
    speed_distri = None
    try:
        with open('speed_distri.json', 'r') as f:
            speed_distri = json.load(f) 
    except FileNotFoundError as e:
        speed_distri = None
        logger.warn('no user\'s network speed trace was found, set all communication time to 0.0s')

    # support device type
    def __init__(self, cfg, model_size = 0):
        self.device_model = None    # later set according to the trace
        self.cfg = cfg

        self.sampled_train_time_per_batch = 0
        self.sampled_download_time = 0
        self.sampled_upload_time = 0
        self.sample_count = 10
        
        self.model_size = model_size / 1024 # change to kb because speed data use 'kb/s'
        if cfg.behav_hete == False and cfg.hard_hete == False:
            # make sure the no_trace mode perform the same as original leaf
            self.model_size = 0
        if Device.speed_distri == None:
            # treat as no_trace mode
            self.model_size = 0
            self.upload_speed_u = 1.0
            self.upload_speed_sigma = 0.0
            self.download_speed_u = 1.0
            self.download_speed_sigma = 0.0
        else:
            if cfg.hard_hete == False:
                # assign a fixed speed distribution
                guid = list(Device.speed_distri.keys())[cfg.seed%len(Device.speed_distri)]
            else:
                guid = random.sample(list(Device.speed_distri.keys()), 1)[0]
            self.download_speed_u = Device.speed_distri[guid]['down_u']
            self.download_speed_sigma = Device.speed_distri[guid]['down_sigma']
            self.upload_speed_u = Device.speed_distri[guid]['up_u']
            self.upload_speed_sigma = Device.speed_distri[guid]['up_sigma']
            
        Device.du.set_model(cfg.model)
        Device.du.set_dataset(cfg.dataset)

    def set_device_model(self, real_device_model, client_id):
        device_train_times_per_batch = []
        self.device_model = Device.du.transfer(real_device_model)

        for i in range(self.sample_count): # sample device speed 10 times
            device_train_times_per_batch.append(Device.du.get_train_time_per_batch(self.device_model))

        self.sampled_train_time_per_batch = np.mean(device_train_times_per_batch)
        return device_train_times_per_batch
            
    def set_device_model_weakDeviceToCertainClass(self, real_device_model, label):
        self.device_model = Device.du.unknown_weakDeviceToCertainClass(label)
    
    def get_upload_time(self):
        if self.model_size == 0.0 :
            return 0.0
        
        upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        while upload_speed < 0:
            upload_speed = np.random.normal(self.upload_speed_u, self.upload_speed_sigma)
        
        upload_time = self.model_size / upload_speed
        return float(upload_time)

    def get_download_time(self):
        if self.model_size == 0.0:            
            return 0.0
        
        download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        while download_speed < 0:
            download_speed = np.random.normal(self.download_speed_u, self.download_speed_sigma)
        download_time = self.model_size / download_speed
        return float(download_time)
    
    def get_expected_download_time(self):
        if self.model_size == 0.0:            
            return 0.0
        
        download_speed = self.download_speed_u
        while download_speed < 0:
            download_speed = self.download_speed_u
        download_time = self.model_size / download_speed
        return float(download_time)
    
    def get_expected_upload_time(self):
        if self.model_size == 0.0 :
            return 0.0

        upload_speed = self.upload_speed_u
        while upload_speed < 0:
            upload_speed = self.upload_speed_u
        upload_time = self.model_size / upload_speed
        return float(upload_time)
    
    def get_train_time_and_train_time_per_batch(self, num_sample, batch_size, num_epoch):
        if self.device_model == None:
            assert False
        return Device.du.get_train_time_and_train_time_per_batch_and_train_time_per_epoch(self.device_model, num_sample, batch_size, num_epoch)[:-1]
    
    def get_train_time_and_train_time_per_batch_and_train_time_per_epoch(self, num_sample, batch_size, num_epoch):
        if self.device_model == None:
            assert False
        return Device.du.get_train_time_and_train_time_per_batch_and_train_time_per_epoch(self.device_model, num_sample, batch_size, num_epoch)
        