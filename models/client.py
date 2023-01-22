import random
import warnings
import timeout_decorator
import sys
import numpy as np
import json

from utils.logger import Logger
from device import Device
from timer import Timer

import torch

L = Logger()
logger = L.get_logger()

class Client:
    
    d = None
    try:
        with open('../data/state_traces.json', 'r', encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError as e:
        d = None
        logger.warn('no user behavior trace was found, running in no-trace mode')
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, device=None, cfg=None):
        self._model = None
        self.id = client_id # integer
        self.group = group
        
        self.deadline = 1 # < 0 for unlimited
        self.cfg = cfg
        
        self.train_data = train_data
        if eval_data != None:
            self.eval_data={"x": self.preprocess_data_x(eval_data["x"]),
                            "y": self.preprocess_data_y(eval_data["y"])}
        else:
            self.eval_data = eval_data
        
        self.fedbalancer = None
        self.oort = None

        # FEDBALANCER parameter
        self.loss_threshold = 0
        self.train_time_per_batch_list = []
        self.sorted_loss = []

        self.inference_times = []
        self.inference_times_per_sample = []
        self.per_epoch_train_times = []
        self.per_batch_train_times = []

        self.trained_num_of_samples = []
        
        self.device = device  # if device == none, it will use real time as train time, and set upload/download time as 0
        if self.device == None:
            logger.warn('client {} with no device init, upload time will be set as 0 and speed will be the gpu speed'.format(self.id))
            self.upload_time = 0
        
        # timer
        d = Client.d
        if d == None:
            cfg.behav_hete = False
        if cfg.behav_hete:
            uid = random.sample(list(d.keys()), 1)[0]
            self.timer = Timer(ubt=d[str(uid)], google=True)
            while self.timer.isSuccess != True:
                uid = random.sample(list(d.keys()), 1)[0]
                self.timer = Timer(ubt=d[str(uid)], google=True)
        else:
            # no behavior heterogeneity, always available
            self.timer = Timer(None)
            self.deadline = sys.maxsize # deadline is meaningless without user trace
        
        real_device_model = self.timer.model
        
        if not self.device: 
            self.device = Device(cfg, 0.0)
        
        # For SampleSelection baseline, FedSS (L. Cai et al., ICC'20)
        if self.cfg.ss_baseline:
            self.is_big_client = False
            self.select_sample_num = 0
        
        # Sample round completion time of a client, to calculate the initial deadline in main.py
        # Here, train time and inference time is calculated
        if self.cfg.hard_hete:
            curr_per_batch_train_times = self.device.set_device_model(real_device_model, self.id) #real_device_model is None if behav_hete is True
            for per_batch_train_time in curr_per_batch_train_times:
                self.per_batch_train_times.append(per_batch_train_time)
            if train_data != None:
                num_train_samples = len(train_data['x'])
            else:
                num_train_samples = 1
            for per_batch_train_time in curr_per_batch_train_times:
                if self.cfg.oortbalancer or self.cfg.oort:
                    self.per_epoch_train_times.append(per_batch_train_time)
                    self.per_batch_train_times.append(per_batch_train_time)
                    
                    self.trained_num_of_samples.append(self.cfg.batch_size)
                    if self.cfg.oortbalancer:
                        self.inference_times.append(((num_train_samples-1)//self.cfg.batch_size + 1) * per_batch_train_time * 0.5)
                        self.inference_times_per_sample.append(self.inference_times[-1]/num_train_samples)
                else:
                    self.per_epoch_train_times.append(((num_train_samples-1)//self.cfg.batch_size + 1) * per_batch_train_time)
                    self.per_batch_train_times.append(per_batch_train_time)

                    self.trained_num_of_samples.append(num_train_samples)
                
                    if self.cfg.fedbalancer:
                        # Inference times are calculated as half of the epoch train times
                        self.inference_times.append(self.per_epoch_train_times[-1]*0.5)
                        self.inference_times_per_sample.append(self.inference_times[-1]/num_train_samples)
        else:
            curr_per_batch_train_times = self.device.set_device_model("Redmi Note 8", self.id)
            for per_batch_train_time in curr_per_batch_train_times:
                self.per_batch_train_times.append(per_batch_train_time)
            if train_data != None:
                num_train_samples = len(train_data['x'])
            else:
                num_train_samples = 1
            for per_batch_train_time in curr_per_batch_train_times:
                self.per_epoch_train_times.append(((num_train_samples-1)//self.cfg.batch_size + 1) * per_batch_train_time)
                self.per_batch_train_times.append(per_batch_train_time)

                self.trained_num_of_samples.append(num_train_samples)
                
                self.inference_times.append(self.per_epoch_train_times[-1]*0.5)
                self.inference_times_per_sample.append(self.inference_times[-1]/num_train_samples)
        
        self.sampled_per_epoch_train_time = np.mean(self.per_epoch_train_times)
        self.whole_data_loss_list = []
        self.is_first_round = True

    def preprocess_data_x(self,data):
        return torch.tensor(data,requires_grad=True)
    def preprocess_data_y(self,data):
        data_y=[]
        for i in data:
            data_float=float(i)
            data_y.append(data_float)
        return torch.tensor(data_y,requires_grad=True)
    
    def inference_on_whole_dataset(self, whole_data):
        return self.model.test(whole_data)['loss_list']

    def train(self, start_t=None, num_epochs=1, batch_size=10):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
        """
        
        def train_with_simulate_time(self, start_t, num_epochs=1, batch_size=10):

            ne=-1

            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            user_whole_data_len = num_data

            if (self.cfg.fedbalancer or self.cfg.fb_client_selection or self.cfg.oortbalancer or (self.cfg.ss_baseline and self.is_big_client)):
                # if cfg.fb_inference_pipelining is True, it means that FedBalancer is not doing a full pass on the client's data 
                # except when a client is first selected for a round...
                # otherwise, FedBalancer do full pass on the client's data at the start of training at a round
                if self.is_first_round or (not self.cfg.fb_inference_pipelining):
                    self.whole_data_loss_list = self.fedbalancer.calculate_loss_on_whole_dataset_with_inference(self.train_data, self.model)

            # Sample selection on a client at a FL round with loss_threshold
            if self.cfg.fedbalancer:
                selected_data, num_data, data_idx, self.sorted_loss = self.fedbalancer.fb_sample_selection(num_data, self.loss_threshold, self.whole_data_loss_list, self.train_data, self.deadline, self.train_time_per_batch_list, num_epochs, batch_size)
            # Oortbalancer sample selection
            elif self.cfg.oortbalancer:
                selected_data, xss, yss, num_data, data_idx, self.sorted_loss = self.fedbalancer.fb_oortbalancer_sample_selection(batch_size, self.loss_threshold, self.whole_data_loss_list, self.train_data, self.deadline, self.train_time_per_batch_list, num_epochs, self.model)
            # Oort randomly selects a batch per epoch for training at this round
            # Oort selects same samples multiple times if the client has less data than batch_size * num_epoch
            elif self.cfg.oort:
                selected_data, xss, yss, num_data, data_idx, self.sorted_loss = self.oort.select_batch_samples(batch_size, self.train_data, num_epochs, self.model)
            elif (self.cfg.ss_baseline and self.is_big_client):
                data_len = self.select_sample_num
                tmp_data = zip(self.train_data["x"], self.train_data["y"])
                tmp_data = zip(tmp_data, range(len(self.train_data["x"])))
                tmp_data = zip(self.whole_data_loss_list, tmp_data)
                tmp_data = sorted(tmp_data, reverse=True, key=lambda elem: elem[0])

                tmp_data_pkg = [x for _,x in tmp_data[:data_len]]
                tmp_data = [x for x,_ in tmp_data_pkg]
                tmp_data_idx = [x for _,x in tmp_data_pkg]

                num_data = self.select_sample_num
                xs, ys = zip(*tmp_data)
                data_idx = tmp_data_idx
                selected_data = {'x': xs, 'y': ys}
                selected_data = {"x": self.preprocess_data_x(selected_data["x"]),
                                "y": self.preprocess_data_y(selected_data["y"])}
            else:
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data_idx = list(range(len(ys)))
                selected_data = {'x': xs, 'y': ys}
                selected_data = {"x": self.preprocess_data_x(selected_data["x"]),
                                "y": self.preprocess_data_y(selected_data["y"])}
            
            
            data = selected_data
            
            train_time, train_time_per_batch, train_time_per_epoch = self.device.get_train_time_and_train_time_per_batch_and_train_time_per_epoch(num_data, batch_size, num_epochs)
            # print(num_data, train_time, train_time_per_batch, train_time_per_epoch)
            self.train_time_per_batch_list.append(train_time_per_batch)

            logger.debug('client {}: num data:{}'.format(self.id, num_data))
            logger.debug('client {}: train time:{}'.format(self.id, train_time))
            
            
            # Calculate the time that elapses for full data inferencing at FedBalancer, if fb_inference_pipelining is False
            # We recommend to let fb_inference_pipelining as True to enjoy better time-to-accuracy performance
            inference_time = 0
            
            if self.cfg.fedbalancer:
                if num_data == user_whole_data_len:
                    inference_time = 0
                elif self.cfg.fb_inference_pipelining and not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            elif (self.cfg.ss_baseline and self.is_big_client):
                if not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            elif self.cfg.oortbalancer:
                if self.cfg.fb_inference_pipelining and not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            
            if self.is_first_round:
                self.is_first_round = False

            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time() # will be re-calculated after training

            self.act_inference_time = 0
            self.ori_inference_time = 0

            if self.cfg.fedbalancer or self.cfg.oortbalancer or (self.cfg.ss_baseline and self.is_big_client):
                down_end_time = self.timer.get_future_time(start_t, download_time)
                logger.debug("client {} download-time-need={}, download-time-cost={} end at {}, "
                            .format(self.id, download_time, down_end_time-start_t, down_end_time))
                
                inference_end_time = self.timer.get_future_time(down_end_time, inference_time)
                logger.debug("client {} inference-time-need={}, inference-time-cost={} end at {}, "
                            .format(self.id, inference_time, inference_end_time-down_end_time, inference_end_time))

                train_end_time = self.timer.get_future_time(inference_end_time, train_time)
                logger.debug("client {} train-time-need={}, train-time-cost={} end at {}, "
                            .format(self.id, train_time, train_end_time-inference_end_time, train_end_time))
                
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                logger.debug("client {} upload-time-need={}, upload-time-cost={} end at {}, "
                            .format(self.id, upload_time, up_end_time-train_end_time, up_end_time))

                self.ori_download_time = download_time  # original
                self.ori_inference_time = inference_time
                self.ori_train_time = train_time
                self.before_comp_upload_time = upload_time
                self.ori_upload_time = upload_time

                self.act_download_time = down_end_time-start_t # actual
                self.act_inference_time = inference_end_time - down_end_time
                self.act_train_time = train_end_time-inference_end_time
                self.act_upload_time = up_end_time-train_end_time   # maybe decrease for the use of conpression algorithm
                
                self.update_size = self.model.size
            else:
                down_end_time = self.timer.get_future_time(start_t, download_time)
                logger.debug("client {} download-time-need={}, download-time-cost={} end at {}, "
                            .format(self.id, download_time, down_end_time-start_t, down_end_time))

                train_end_time = self.timer.get_future_time(down_end_time, train_time)
                logger.debug("client {} train-time-need={}, train-time-cost={} end at {}, "
                            .format(self.id, train_time, train_end_time-down_end_time, train_end_time))
                
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                logger.debug("client {} upload-time-need={}, upload-time-cost={} end at {}, "
                            .format(self.id, upload_time, up_end_time-train_end_time, up_end_time))

                self.ori_download_time = download_time  # original
                self.ori_train_time = train_time
                self.before_comp_upload_time = upload_time
                self.ori_upload_time = upload_time

                self.act_download_time = down_end_time-start_t # actual
                self.act_train_time = train_end_time-down_end_time
                self.act_upload_time = up_end_time-train_end_time   # maybe decrease for the use of conpression algorithm
                
                self.update_size = self.model.size

            data_loss_list_and_idx = []

            # Train with selected samples on a client for FL round
            # If self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc, clients perform training till the end -- clients that will not be accepted are handled on server.py in this case
            # Otherwise, if training exceeds the deadline, the client round fails and raises timeouterror
            if not (self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc):
                if not (self.cfg.fedbalancer or self.cfg.oortbalancer or (self.cfg.ss_baseline and self.is_big_client)):
                    if (down_end_time-start_t) > self.deadline:
                        # download too long
                        self.update_size = 0
                        failed_reason = 'failed when downloading'
                        self.sorted_loss = []
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (train_end_time-start_t) > self.deadline:
                        # failed when training
                        train_time_limit = self.deadline - self.act_download_time
                        if train_time_limit <= 0:
                            train_time_limit = 0.001
                        available_time = self.timer.get_available_time(start_t + self.act_download_time, train_time_limit)
                        self.update_size = 0
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.oortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(down_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                update, acc, loss, = -1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.oortbalancer) and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when training'
                                self.sorted_loss = []
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when training'
                            self.sorted_loss = []
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (up_end_time-start_t) > self.deadline:
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.oortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(down_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                update, acc, loss, = -1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.oortbalancer) and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when uploading'
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                self.per_epoch_train_times.append(train_time_per_epoch)
                                self.per_batch_train_times.append(train_time_per_batch)
                                if not self.cfg.oortbalancer:
                                    self.trained_num_of_samples.append(len(data['x']))
                                self.sorted_loss = []
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when uploading'
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            self.per_epoch_train_times.append(train_time_per_epoch)
                            self.per_batch_train_times.append(train_time_per_batch)
                            if not self.cfg.oortbalancer:
                                self.trained_num_of_samples.append(len(data['x']))
                            self.sorted_loss = []
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    else:
                        if self.cfg.no_training:
                            update, acc, loss = -1,-1,-1,-1,-1
                        else:
                            if self.cfg.oort or self.cfg.oortbalancer:
                                update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, num_epochs, batch_size, self.cfg.oortbalancer)
                            else:
                                update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                            logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
                else: # fedbalancer or oortbalancer or (self.cfg.ss_baseline and self.is_big_client)
                    if (down_end_time-start_t) > self.deadline:
                        # download too long
                        self.update_size = 0
                        failed_reason = 'failed when downloading'
                        self.sorted_loss = []
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (inference_end_time-start_t) > self.deadline:
                        # download too long
                        self.update_size = 0
                        failed_reason = 'failed when inferencing'
                        self.sorted_loss = []
                        #self.download_times.append(download_time)
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (train_end_time-start_t) > self.deadline:
                        # failed when training
                        train_time_limit = self.deadline - self.act_download_time - self.act_inference_time
                        if train_time_limit <= 0:
                            train_time_limit = 0.001
                        available_time = self.timer.get_available_time(start_t + self.act_download_time + self.act_inference_time, train_time_limit)
                        self.update_size = 0
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.oortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(inference_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                update, acc, loss = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.oort:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.oortbalancer) and ne != -1:
                                if self.cfg.oort:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when training'
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                if (self.act_download_time + self.act_inference_time) > self.deadline:
                                    self.sorted_loss = []
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            if (self.act_download_time + self.act_inference_time) > self.deadline:
                                self.sorted_loss = []
                            failed_reason = 'failed when training'
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (up_end_time-start_t) > self.deadline:
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.oortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(inference_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                update, acc, loss = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.oortbalancer) and ne != -1:
                                if self.cfg.oort or self.cfg.oortbalancer:
                                    update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, ne, batch_size, self.cfg.oortbalancer)
                                else:
                                    update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when uploading'
                                if (self.act_download_time + self.act_inference_time) > self.deadline:
                                    self.sorted_loss = []
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                self.per_epoch_train_times.append(train_time_per_epoch)
                                self.per_batch_train_times.append(train_time_per_batch)
                                if not self.cfg.oortbalancer:
                                    self.trained_num_of_samples.append(len(data['x']))
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when uploading'
                            if (self.act_download_time + self.act_inference_time) > self.deadline:
                                self.sorted_loss = []
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            self.per_epoch_train_times.append(train_time_per_epoch)
                            self.per_batch_train_times.append(train_time_per_batch)
                            if not self.cfg.oortbalancer:
                                self.trained_num_of_samples.append(len(data['x']))
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    else:
                        if self.cfg.no_training:
                            update, acc, loss = -1,-1,-1,-1,-1
                        else:
                            if self.cfg.oort or self.cfg.oortbalancer:
                                update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, num_epochs, batch_size, self.cfg.oortbalancer)
                            else:
                                update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                            logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
            else: # oort_pacer or ddl_baseline_smartpc
                if self.cfg.no_training:
                    update, acc, loss = -1,-1,-1,-1,-1
                else:
                    if self.cfg.oort or self.cfg.oortbalancer:
                        update, acc, loss, data_loss_list_and_idx = self.model.oorttrain(data_idx, xss, yss, num_epochs, batch_size, self.cfg.oortbalancer)
                    else:
                        update, acc, loss, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                    logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
            
            # If fb_inference_pipelining == True, update the self.whole_data_loss_list of a client by the loss that is naturally acquired from the training process
            # only the selected samples' loss is updated in this case
            if (self.cfg.fb_inference_pipelining or (self.cfg.ss_baseline and self.is_big_client)) and len(data_loss_list_and_idx) > 0:
                data_loss_list = [x for x,_ in data_loss_list_and_idx]
                data_loss_list_idx = [x for _,x in data_loss_list_and_idx]

                for dll_idx, d_idx in enumerate(data_loss_list_idx):
                    self.whole_data_loss_list[d_idx] = data_loss_list[dll_idx]

            num_train_samples = len(data['y'])
            simulate_time_c = download_time + inference_time + train_time + upload_time
            
            # Changing train time for sub-epoch training
            if (self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.oortbalancer) and ne != -1:
                self.act_train_time = self.act_train_time * ne / num_epochs
            
            total_cost = self.act_download_time + self.act_inference_time + self.act_train_time + self.act_upload_time
            
            if total_cost > self.deadline and not (self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc):
                # failed when uploading
                failed_reason = 'failed when uploading'
                # Note that, to simplify, we did not change the update_size here, actually the actual update size is less.
                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                    self.inference_times.append(inference_time)
                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                self.per_epoch_train_times.append(train_time_per_epoch)
                self.per_batch_train_times.append(train_time_per_batch)
                if not self.cfg.oortbalancer:
                    self.trained_num_of_samples.append(len(data['x']))

                if (self.act_download_time + self.act_inference_time) > self.deadline:
                    self.sorted_loss = []
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)

            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                self.inference_times.append(inference_time)
                self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
            self.per_epoch_train_times.append(train_time_per_epoch)
            self.per_batch_train_times.append(train_time_per_batch)
            if not self.cfg.oortbalancer:
                self.trained_num_of_samples.append(len(data['x']))

            if ne == -1:
                return simulate_time_c, num_train_samples, update, acc, loss, self.update_size, self.sorted_loss, download_time, upload_time, train_time, inference_time, num_epochs
            else:
                return simulate_time_c, num_train_samples, update, acc, loss, self.update_size, self.sorted_loss, download_time, upload_time, train_time, inference_time, ne

        return train_with_simulate_time(self, start_t, num_epochs, batch_size)


    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
    
    
    def set_deadline(self, deadline = -1):
        #if deadline < 0 or not self.cfg.behav_hete:
        if deadline < 0:
            self.deadline = sys.maxsize
        else:
            self.deadline = deadline
        logger.debug('client {}\'s deadline is set to {}'.format(self.id, self.deadline))
    
    def set_loss_threshold(self, loss_threshold = -1):
        #if deadline < 0 or not self.cfg.behav_hete:
        if loss_threshold < 0:
            self.loss_threshold = 0
        else:
            self.loss_threshold = loss_threshold
        logger.debug('client {}\'s loss threshold is set to {}'.format(self.id, self.loss_threshold))
    
    def upload_suc(self, start_t, num_epochs=1, batch_size=10):
        """Test if this client will upload successfully

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            result: test result(True or False)
        """
        num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
        if self.device == None:
            download_time = 0.0
            upload_time = 0.0
        else:
            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time()
        train_time, _ = self.device.get_train_time_and_train_time_per_batch(num_data, batch_size, num_epochs)
        train_time_limit = self.deadline - download_time - upload_time
        if train_time_limit < 0:
            train_time_limit = 0.001
        available_time = self.timer.get_available_time(start_t + download_time, train_time_limit)
        
        logger.debug('client {}: train time:{}'.format(self.id, train_time))
        logger.debug('client {}: available time:{}'.format(self.id, available_time))
        
        # compute num_data
        num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
        xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
        data = {'x': xs, 'y': ys}
        
        if not self.timer.check_comm_suc(start_t, download_time):
            return False
        if train_time > train_time_limit:
            return False
        elif train_time > available_time:
            return False
        if not self.timer.check_comm_suc(start_t + download_time + train_time, upload_time):
            return False
        else :
            return True

    
    def get_device_model(self):
        if self.device == None:
            return 'None'
        return self.device.device_model
