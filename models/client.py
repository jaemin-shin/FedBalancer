import random
import warnings
import timeout_decorator
import sys
import numpy as np
import json

from utils.logger import Logger
from device import Device
from timer import Timer

from grad_compress.grad_drop import GDropUpdate
from grad_compress.sign_sgd import SignSGDUpdate
from comm_effi import StructuredUpdate

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
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None, device=None, cfg=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.deadline = 1 # < 0 for unlimited
        self.cfg = cfg

        # FEDBALANCER parameter
        self.loss_threshold = 0
        self.upper_loss_threshold = 0

        self.train_time_per_batch_list = []

        self.sorted_loss = []

        self.inference_times = []
        self.inference_times_per_sample = []
        self.per_epoch_train_times = []
        self.per_batch_train_times = []

        self.trained_num_of_samples = []
        self.uniquely_seen_num_of_samples = []

        self.compressor = None
        if self.cfg.compress_algo:
            if self.cfg.compress_algo == 'sign_sgd':
                self.compressor = SignSGDUpdate()
            elif self.cfg.compress_algo == 'grad_drop':
                self.compressor = GDropUpdate(client_id,cfg)
            else:
                logger.error("compress algorithm is not defined")
        
        self.structured_updater = None
        if self.cfg.structure_k:
            self.structured_updater = StructuredUpdate(self.cfg.structure_k, self.cfg.seed)
        
        self.device = device  # if device == none, it will use real time as train time, and set upload/download time as 0
        if self.device == None:
            logger.warn('client {} with no device init, upload time will be set as 0 and speed will be the gpu speed'.format(self.id))
            self.upload_time = 0
        
        # timer
        d = Client.d
        if d == None:
            cfg.behav_hete = False
        # uid = random.randint(0, len(d))
        if cfg.behav_hete:
            if cfg.real_world == False:
                uid = random.sample(list(d.keys()), 1)[0]
                self.timer = Timer(ubt=d[str(uid)], google=True)
                while self.timer.isSuccess != True:
                    uid = random.sample(list(d.keys()), 1)[0]
                    self.timer = Timer(ubt=d[str(uid)], google=True)
            else:
                uid = self.id
                self.timer = Timer(ubt=d[str(uid)], google=True)
        else:
            # no behavior heterogeneity, always available
            self.timer = Timer(None)
            self.deadline = sys.maxsize # deadline is meaningless without user trace
        
        real_device_model = self.timer.model
        
        if not self.device: 
            self.device = Device(cfg, 0.0)
        
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
                if self.cfg.realoortbalancer:
                    self.per_epoch_train_times.append(per_batch_train_time)
                    self.per_batch_train_times.append(per_batch_train_time)
                    
                    if num_train_samples > self.cfg.num_epochs * self.cfg.batch_size:
                        self.uniquely_seen_num_of_samples.append(self.cfg.num_epochs * self.cfg.batch_size)
                    elif num_train_samples > self.cfg.batch_size:
                        self.uniquely_seen_num_of_samples.append((num_train_samples // self.cfg.batch_size) * self.cfg.batch_size)
                    else:
                        self.uniquely_seen_num_of_samples.append(num_train_samples)
                    
                    self.trained_num_of_samples.append(self.cfg.batch_size)

                    self.inference_times.append(((num_train_samples-1)//self.cfg.batch_size + 1) * per_batch_train_time * 0.5)
                    self.inference_times_per_sample.append(self.inference_times[-1]/num_train_samples)
                else:
                    self.per_epoch_train_times.append(((num_train_samples-1)//self.cfg.batch_size + 1) * per_batch_train_time)
                    self.per_batch_train_times.append(per_batch_train_time)

                    self.trained_num_of_samples.append(num_train_samples)
                    self.uniquely_seen_num_of_samples.append(num_train_samples)
                    
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
                self.uniquely_seen_num_of_samples.append(num_train_samples)
                
                self.inference_times.append(self.per_epoch_train_times[-1]*0.5)
                self.inference_times_per_sample.append(self.inference_times[-1]/num_train_samples)
        
        if self.cfg.dataset == 'big_reddit':
            self.train_data = self.model.fedbalancer_xy_processing(self.train_data)
            self.eval_data = self.model.fedbalancer_xy_processing(self.eval_data)
        
        self.sampled_per_epoch_train_time = np.mean(self.per_epoch_train_times)

        self.whole_data_loss_list = []
        
        self.is_first_round = True

    def train(self, start_t=None, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
        """
        
        def train_with_simulate_time(self, start_t, num_epochs=1, batch_size=10, minibatch=None):

            if (self.cfg.fedbalancer or self.cfg.fb_client_selection or self.cfg.realoortbalancer or (self.cfg.ss_baseline and self.is_big_client)):
                whole_xs, whole_ys = zip(*list(zip(self.train_data["x"], self.train_data["y"])))
                whole_data = {'x': whole_xs, 'y': whole_ys}

                # if cfg.fb_inference_pipelining is True, it means that FedBalancer is not doing a full pass on the client's data 
                # except when a client is first selected for a round...
                # otherwise, FedBalancer do full pass on the client's data at the start of training at a round
                if self.cfg.fb_inference_pipelining or (self.cfg.ss_baseline and self.is_big_client):
                    if self.is_first_round:
                        whole_data_loss_list_saved = self.model.test(whole_data)['loss_list']
                        whole_data_loss_list = whole_data_loss_list_saved
                        self.whole_data_loss_list = whole_data_loss_list_saved
                    else:
                        whole_data_loss_list = self.whole_data_loss_list
                else:
                    whole_data_loss_list_saved = self.model.test(whole_data)['loss_list']
                    whole_data_loss_list = whole_data_loss_list_saved

            user_whole_data_len = 0
            fedbalancer_data_len = 0
            uniquely_seen_data_cnt = 0
            tmp_data = []
            sorted_loss = []
            ne=-1

            # Sample selection on a client at a FL round with loss_threshold
            if minibatch is None:

                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
                user_whole_data_len = num_data
                if self.cfg.fedbalancer:

                    data_len = len(whole_data_loss_list)
                    tmp_data = zip(self.train_data["x"], self.train_data["y"])
                    tmp_data = zip(tmp_data, range(len(self.train_data["x"])))
                    tmp_data = zip(whole_data_loss_list, tmp_data)
                    tmp_data = sorted(tmp_data, reverse=False, key=lambda elem: elem[0])

                    for i, item in enumerate(tmp_data):
                        if item[0] >= self.loss_threshold:
                            break

                    j = len(tmp_data)
                    
                    sorted_loss = [x for x,_ in tmp_data]
                    self.sorted_loss = sorted_loss

                    train_time_per_batch_mean = np.mean(self.train_time_per_batch_list)

                    # If a client is fast enough to train its whole data for num_epochs, then just include all data without selection
                    if len(self.train_time_per_batch_list) > 0 and self.deadline > num_epochs * ((num_data-1)//batch_size+1) * train_time_per_batch_mean:
                        tmp_data_pkg = [x for _,x in tmp_data]
                        tmp_data = [x for x,_ in tmp_data_pkg]
                        tmp_data_idx = [x for _,x in tmp_data_pkg]
                    # Else If a client is fast enough to train its whole data that is over the loss_threshold for num_epochs, the client will select max trainable number of samples
                    elif len(self.train_time_per_batch_list) > 0 and self.deadline > num_epochs * ((j-i-1)//batch_size+1) * train_time_per_batch_mean:
                        data_cnt = min(int((self.deadline / (num_epochs * train_time_per_batch_mean)) * batch_size), len(tmp_data)) # Measuring max trainiable number of samples
                        easy_data_cnt = int(data_cnt * self.cfg.fb_p) # SAMPLE P FROM EASY DATA, which is UT
                        hard_data_cnt = data_cnt - easy_data_cnt # SAMPLE 1-P FROM HARD DATA, which is OT
                        
                        if easy_data_cnt > i:
                            easy_data_cnt = i
                            hard_data_cnt = data_cnt - easy_data_cnt
                        elif hard_data_cnt > (j-i):
                            hard_data_cnt = (j-i)
                            easy_data_cnt = data_cnt - hard_data_cnt

                        easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                        easy_data = [x for x,_ in easy_data_pkg]
                        easy_data_idx = [x for _,x in easy_data_pkg]

                        hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]
                        hard_data = [x for x,_ in hard_data_pkg]
                        hard_data_idx = [x for _,x in hard_data_pkg]

                        tmp_data = easy_data + hard_data
                        tmp_data_idx = easy_data_idx + hard_data_idx
                    # Otherwise, the client will select samples as much as the number of samples that are over loss_threshold (i.e., length of OT)
                    else:
                        easy_data_cnt = int((j - i) * self.cfg.fb_p) # SAMPLE P FROM EASY DATA, which is UT
                        hard_data_cnt = (j - i) - easy_data_cnt # SAMPLE 1-P FROM HARD DATA, which is OT

                        if easy_data_cnt > i:
                            easy_data_cnt = i
                            hard_data_cnt = (j - i) - easy_data_cnt

                        easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                        easy_data = [x for x,_ in easy_data_pkg]
                        easy_data_idx = [x for _,x in easy_data_pkg]
                        hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]
                        hard_data = [x for x,_ in hard_data_pkg]
                        hard_data_idx = [x for _,x in hard_data_pkg]

                        tmp_data = easy_data + hard_data
                        tmp_data_idx = easy_data_idx + hard_data_idx

                    fedbalancer_data_len = len(tmp_data)
                    num_data = fedbalancer_data_len
                    data_portion = fedbalancer_data_len/data_len
                # In case of OortBalancer, samples will be selected later
                elif self.cfg.realoortbalancer:
                    data_len = len(whole_data_loss_list)
                    tmp_data = zip(self.train_data["x"], self.train_data["y"])
                    tmp_data = zip(tmp_data, range(len(self.train_data["x"])))
                    tmp_data = zip(whole_data_loss_list, tmp_data)
                    tmp_data = sorted(tmp_data, reverse=False, key=lambda elem: elem[0])

                    for i, item in enumerate(tmp_data):
                        if item[0] >= self.loss_threshold:
                            break
                    j = len(tmp_data)
                    # print(self.loss_threshold, i, j)
                elif self.cfg.fb_client_selection:
                    sorted_loss = list(sorted(whole_data_loss_list, reverse=False))
                    self.sorted_loss = sorted_loss
                elif (self.cfg.ss_baseline and self.is_big_client):
                    data_len = self.select_sample_num
                    tmp_data = zip(self.train_data["x"], self.train_data["y"])
                    tmp_data = zip(tmp_data, range(len(self.train_data["x"])))
                    tmp_data = zip(whole_data_loss_list, tmp_data)
                    tmp_data = sorted(tmp_data, reverse=True, key=lambda elem: elem[0])

                    tmp_data_pkg = [x for _,x in tmp_data[:data_len]]
                    tmp_data = [x for x,_ in tmp_data_pkg]
                    tmp_data_idx = [x for _,x in tmp_data_pkg]

                    num_data = self.select_sample_num
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
            
            train_time, train_time_per_batch, train_time_per_epoch = self.device.get_train_time_and_train_time_per_batch_and_train_time_per_epoch(num_data, batch_size, num_epochs)
            self.train_time_per_batch_list.append(train_time_per_batch)

            logger.debug('client {}: num data:{}'.format(self.id, num_data))
            logger.debug('client {}: train time:{}'.format(self.id, train_time))
            
            if minibatch is None:
                num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
                if self.cfg.fedbalancer:
                    num_data = fedbalancer_data_len
                    xs, ys = zip(*tmp_data)
                    data_idx = tmp_data_idx
                elif (self.cfg.ss_baseline and self.is_big_client):
                    num_data = self.select_sample_num
                    xs, ys = zip(*tmp_data)
                    data_idx = tmp_data_idx
                # Oort randomly selects a batch per epoch for training at this round
                # Oort selects same samples multiple times if the client has less data than batch_size * num_epoch
                elif self.cfg.realoort:
                    num_data = batch_size
                    if len(self.train_data["x"]) >= num_data * num_epochs:
                        xss, yss = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data*num_epochs))
                    elif len(self.train_data["x"]) >= num_data:
                        xss = []
                        yss = []
                        nb = len(self.train_data["x"]) // num_data
                        sampled_batch_cnt = 0
                        while sampled_batch_cnt != num_epochs:
                            this_iteration_sample_batch_cnt = min(num_epochs - sampled_batch_cnt, nb)
                            xsss, ysss = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data*this_iteration_sample_batch_cnt))
                            xss += xsss
                            yss += ysss
                            sampled_batch_cnt += this_iteration_sample_batch_cnt
                    else:
                        xss = []
                        yss = []
                        for epoch_idx in range(num_epochs):
                            xsss, ysss = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), len(self.train_data["x"])))
                            xss += xsss
                            yss += ysss
                    xs = xss[:min(batch_size, len(self.train_data["x"]))]
                    ys = yss[:min(batch_size, len(self.train_data["x"]))]
                    realoort_whole_data = {'x': xs, 'y': ys}
                    realoort_whole_data_loss_list = self.model.test(realoort_whole_data)['loss_list']
                    sorted_loss = list(realoort_whole_data_loss_list)
                    self.sorted_loss = sorted_loss
                    data_idx = range(len(ys))

                # OortBalancer selects a batch per epoch based on how FedBalancer selects samples; only the number of selected samples are fixed as batch_size * num_epoch
                # Same as Oort, OortBalancer selects same samples multiple times if the client has less data than batch_size * num_epoch

                elif self.cfg.realoortbalancer:
                    num_data = batch_size
                    uniquely_seen_data_cnt = 0 # for required inference time calculation
                    if len(self.train_data["x"]) >= num_data * num_epochs:
                        easy_data_cnt = int((num_data) * num_epochs * self.cfg.fb_p)
                        hard_data_cnt = num_data * num_epochs - easy_data_cnt

                        if easy_data_cnt > i:
                            easy_data_cnt = i
                            hard_data_cnt = num_data * num_epochs - easy_data_cnt
                        
                        elif hard_data_cnt > j - i:
                            hard_data_cnt = j - i
                            easy_data_cnt = num_data * num_epochs - hard_data_cnt
                        
                        easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                        hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]

                        added_data_pkg = easy_data_pkg + hard_data_pkg
                        
                        np.random.shuffle(added_data_pkg)
                        added_data = [x for x,_ in added_data_pkg]
                        added_data_idx = [x for _,x in added_data_pkg]
                        xss, yss = zip(*added_data)
                        
                        uniquely_seen_data_cnt = num_data * num_epochs

                    elif len(self.train_data["x"]) >= num_data:
                        xss = []
                        yss = []
                        nb = len(self.train_data["x"]) // num_data
                        sampled_batch_cnt = 0
                        added_data_idx = []
                        while sampled_batch_cnt != num_epochs:
                            #print(len(tmp_data))
                            this_iteration_sample_batch_cnt = min(num_epochs - sampled_batch_cnt, nb)

                            easy_data_cnt = int((num_data) * this_iteration_sample_batch_cnt * self.cfg.fb_p)
                            hard_data_cnt = num_data * this_iteration_sample_batch_cnt - easy_data_cnt

                            if easy_data_cnt > i:
                                easy_data_cnt = i
                                hard_data_cnt = num_data * this_iteration_sample_batch_cnt - easy_data_cnt
                            elif hard_data_cnt > j - i:
                                hard_data_cnt = j - i
                                easy_data_cnt = num_data * this_iteration_sample_batch_cnt - hard_data_cnt
                            
                            easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                            hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]

                            added_data_pkg = easy_data_pkg + hard_data_pkg
                            
                            #print("TMP", tmp_data)
                            np.random.shuffle(added_data_pkg)

                            added_data = [x for x,_ in added_data_pkg]
                            added_data_idx += [x for _,x in added_data_pkg]

                            xsss, ysss = zip(*added_data)
                            xss += xsss
                            yss += ysss
                            sampled_batch_cnt += this_iteration_sample_batch_cnt

                            if uniquely_seen_data_cnt == 0:
                                uniquely_seen_data_cnt = this_iteration_sample_batch_cnt * num_data
                    else:
                        xss = []
                        yss = []
                        for epoch_idx in range(num_epochs):
                            easy_data_cnt = int(len(self.train_data["x"]) * self.cfg.fb_p)
                            hard_data_cnt = len(self.train_data["x"]) - easy_data_cnt

                            if easy_data_cnt > i:
                                easy_data_cnt = i
                                hard_data_cnt = len(self.train_data["x"]) - easy_data_cnt
                            elif hard_data_cnt > j - i:
                                hard_data_cnt = j - i
                                easy_data_cnt = len(self.train_data["x"]) - hard_data_cnt
                            
                            easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                            hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]

                            added_data_pkg = easy_data_pkg + hard_data_pkg

                            np.random.shuffle(added_data_pkg)
                            added_data = [x for x,_ in added_data_pkg]
                            added_data_idx = [x for _,x in added_data_pkg]

                            xsss, ysss = zip(*added_data)
                            xss += xsss
                            yss += ysss
                        uniquely_seen_data_cnt = len(self.train_data["x"])

                    xs = xss[:min(batch_size, len(self.train_data["x"]))]
                    ys = yss[:min(batch_size, len(self.train_data["x"]))]
                    realoort_whole_data = {'x': xs, 'y': ys}
                    realoort_whole_data_loss_list = self.model.test(realoort_whole_data)['loss_list']
                    sorted_loss = list(realoort_whole_data_loss_list)
                    self.sorted_loss = sorted_loss
                    
                    data_idx = added_data_idx
                else:
                    xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                    data_idx = list(range(len(ys)))
                data = {'x': xs, 'y': ys}
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}
                data_idx = list(range(len(ys)))
            
            # Calculate the time that elapses for full data inferencing at FedBalancer, if fb_inference_pipelining is False
            # We recommend to let fb_inference_pipelining as True to enjoy better time-to-accuracy performance
            inference_time = 0
            
            if self.cfg.fedbalancer:
                if fedbalancer_data_len == user_whole_data_len:
                    inference_time = 0
                elif self.cfg.fb_inference_pipelining and not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            elif (self.cfg.ss_baseline and self.is_big_client):
                #inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len - fedbalancer_data_len, batch_size, 1)
                if not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            elif self.cfg.realoortbalancer:
                if self.cfg.fb_inference_pipelining and not self.is_first_round:
                    inference_time = 0
                else:
                    inference_time, _ = self.device.get_train_time_and_train_time_per_batch(user_whole_data_len, batch_size, 1)
                    inference_time = inference_time * 0.5
            
            if self.is_first_round:
                self.is_first_round = False

            download_time = self.device.get_download_time()
            upload_time = self.device.get_upload_time(self.model.size) # will be re-calculated after training

            self.act_inference_time = 0
            self.ori_inference_time = 0

            if self.cfg.fedbalancer or self.cfg.realoortbalancer or (self.cfg.ss_baseline and self.is_big_client):
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
                if not (self.cfg.fedbalancer or self.cfg.realoortbalancer or (self.cfg.ss_baseline and self.is_big_client)):
                    if (down_end_time-start_t) > self.deadline:
                        # download too long
                        self.actual_comp = 0.0
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
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                        self.update_size = 0
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.realoortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(down_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.realoortbalancer) and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when training'
                                self.sorted_loss = []
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    if not self.cfg.realoortbalancer:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                    else:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                if not self.cfg.realoortbalancer:
                                    self.uniquely_seen_num_of_samples.append(len(data['x']))
                                else:
                                    self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when training'
                            self.sorted_loss = []
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                if not self.cfg.realoortbalancer:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                else:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            if not self.cfg.realoortbalancer:
                                self.uniquely_seen_num_of_samples.append(len(data['x']))
                            else:
                                self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (up_end_time-start_t) > self.deadline:
                        self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.realoortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(down_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.realoortbalancer) and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when uploading'
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    if not self.cfg.realoortbalancer:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                    else:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                if not self.cfg.realoortbalancer:
                                    self.uniquely_seen_num_of_samples.append(len(data['x']))
                                else:
                                    self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                                self.per_epoch_train_times.append(train_time_per_epoch)
                                self.per_batch_train_times.append(train_time_per_batch)
                                if not self.cfg.realoortbalancer:
                                    self.trained_num_of_samples.append(len(data['x']))
                                self.sorted_loss = []
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when uploading'
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                if not self.cfg.realoortbalancer:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                else:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            if not self.cfg.realoortbalancer:
                                self.uniquely_seen_num_of_samples.append(len(data['x']))
                            else:
                                self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                            self.per_epoch_train_times.append(train_time_per_epoch)
                            self.per_batch_train_times.append(train_time_per_batch)
                            if not self.cfg.realoortbalancer:
                                self.trained_num_of_samples.append(len(data['x']))
                            self.sorted_loss = []
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    else:
                        if minibatch is None:
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            else:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                                logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
                        else:
                            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                            num_epochs = 1
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, num_data)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            else:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                                logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
                else: # fedbalancer or realoortbalancer or (self.cfg.ss_baseline and self.is_big_client)
                    if (down_end_time-start_t) > self.deadline:
                        # download too long
                        self.actual_comp = 0.0
                        self.update_size = 0
                        failed_reason = 'failed when downloading'
                        self.sorted_loss = []
                        raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (inference_end_time-start_t) > self.deadline:
                        # download too long
                        self.actual_comp = 0.0
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
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        self.actual_comp = int(comp*available_time/train_time)    # will be used in get_actual_comp
                        self.update_size = 0
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.realoortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(inference_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.realoort:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.realoortbalancer) and ne != -1:
                                if self.cfg.realoort:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when training'
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    if not self.cfg.realoortbalancer:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                    else:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                if not self.cfg.realoortbalancer:
                                    self.uniquely_seen_num_of_samples.append(len(data['x']))
                                else:
                                    self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
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
                                if not self.cfg.realoortbalancer:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                else:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            if not self.cfg.realoortbalancer:
                                self.uniquely_seen_num_of_samples.append(len(data['x']))
                            else:
                                self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    elif (up_end_time-start_t) > self.deadline:
                        self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                        if self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.realoortbalancer:
                            ne = -1
                            for i in range(1, num_epochs):
                                et = self.timer.get_future_time(inference_end_time, train_time*i/num_epochs + upload_time)
                                if et - start_t <= self.deadline:
                                    ne = i
                            #print(down_end_time, train_time*ne/num_epochs + upload_time, et, et-start_t, self.deadline)
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            elif self.cfg.fedprox and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            elif (self.cfg.fedbalancer or self.cfg.realoortbalancer) and ne != -1:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, ne, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, ne, batch_size)
                                train_time *= ne / num_epochs
                                logger.debug("client {} train-epochs={}".format(self.id, ne))
                            else:
                                failed_reason = 'failed when uploading'
                                if (self.act_download_time + self.act_inference_time) > self.deadline:
                                    self.sorted_loss = []
                                #self.download_times.append(download_time)
                                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                    self.inference_times.append(inference_time)
                                    if not self.cfg.realoortbalancer:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                    else:
                                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                if not self.cfg.realoortbalancer:
                                    self.uniquely_seen_num_of_samples.append(len(data['x']))
                                else:
                                    self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                                self.per_epoch_train_times.append(train_time_per_epoch)
                                self.per_batch_train_times.append(train_time_per_batch)
                                if not self.cfg.realoortbalancer:
                                    self.trained_num_of_samples.append(len(data['x']))
                                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                        else:
                            failed_reason = 'failed when uploading'
                            if (self.act_download_time + self.act_inference_time) > self.deadline:
                                self.sorted_loss = []
                            #self.download_times.append(download_time)
                            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                                self.inference_times.append(inference_time)
                                if not self.cfg.realoortbalancer:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                                else:
                                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                            if not self.cfg.realoortbalancer:
                                self.uniquely_seen_num_of_samples.append(len(data['x']))
                            else:
                                self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                            self.per_epoch_train_times.append(train_time_per_epoch)
                            self.per_batch_train_times.append(train_time_per_batch)
                            if not self.cfg.realoortbalancer:
                                self.trained_num_of_samples.append(len(data['x']))
                            raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)
                    else :
                        if minibatch is None:
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, batch_size)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            else:
                                if self.cfg.realoort or self.cfg.realoortbalancer:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                                logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
                        else:
                            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                            num_epochs = 1
                            if self.cfg.no_training:
                                comp = self.model.get_comp(data, num_epochs, num_data)
                                update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                            else:
                                if self.cfg.realoort:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                                else:
                                    comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                                logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
            else: # oort_pacer or ddl_baseline_smartpc
                if minibatch is None:
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, batch_size)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    else:
                        if self.cfg.realoort or self.cfg.realoortbalancer:
                            comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                        else:
                            comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
                        logger.debug("client {} train-epochs={}".format(self.id, num_epochs))
                else:
                    # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                    num_epochs = 1
                    if self.cfg.no_training:
                        comp = self.model.get_comp(data, num_epochs, num_data)
                        update, acc, loss, grad, loss_old = -1,-1,-1,-1,-1
                    else:
                        if self.cfg.realoort or self.cfg.realoortbalancer:
                            comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.realoorttrain(data, data_idx, xss, yss, num_epochs, batch_size)
                        else:
                            comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx = self.model.train(data, data_idx, num_epochs, batch_size)
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
            self.actual_comp = comp

            # gradiant compress and Federated Learning Strategies are mutually-exclusive
            # gradiant compress
            if self.compressor != None and not self.cfg.no_training:
                grad, size_old, size_new = self.compressor.GradientCompress(grad)
                # logger.info('compression ratio: {}'.format(size_new/size_old))
                self.update_size = self.update_size*size_new/size_old
                # re-calculate upload_time
                upload_time = self.device.get_upload_time(self.update_size)
                self.ori_upload_time = upload_time
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                self.act_upload_time = up_end_time-train_end_time

            # Federated Learning Strategies for Improving Communication Efficiency
            seed = None
            shape_old = None
            if self.structured_updater and not self.cfg.no_training:
                seed, shape_old, grad = self.structured_updater.struc_update(grad)
                # logger.info('compression ratio: {}'.format(sum([np.prod(g.shape) for g in grad]) / sum([np.prod(s) for s in shape_old])))
                self.update_size *= sum([np.prod(g.shape) for g in grad]) / sum([np.prod(s) for s in shape_old])
                # logger.info("UPDATE SIZE"+str(self.update_size))
                # re-calculate upload_time
                upload_time = self.device.get_upload_time(self.update_size)
                self.ori_upload_time = upload_time
                up_end_time = self.timer.get_future_time(train_end_time, upload_time)
                self.act_upload_time = up_end_time-train_end_time
            
            # Changing train time for sub-epoch training
            if (self.cfg.fedprox or self.cfg.fedbalancer or self.cfg.realoortbalancer) and ne != -1:
                self.act_train_time = self.act_train_time * ne / num_epochs
            
            total_cost = self.act_download_time + self.act_inference_time + self.act_train_time + self.act_upload_time

            if total_cost > self.deadline and not (self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc):
                # failed when uploading
                self.actual_comp = self.model.get_comp(data, num_epochs, batch_size)
                failed_reason = 'failed when uploading'
                # Note that, to simplify, we did not change the update_size here, actually the actual update size is less.
                if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                    self.inference_times.append(inference_time)
                    if not self.cfg.realoortbalancer:
                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                    else:
                        self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                if not self.cfg.realoortbalancer:
                    self.uniquely_seen_num_of_samples.append(len(data['x']))
                else:
                    self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
                self.per_epoch_train_times.append(train_time_per_epoch)
                self.per_batch_train_times.append(train_time_per_batch)
                if not self.cfg.realoortbalancer:
                    self.trained_num_of_samples.append(len(data['x']))

                if (self.act_download_time + self.act_inference_time) > self.deadline:
                    self.sorted_loss = []
                    sorted_loss = []
                raise timeout_decorator.timeout_decorator.TimeoutError(failed_reason)

            if inference_time != 0 and not self.cfg.fb_inference_pipelining:
                self.inference_times.append(inference_time)
                if not self.cfg.realoortbalancer:
                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
                else:
                    self.inference_times_per_sample.append(self.inference_times[-1]/(user_whole_data_len))
            if not self.cfg.realoortbalancer:
                self.uniquely_seen_num_of_samples.append(len(data['x']))
            else:
                self.uniquely_seen_num_of_samples.append(uniquely_seen_data_cnt)
            self.per_epoch_train_times.append(train_time_per_epoch)
            self.per_batch_train_times.append(train_time_per_batch)
            if not self.cfg.realoortbalancer:
                self.trained_num_of_samples.append(len(data['x']))

            if ne == -1:
                return simulate_time_c, comp, num_train_samples, update, acc, loss, grad, self.update_size, seed, shape_old, loss_old, sorted_loss, download_time, upload_time, train_time, inference_time, num_epochs
            else:
                return simulate_time_c, comp, num_train_samples, update, acc, loss, grad, self.update_size, seed, shape_old, loss_old, sorted_loss, download_time, upload_time, train_time, inference_time, ne

        return train_with_simulate_time(self, start_t, num_epochs, batch_size, minibatch)


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
    
    def set_loss_threshold(self, loss_threshold = -1, upper_loss_threshold = -1):
        #if deadline < 0 or not self.cfg.behav_hete:
        if loss_threshold < 0:
            self.loss_threshold = 0
        else:
            self.loss_threshold = loss_threshold
        if upper_loss_threshold < 0:
            self.upper_loss_threshold = 0
        else:
            self.upper_loss_threshold = upper_loss_threshold
        logger.debug('client {}\'s loss threshold is set to {} and upper loss threshold is set to {}'.format(self.id, self.loss_threshold, self.upper_loss_threshold))
    

    def upload_suc(self, start_t, num_epochs=1, batch_size=10, minibatch=None):
        """Test if this client will upload successfully

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
            start_t: strat time of the training, only used in train_with_simulate_time
        Return:
            result: test result(True or False)
        """
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
        else :
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
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
        if minibatch is None:
            num_data = min(len(self.train_data["x"]), self.cfg.max_sample)
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac*len(self.train_data["x"])))
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
        
    def get_actual_comp(self):
        '''
        get the actual computation in the training process
        '''
        return self.actual_comp
