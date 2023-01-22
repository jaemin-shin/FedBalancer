import numpy as np
import timeout_decorator
import traceback
from utils.logger import Logger
import json
import random
import math
import sys
import copy
import torch

L = Logger()
logger = L.get_logger()

class FedBalancer:
    def __init__(self, fb_inference_pipelining, fb_p, fb_w, fb_simple_control_lt_stepsize, fb_simple_control_ddl_stepsize):
        self.fb_inference_pipelining = fb_inference_pipelining
        self.fb_p = fb_p
        self.fb_w = fb_w
        self.fb_simple_control_lt_stepsize = fb_simple_control_lt_stepsize
        self.fb_simple_control_ddl_stepsize = fb_simple_control_ddl_stepsize

        self.loss_threshold_ratio = 0.0
        self.loss_threshold = 0
        self.upper_loss_threshold = 0 

        self.deadline_ratio = 1.0

        self.guard_time = 0

        self.if_any_client_sent_response_for_current_round = False

        self.if_any_client_sent_response_for_current_round = False
        self.current_round_loss_min = []
        self.current_round_loss_max = []

        self.prev_train_losses = []

    
    def filter_if_more_than_ratio_is_explored(self, client_one_epoch_train_times, ratio):
        client_time_values = list(client_one_epoch_train_times)
        client_num = len(client_time_values)
        return client_time_values.count(-1) < client_num * (1 - ratio)
    
    def filter_if_more_than_number_is_explored(self, client_one_epoch_train_times, number):
        client_time_values = list(client_one_epoch_train_times)
        client_num = len(client_time_values)
        return (client_num - client_time_values.count(-1)) > number
    
    def preprocess_data_x(self,data):
        return torch.tensor(data,requires_grad=True)
    def preprocess_data_y(self,data):
        data_y=[]
        for i in data:
            data_float=float(i)
            data_y.append(data_float)
        return torch.tensor(data_y,requires_grad=True)

    def calculate_loss_on_whole_dataset_with_inference(self, train_data, model):
        whole_xs, whole_ys = zip(*list(zip(train_data["x"], train_data["y"])))
        whole_data = {'x': whole_xs, 'y': whole_ys}
        whole_data={"x": self.preprocess_data_x(whole_data["x"]),
            "y": self.preprocess_data_y(whole_data["y"])}
        return model.test(whole_data)['loss_list']
    
    # Algorithm 1 from the FedBalancer paper
    def fb_sample_selection(self, num_data, loss_threshold, whole_data_loss_list, train_data, deadline, train_time_per_batch_list, num_epochs, batch_size):
        data_len = len(whole_data_loss_list)
        tmp_data = zip(train_data["x"], train_data["y"])
        tmp_data = zip(tmp_data, range(len(train_data["x"])))
        tmp_data = zip(whole_data_loss_list, tmp_data)

        # sort samples with its loss in ascending order
        tmp_data = sorted(tmp_data, reverse=False, key=lambda elem: elem[0])

        # let i indicate the index which samples with bigger index has loss bigger than loss threshold
        for i, item in enumerate(tmp_data):
            if item[0] >= loss_threshold:
                break

        j = len(tmp_data)
        
        sorted_loss = [x for x,_ in tmp_data]
        train_time_per_batch_mean = np.mean(train_time_per_batch_list)

        # If a client is fast enough to train its whole data for num_epochs, then just include all data without selection
        if len(train_time_per_batch_list) > 0 and deadline > num_epochs * ((num_data-1)//batch_size+1) * train_time_per_batch_mean:
            tmp_data_pkg = [x for _,x in tmp_data]
            tmp_data = [x for x,_ in tmp_data_pkg]
            tmp_data_idx = [x for _,x in tmp_data_pkg]
        # Else If a client is fast enough to train its whole data that is over the loss_threshold for num_epochs, the client will select max trainable number of samples
        elif len(train_time_per_batch_list) > 0 and deadline > num_epochs * ((j-i-1)//batch_size+1) * train_time_per_batch_mean:
            data_cnt = min(int((deadline / (num_epochs * train_time_per_batch_mean)) * batch_size), len(tmp_data)) # Measuring max trainiable number of samples

            easy_data_cnt = int(data_cnt * self.fb_p) # SAMPLE P FROM EASY DATA, which is UT
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
            easy_data_cnt = int((j - i) * self.fb_p) # SAMPLE P FROM EASY DATA, which is UT
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

        num_data = fedbalancer_data_len
        xs, ys = zip(*tmp_data)
        data_idx = tmp_data_idx

        selected_data = {'x': xs, 'y': ys}
        selected_data={"x": self.preprocess_data_x(selected_data["x"]),
            "y": self.preprocess_data_y(selected_data["y"])}

        return selected_data, num_data, data_idx, sorted_loss
    
    # Algorithm 1, Oort + FedBalancer (oortbalancer) version
    def fb_oortbalancer_sample_selection(self, batch_size, loss_threshold, whole_data_loss_list, train_data, deadline, train_time_per_batch_list, num_epochs, model):
        data_len = len(whole_data_loss_list)
        tmp_data = zip(train_data["x"], train_data["y"])
        tmp_data = zip(tmp_data, range(len(train_data["x"])))
        tmp_data = zip(whole_data_loss_list, tmp_data)
        tmp_data = sorted(tmp_data, reverse=False, key=lambda elem: elem[0])

        for i, item in enumerate(tmp_data):
            if item[0] >= loss_threshold:
                break
        j = len(tmp_data)

        num_data = batch_size

        # OortBalancer selects a batch per epoch based on how FedBalancer selects samples; only the number of selected samples are fixed as batch_size * num_epoch
        # Same as Oort, OortBalancer selects same samples multiple times if the client has less data than batch_size * num_epoch
        if len(train_data["x"]) >= num_data * num_epochs:
            easy_data_cnt = int((num_data) * num_epochs * self.fb_p)
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

        elif len(train_data["x"]) >= num_data:
            xss = []
            yss = []
            nb = len(train_data["x"]) // num_data
            sampled_batch_cnt = 0
            added_data_idx = []
            while sampled_batch_cnt != num_epochs:
                this_iteration_sample_batch_cnt = min(num_epochs - sampled_batch_cnt, nb)

                easy_data_cnt = int((num_data) * this_iteration_sample_batch_cnt * self.fb_p)
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
                
                np.random.shuffle(added_data_pkg)

                added_data = [x for x,_ in added_data_pkg]
                added_data_idx += [x for _,x in added_data_pkg]

                xsss, ysss = zip(*added_data)
                xss += xsss
                yss += ysss
                sampled_batch_cnt += this_iteration_sample_batch_cnt
        else:
            xss = []
            yss = []
            for epoch_idx in range(num_epochs):
                easy_data_cnt = int(len(train_data["x"]) * self.fb_p)
                hard_data_cnt = len(train_data["x"]) - easy_data_cnt

                if easy_data_cnt > i:
                    easy_data_cnt = i
                    hard_data_cnt = len(train_data["x"]) - easy_data_cnt
                elif hard_data_cnt > j - i:
                    hard_data_cnt = j - i
                    easy_data_cnt = len(train_data["x"]) - hard_data_cnt
                
                easy_data_pkg = [x for _,x in random.sample(tmp_data[:i], easy_data_cnt)]
                hard_data_pkg = [x for _,x in random.sample(tmp_data[i:j], hard_data_cnt)]

                added_data_pkg = easy_data_pkg + hard_data_pkg

                np.random.shuffle(added_data_pkg)
                added_data = [x for x,_ in added_data_pkg]
                added_data_idx = [x for _,x in added_data_pkg]

                xsss, ysss = zip(*added_data)
                xss += xsss
                yss += ysss

        xs = xss[:min(batch_size, len(train_data["x"]))]
        ys = yss[:min(batch_size, len(train_data["x"]))]
        oort_whole_data = {'x': xs, 'y': ys}
        oort_whole_data ={"x": self.preprocess_data_x(oort_whole_data["x"]),
            "y": self.preprocess_data_y(oort_whole_data["y"])}
        oort_whole_data_loss_list = model.test(oort_whole_data)['loss_list']
        sorted_loss = list(oort_whole_data_loss_list)
        
        data_idx = added_data_idx

        xss = self.preprocess_data_x(xss)
        yss = self.preprocess_data_y(yss)

        return oort_whole_data, xss, yss, num_data, data_idx, sorted_loss
    
    # Algorithm 2 from the FedBalancer paper
    def loss_threshold_selection(self):
        if self.loss_threshold == 0 and self.loss_threshold_ratio == 0:
            self.loss_threshold = 0
            logger.info('loss_threshold {}'.format(self.loss_threshold))
        else:
            loss_low = np.min(self.current_round_loss_min)
            loss_high = np.mean(self.current_round_loss_max)
            self.loss_threshold = loss_low + (loss_high - loss_low) * self.loss_threshold_ratio
            logger.info('loss_low {}, loss_high {}, loss_threshold {}'.format(loss_low, loss_high, self.loss_threshold))
    
    # Algorithm 3 from the FedBalancer paper
    def ratio_control(self, prev_train_losses, current_round):
        if current_round % int(self.fb_w) == int(self.fb_w) - 1 :
            if len(prev_train_losses) >= 2 * self.fb_w:
                nonscaled_reward = (np.mean(prev_train_losses[-(self.fb_w*2):-(self.fb_w)]) - np.mean(prev_train_losses[-(self.fb_w):]))

                if nonscaled_reward > 0:
                    if self.loss_threshold_ratio + self.fb_simple_control_lt_stepsize <= 1:
                        self.loss_threshold_ratio += self.fb_simple_control_lt_stepsize
                else:
                    if self.loss_threshold_ratio - self.fb_simple_control_lt_stepsize >= 0:
                        self.loss_threshold_ratio -= self.fb_simple_control_lt_stepsize

                if nonscaled_reward > 0:
                    if self.deadline_ratio - self.fb_simple_control_ddl_stepsize >= 0:
                        self.deadline_ratio -= self.fb_simple_control_ddl_stepsize
                else:
                    if self.deadline_ratio + self.fb_simple_control_ddl_stepsize <= 1:
                        self.deadline_ratio += self.fb_simple_control_ddl_stepsize
    
    # Algorithm 4 from the FedBalancer paper
    def deadline_selection(self, selected_clients, clients_info, num_epochs, deadline, batch_size, current_round):
        # Deadline is only updated when the round completion time of more than half clients (or more than 200 clients) are explored
        if (self.filter_if_more_than_ratio_is_explored([clients_info[str(cid)]["one_epoch_train_time"] for cid in clients_info.keys()], 0.5) or self.filter_if_more_than_number_is_explored([clients_info[str(cid)]["one_epoch_train_time"] for cid in clients_info.keys()], 200)):
            deadline_low = self.findPeakDDLE(selected_clients, 1, batch_size, current_round)
            deadline_high = self.findPeakDDLE(selected_clients, num_epochs, batch_size, current_round)
            deadline = deadline_low + (deadline_high - deadline_low) * self.deadline_ratio

            logger.info('deadline_low {}, deadline_high {}, deadline {}'.format(deadline_low, deadline_high, deadline))
        
        return deadline
    
    # Algorithm 4 from the FedBalancer paper
    def findPeakDDLE(self, selected_clients, num_epochs, batch_size, current_round):
        t_max = sys.maxsize
        total_user_count = len(selected_clients)
        
        complete_user_counts_per_time = []
        max_complete_user_counts_per_time = -1
        max_complete_user_counts_per_time_idx = -1

        client_complete_time = {}

        for c in selected_clients:
            if self.fb_inference_pipelining:
                client_complete_time[str(c.id)] = (c.device.get_expected_download_time()) + (c.device.get_expected_upload_time()) + ((np.mean(c.trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c.per_batch_train_times) * num_epochs + self.guard_time
            else:
                client_complete_time[str(c.id)] = (c.device.get_expected_download_time()) + (c.device.get_expected_upload_time()) + np.mean(c.inference_times_per_sample) * (c.num_train_samples) +  ((np.mean(c.trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c.per_batch_train_times) * num_epochs + self.guard_time
        
        for i in range(1, t_max):
            complete_user_count = 0
            for c in selected_clients:
                if client_complete_time[str(c.id)] <= i:
                    complete_user_count += 1
            complete_user_counts_per_time.append(complete_user_count/(i))
            
            if max_complete_user_counts_per_time < complete_user_count/(i):
                max_complete_user_counts_per_time = complete_user_count/(i)
                max_complete_user_counts_per_time_idx = i
            
            if complete_user_count == total_user_count:
                break
            
        return max_complete_user_counts_per_time_idx