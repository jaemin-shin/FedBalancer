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

class Oort:
    def __init__(self, all_clients, deadline, oort_pacer=True, pacer_window=20, pacer_delta=10, oort_blacklist=False, oort_blacklist_rounds=10):
        self.all_clients = all_clients
        
        self.oort_pacer = oort_pacer
        self.pacer_window = pacer_window
        self.pacer_delta = pacer_delta

        self.oort_blacklist = oort_blacklist
        self.oort_blacklist_rounds = oort_blacklist_rounds

        # Oort client selection variables
        self.epsilon = 0.9 # time-based exploration factor introduced in Oort, which decreases by a factor 0.98 after each round when it is larger than 0.2

        self.system_utility_penalty_alpha = 8
        self.round_exploited_utility = []

        self.curr_round_exploited_utility = 0.0

        # This is a deadline only for client selection algorithm of Oort (Fan Lai et al., OSDI'21)
        self.oort_non_pacer_deadline = deadline

        if self.oort_pacer:
            self.system_utility_penalty_alpha = 2
    
    def pacer_deadline_update(self, deadline, current_round):
        # oort pacer updates the deadline based on the utility of past rounds
        if self.oort_pacer:
            if current_round > self.pacer_window * 2:
                if sum(self.round_exploited_utility[-(2*self.pacer_window):-(self.pacer_window)]) > sum(self.round_exploited_utility[-(self.pacer_window):]):
                    logger.info('by oort pacer, the deadline is changed from {} to {}'.format(deadline, deadline+self.pacer_delta))
                    deadline += self.pacer_delta
        return deadline

    def update_epsilon(self):
        if self.epsilon > 0.2:
            logger.info('epsilon {}'.format(self.epsilon))
            self.epsilon = self.epsilon * 0.98
    
    def zero_epsilon(self, all_clients, clients_info):
        is_everyone_explored = True
        for value in [clients_info[str(c.id)]["one_epoch_train_time"] for c in self.all_clients]:
            if value == -1:
                is_everyone_explored = False
                break
        if is_everyone_explored:
            self.epsilon = 0.0
            logger.info('epsilon {}'.format(self.epsilon))
    
    def select_clients(self, all_clients, possible_clients, num_clients, clients_info, current_round, deadline, batch_size, num_epochs, behav_hete = False, fb_client_selection = False, fb_inference_pipelining = False, oortbalancer = False):
        # if oort_blacklist is on, add clients that are selected for more rounds than a threshold 
        # to the blacklist_clients list
        if self.oort_blacklist:
            blacklist_clients = []
            for c in all_clients:
                if clients_info[str(c.id)]["selected_count"] > self.oort_blacklist_rounds:
                    blacklist_clients.append(str(c.id))
        
        possible_clients_ids = []
        if behav_hete:
            for p_c_ in possible_clients:
                possible_clients_ids.append(str(p_c_.id))

        c_id_and_overthreshold_loss = []
        overthreshold_loss_list = []
        c_id_to_client_object = {}

        # Sort client's recorded loss that is over current loss_threshold

        # Calculate the utility of each clients, based on Oort paper (Fan Lai et al., OSDI 2021)
        # If FedBalancer, only the sample loss above loss_threshold is calculated for utility
        # Otherwise, all samples are calculated (this is possible because loss_threshold = 0)
        for c in all_clients:
            if self.oort_blacklist:
                if str(c.id) in blacklist_clients:
                    continue
            summ = 0
            
            overthreshold_loss_count = clients_info[str(c.id)]["overthreshold_loss_count"]
            summ = clients_info[str(c.id)]["overthreshold_loss_sum"]

            if overthreshold_loss_count == -1 or overthreshold_loss_count == 0:
                summ = 0
            else:
                if fb_client_selection:
                    summ = math.sqrt(summ / overthreshold_loss_count) * overthreshold_loss_count
                else:
                    summ = math.sqrt(summ / batch_size) * batch_size

            c_id_and_overthreshold_loss.append((str(c.id), summ))
            overthreshold_loss_list.append(summ)
            clients_info[str(c.id)]["utility"] = summ
            c_id_to_client_object[str(c.id)] = c
        
        # Calculate the clip utility of the utility(loss sum) distribution, by 95% value
        overthreshold_loss_list.sort()
        clip_value = overthreshold_loss_list[min(int(len(overthreshold_loss_list)*0.95), len(overthreshold_loss_list)-1)]

        # Add incentive term for clients that have been overlooked for a long time
        for tmp_idx in range(len(c_id_and_overthreshold_loss)):
            c_id = c_id_and_overthreshold_loss[tmp_idx][0]
            summ = min(c_id_and_overthreshold_loss[tmp_idx][1], clip_value)
            if clients_info[str(c_id)]["last_selected_round"] != -1:
                summ += math.sqrt(0.1*math.log(current_round + 1)/(clients_info[str(c_id)]["last_selected_round"]+1)) #To avoid zero division, we regard the training round starts from 1 only at this equation
            if self.oort_pacer:
                if clients_info[str(c_id)]["last_selected_round_duration"] != -1 and clients_info[str(c_id)]["last_selected_round_duration"] > deadline:
                    summ *= math.pow(deadline / clients_info[str(c_id)]["last_selected_round_duration"], self.system_utility_penalty_alpha)
            else:
                if fb_client_selection and fb_inference_pipelining:
                # if fb_inference_pipelining:
                    client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * num_epochs
                else:
                    client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) + np.mean(c_id_to_client_object[c_id].inference_times_per_sample) * (c_id_to_client_object[c_id].num_train_samples) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * num_epochs
                    
                if client_complete_time > self.oort_non_pacer_deadline:
                    summ *= math.pow(self.oort_non_pacer_deadline / client_complete_time, self.system_utility_penalty_alpha)
            c_id_and_overthreshold_loss[tmp_idx] = (c_id, summ)
        
        sorted_c_id_and_overthreshold_loss = sorted(c_id_and_overthreshold_loss, key=lambda tup: tup[1], reverse=True)

        # Sample clients from 1-epsilon, prioritizing statistical utility
        cutoff_loss = 0.95*(sorted_c_id_and_overthreshold_loss[int(num_clients*(1-self.epsilon))][1])

        # Perform random client selection if cutoff_loss == 0
        if cutoff_loss == 0:
            selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
            for c in selected_clients:
                clients_info[str(c.id)]["last_selected_round"] = current_round
            return selected_clients, clients_info
        
        c_id_over_cutoff_loss_ids = []
        c_id_over_cutoff_loss_probs = []
        c_id_over_cutoff_loss_sum = 0

        c_id_less_cutoff_loss_ids = []
        
        # Divide clients based on the cutoff_loss
        for item in sorted_c_id_and_overthreshold_loss:
            if item[1] >= cutoff_loss:
                c_id_over_cutoff_loss_ids.append(item[0])
                c_id_over_cutoff_loss_probs.append(item[1])
                c_id_over_cutoff_loss_sum += item[1]
            else:
                c_id_less_cutoff_loss_ids.append(item[0])
        # Pick each clients based on the probability based on the utility divided by the utility sum
        for probs_idx in range(len(c_id_over_cutoff_loss_probs)):
            c_id_over_cutoff_loss_probs[probs_idx] /= c_id_over_cutoff_loss_sum

        selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, int(num_clients*(1-self.epsilon)), replace=False, p=c_id_over_cutoff_loss_probs)
        
        # Sample clients from epsilon, which have less utility than the cutoff_loss
        # Prioritize clients which have faster speed;
        if self.epsilon > 0.0:
            c_ids_tobe_removed = []
            for c_id in c_id_less_cutoff_loss_ids:
                if clients_info[str(c_id)]["one_epoch_train_time"] != -1:
                    c_ids_tobe_removed.append(c_id)
            
            for c_id in c_ids_tobe_removed:
                c_id_less_cutoff_loss_ids.remove(c_id)

            epsilon_selected_clients_ids = []
            epsilon_selected_clients_len = (num_clients - int(num_clients * (1-self.epsilon)))

            if len(c_id_less_cutoff_loss_ids) < epsilon_selected_clients_len:
                additional_c_id_less_cutoff_loss_ids = np.random.choice(c_ids_tobe_removed, min(len(c_ids_tobe_removed), int(epsilon_selected_clients_len - len(c_id_less_cutoff_loss_ids))), replace=False)
                c_id_less_cutoff_loss_ids = [*c_id_less_cutoff_loss_ids, *additional_c_id_less_cutoff_loss_ids]
            
            unselected_devices_round_latencies = []
            for c_id in c_id_less_cutoff_loss_ids:
                if fb_inference_pipelining:
                    client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * num_epochs
                else:
                    client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) + np.mean(c_id_to_client_object[c_id].inference_times_per_sample) * (c_id_to_client_object[c_id].num_train_samples) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * num_epochs
                    
                unselected_devices_round_latencies.append((c_id, client_complete_time))
            
            unselected_devices_round_latencies = sorted(unselected_devices_round_latencies, key = lambda x: x[1])
            epsilon_selected_clients_ids = [elem[0] for elem in unselected_devices_round_latencies[:min(epsilon_selected_clients_len, len(unselected_devices_round_latencies))]]

            if len(epsilon_selected_clients_ids) < epsilon_selected_clients_len:
                selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, min(int(num_clients - len(epsilon_selected_clients_ids)), len(c_id_over_cutoff_loss_ids)), replace=False, p=c_id_over_cutoff_loss_probs)

            
            logger.debug("UTILITY SELECTED CLIENTS: " + str(len(selected_clients_ids)))
            logger.debug(str(selected_clients_ids))
            logger.debug("EPSILON SELECTED CLIENTS: " + str(len(epsilon_selected_clients_ids)))
            logger.debug(str(epsilon_selected_clients_ids))
            logger.debug(str(sorted_c_id_and_overthreshold_loss))
            selected_clients_ids = [*selected_clients_ids, *epsilon_selected_clients_ids]

            if len(selected_clients_ids) < num_clients:
                additional_clients_ids = np.random.choice(c_ids_tobe_removed, min(int(num_clients - len(selected_clients_ids)), len(c_ids_tobe_removed)), replace=False)
                selected_clients_ids = [*selected_clients_ids, *additional_clients_ids]
        else:
            logger.debug("UTILITY SELECTED CLIENTS: " + str(len(selected_clients_ids)))
            logger.debug(str(selected_clients_ids))
            logger.debug(str(sorted_c_id_and_overthreshold_loss))
        
        #Update client_last_selected_round of each selected clients
        for c_id in selected_clients_ids:
            clients_info[str(c.id)]["last_selected_round"] = current_round

        #Retrieve clients from ids, and add statistical utility for oort pacer
        selected_clients = []
        oort_pacer_utility_sum = 0
        for c_id in selected_clients_ids:
            if self.oort_pacer:
                oort_pacer_utility_sum += clients_info[str(c_id)]["utility"]
            for p_c in possible_clients:
                if c_id == str(p_c.id):
                    selected_clients.append(p_c)
        
        # Save the exploited utility of this round for further calculation of oort_pacer
        if self.oort_pacer:
            self.round_exploited_utility.append(oort_pacer_utility_sum)
        
        # Change the epsilon value as in Oort paper
        self.update_epsilon()
        
        if self.oort_blacklist:
            for c in selected_clients:
                clients_info[str(c.id)]["selected_count"] += 1

        return selected_clients, clients_info
    
    # Oort randomly selects a batch per epoch for training at this round
    # Oort selects same samples multiple times if the client has less data than batch_size * num_epoch
    def select_batch_samples(self, batch_size, train_data, num_epochs, model):
        num_data = batch_size
        if len(train_data["x"]) >= num_data * num_epochs:
            xss, yss = zip(*random.sample(list(zip(train_data["x"], train_data["y"])), num_data*num_epochs))
        elif len(train_data["x"]) >= num_data:
            xss = []
            yss = []
            nb = len(train_data["x"]) // num_data
            sampled_batch_cnt = 0
            while sampled_batch_cnt != num_epochs:
                this_iteration_sample_batch_cnt = min(num_epochs - sampled_batch_cnt, nb)
                xsss, ysss = zip(*random.sample(list(zip(train_data["x"], train_data["y"])), num_data*this_iteration_sample_batch_cnt))
                xss += xsss
                yss += ysss
                sampled_batch_cnt += this_iteration_sample_batch_cnt
        else:
            xss = []
            yss = []
            for epoch_idx in range(num_epochs):
                xsss, ysss = zip(*random.sample(list(zip(train_data["x"], train_data["y"])), len(train_data["x"])))
                xss += xsss
                yss += ysss
        xs = xss[:min(batch_size, len(train_data["x"]))]
        ys = yss[:min(batch_size, len(train_data["x"]))]
        oort_whole_data = {'x': xs, 'y': ys}
        oort_whole_data ={"x": self.preprocess_data_x(oort_whole_data["x"]),
                            "y": self.preprocess_data_y(oort_whole_data["y"])}
        oort_whole_data_loss_list = model.test(oort_whole_data)['loss_list']
        sorted_loss = list(oort_whole_data_loss_list)
        data_idx = list(range(len(ys)))

        xss = self.preprocess_data_x(xss)
        yss = self.preprocess_data_y(yss)

        return oort_whole_data, xss, yss, num_data, data_idx, sorted_loss
    
    def preprocess_data_x(self,data):
        return torch.tensor(data,requires_grad=True)
    def preprocess_data_y(self,data):
        data_y=[]
        for i in data:
            data_float=float(i)
            data_y.append(data_float)
        return torch.tensor(data_y,requires_grad=True)