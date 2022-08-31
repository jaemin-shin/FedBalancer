import numpy as np
import timeout_decorator
import traceback
from utils.logger import Logger
from utils.torch_utils import norm_grad
from collections import defaultdict, OrderedDict
import json
import random
import math
import sys
import copy

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

L = Logger()
logger = L.get_logger()

class Server:
    
    def __init__(self, model, clients=[], cfg=None, deadline=0):
        self._cur_time = 0      # simulation time
        self.cfg = cfg
        self.model = model
        self.selected_clients = []
        self.all_clients = clients
        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []
        self.clients_info = defaultdict(dict)
        self.structure_updater = None

        self.loss_threshold_percentage = 0.0
        self.loss_threshold = 0
        self.upper_loss_threshold = 0 
        self.deadline = deadline
        self.deadline_percentage = 1.0
        self.current_round = 0

        self.guard_time = 0

        # This is a deadline only used in client selection algorithm of Oort (Fan Lai et al., OSDI'21)
        self.oort_non_pacer_deadline = deadline

        self.client_download_time = {}
        self.client_upload_time = {}

        self.client_one_epoch_train_time_dict = {}

        if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
            # Oort client selection variables
            self.client_overthreshold_loss_count = {}
            self.client_overthreshold_loss_sum = {}
            self.client_last_selected_round = {}
            self.client_last_selected_round_duration = {}
            self.client_utility = {}
            self.epsilon = 0.9 # time-based exploration factor introduced in Oort, which decreases by a factor 0.98 after each round when it is larger than 0.2
        
        self.system_utility_penalty_alpha = 0.5
        self.pacer_window = 20
        self.round_exploited_utility = []
        
        if self.cfg.oort_pacer:
            self.system_utility_penalty_alpha = 2
        
        if self.cfg.oort_blacklist:
            self.client_selected_count = {}
        
        if self.cfg.fedbalancer or self.cfg.oortbalancer:
            # FEDBALANCER parameters

            # self.current_round_losses = []
            self.if_any_client_sent_response_for_current_round = False
            self.current_round_loss_min = []
            self.current_round_loss_max = []

            self.past_deadlines = []
            self.deadline_just_updated = False

            self.prev_train_losses = []

        for c in self.all_clients:
            self.clients_info[str(c.id)]["acc"] = 0.0
            self.clients_info[str(c.id)]["device"] = c.device.device_model
            self.client_download_time[str(c.id)] = []
            self.client_upload_time[str(c.id)] = []
            self.clients_info[str(c.id)]["sample_num"] = len(c.train_data['y'])
            if self.cfg.fedbalancer or self.cfg.oortbalancer:
                self.client_one_epoch_train_time_dict[str(c.id)] = -1
            if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                self.client_one_epoch_train_time_dict[str(c.id)] = -1
                self.client_overthreshold_loss_count[str(c.id)] = -1
                self.client_overthreshold_loss_sum[str(c.id)] = -1
                self.client_utility[str(c.id)] = 0.0
                self.client_last_selected_round[str(c.id)] = -1
            if self.cfg.oort_pacer:
                self.client_last_selected_round_duration[str(c.id)] = -1
            if self.cfg.oort_blacklist:
                self.client_selected_count[str(c.id)] = 0

    def select_clients(self, my_round, possible_clients, num_clients=20, batch_size=10):
        """Selects num_clients clients from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        
        If Oort, OortBalancer, or FedBalancer, we select clients based on Oort paper (Fan Lai et al., OSDI 2021)
        and Section 3.2.3., "Client Selection with Sample Selection" in our paper (Jaemin Shin et al., MobiSys 2022)

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        if num_clients < self.cfg.min_selected:
            logger.info('insufficient clients: need {} while get {} online'.format(self.cfg.min_selected, num_clients))
            return False
        np.random.seed(my_round)

        # if oort_blacklist is on, add clients that are selected for more rounds than a threshold 
        # to the blacklist_clients list
        if self.cfg.oort_blacklist:
            blacklist_clients = []
            for c in self.all_clients:
                if self.client_selected_count[str(c.id)] > self.cfg.oort_blacklist_rounds:
                    blacklist_clients.append(str(c.id))

        if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
            # Sort client's recorded loss that is over current self.loss_threshold
            possible_clients_ids = []
            if self.cfg.behav_hete:
                for p_c_ in possible_clients:
                    possible_clients_ids.append(str(p_c_.id))

            c_id_and_overthreshold_loss = []
            overthreshold_loss_list = []
            c_id_to_client_object = {}
            
            # Calculate the utility of each clients, based on Oort paper (Fan Lai et al., OSDI 2021)
            # If FedBalancer, only the sample loss above loss_threshold is calculated for utility
            # Otherwise, all samples are calculated (this is possible because loss_threshold = 0)
            for c in self.all_clients:
                if self.cfg.oort_blacklist:
                    if str(c.id) in blacklist_clients:
                        continue
                summ = 0
                
                overthreshold_loss_count = self.client_overthreshold_loss_count[str(c.id)]
                summ = self.client_overthreshold_loss_sum[str(c.id)]

                if overthreshold_loss_count == -1 or overthreshold_loss_count == 0:
                    summ = 0
                else:
                    if self.cfg.oort or self.cfg.oortbalancer:
                        summ = math.sqrt(summ / batch_size) * batch_size
                    elif self.cfg.fb_client_selection:
                        summ = math.sqrt(summ / overthreshold_loss_count) * overthreshold_loss_count
                
                c_id_and_overthreshold_loss.append((str(c.id), summ))
                overthreshold_loss_list.append(summ)
                self.client_utility[str(c.id)] = summ
                c_id_to_client_object[str(c.id)] = c
                
            # Calculate the clip utility of the utility(loss sum) distribution, by 95% value
            overthreshold_loss_list.sort()
            clip_value = overthreshold_loss_list[min(int(len(overthreshold_loss_list)*0.95), len(overthreshold_loss_list)-1)]
            
            # Add incentive term for clients that have been overlooked for a long time
            for tmp_idx in range(len(c_id_and_overthreshold_loss)):
                c_id = c_id_and_overthreshold_loss[tmp_idx][0]
                summ = min(c_id_and_overthreshold_loss[tmp_idx][1], clip_value)
                if self.client_last_selected_round[c_id] != -1:
                    summ += math.sqrt(0.1*math.log(self.current_round + 1)/(self.client_last_selected_round[c_id]+1)) #To avoid zero division, we regard the training round starts from 1 only at this equation
                if self.cfg.oort_pacer:
                    if self.client_last_selected_round_duration[c_id] != -1 and self.client_last_selected_round_duration[c_id] > self.deadline:
                        summ *= math.pow(self.deadline / self.client_last_selected_round_duration[c_id], self.system_utility_penalty_alpha)
                else:
                    if self.cfg.fb_inference_pipelining:
                        client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//self.cfg.batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * self.cfg.num_epochs
                    else:
                        client_complete_time = (c_id_to_client_object[c_id].device.get_expected_download_time()) + (c_id_to_client_object[c_id].device.get_expected_upload_time()) + np.mean(c_id_to_client_object[c_id].inference_times_per_sample) * (c_id_to_client_object[c_id].num_train_samples) +  ((np.mean(c_id_to_client_object[c_id].trained_num_of_samples[-5:])-1)//self.cfg.batch_size + 1) * np.mean(c_id_to_client_object[c_id].per_batch_train_times) * self.cfg.num_epochs
                        
                    if client_complete_time > self.oort_non_pacer_deadline:
                        summ *= math.pow(self.oort_non_pacer_deadline / client_complete_time, self.system_utility_penalty_alpha)
                c_id_and_overthreshold_loss[tmp_idx] = (c_id, summ)

            sorted_c_id_and_overthreshold_loss = sorted(c_id_and_overthreshold_loss, key=lambda tup: tup[1], reverse=True)
            
            # Sample clients from 1-epsilon, prioritizing statistical utility
            cutoff_loss = 0.95*(sorted_c_id_and_overthreshold_loss[int(num_clients*(1-self.epsilon))][1])

            # Perform random client selection if cutoff_loss == 0
            if cutoff_loss == 0:
                self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
                for c in self.selected_clients:
                    self.client_last_selected_round[str(c.id)] = self.current_round
                return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]
            
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

            intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids = []
            intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs = []
            possible_clients_not_in_c_id_over_cutoff_loss_ids = []
            
            # Behav_hete needs more debugging.. this will be patched soon. However, behav_hete was not related to evaluation in our FedBalancer paper!
            if self.cfg.behav_hete:
                for c_id_idx in range(len(c_id_over_cutoff_loss_ids)):
                    if c_id_over_cutoff_loss_ids[c_id_idx] in possible_clients_ids:
                        intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids.append(c_id_over_cutoff_loss_ids[c_id_idx])
                        intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs.append(c_id_over_cutoff_loss_probs[c_id_idx])
                
                i_probs_sum = 0
                for i_prob in intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs:
                    i_probs_sum += i_prob
                
                for i_prob_idx in range(len(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs)):
                    intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs[i_prob_idx] /= i_probs_sum
                
                for p_c_id_ in possible_clients_ids:
                    if p_c_id_ not in intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids:
                        possible_clients_not_in_c_id_over_cutoff_loss_ids.append(p_c_id_)

                if len(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids) > int(num_clients*(1-self.epsilon)):
                    selected_clients_ids = np.random.choice(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids, int(num_clients*(1-self.epsilon)), replace=False, p=intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs)    
                else:
                    selected_clients_ids = intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids
            else:
                selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, int(num_clients*(1-self.epsilon)), replace=False, p=c_id_over_cutoff_loss_probs)
            
            
            # Sample clients from epsilon, which have less utility than the cutoff_loss
            # Prioritize clients which have faster speed;
            if self.epsilon > 0.0:
                c_ids_tobe_removed = []
                for c_id in c_id_less_cutoff_loss_ids:
                    if self.client_one_epoch_train_time_dict[str(c_id)] != -1:
                        c_ids_tobe_removed.append(c_id)
                
                for c_id in c_ids_tobe_removed:
                    c_id_less_cutoff_loss_ids.remove(c_id)

                epsilon_selected_clients_ids = []
                epsilon_selected_clients_len = (num_clients - int(num_clients * (1-self.epsilon)))

                if len(c_id_less_cutoff_loss_ids) < epsilon_selected_clients_len:
                    additional_c_id_less_cutoff_loss_ids = np.random.choice(c_ids_tobe_removed, min(len(c_ids_tobe_removed), int(epsilon_selected_clients_len - len(c_id_less_cutoff_loss_ids))), replace=False)
                    c_id_less_cutoff_loss_ids = [*c_id_less_cutoff_loss_ids, *additional_c_id_less_cutoff_loss_ids]

                devices_lists_from_fast_to_slow = ['Google Pixel 4', 'Xiaomi Redmi Note 7 Pro', 'Google Nexus S']
                device_idx = 0

                while len(epsilon_selected_clients_ids) < epsilon_selected_clients_len and device_idx < 3:
                    tmp_epsilon_selected_clients_ids = []
                    for c_id in c_id_less_cutoff_loss_ids:
                        if (self.clients_info[str(c_id)]["device"] == devices_lists_from_fast_to_slow[device_idx]):
                            if self.cfg.behav_hete:
                                if c_id in possible_clients_ids:
                                    tmp_epsilon_selected_clients_ids.append(c_id)
                            else:
                                tmp_epsilon_selected_clients_ids.append(c_id)
                    if (len(epsilon_selected_clients_ids) + len(tmp_epsilon_selected_clients_ids)) > epsilon_selected_clients_len:
                        curr_device_epsilon_selected_clients_ids = np.random.choice(tmp_epsilon_selected_clients_ids, epsilon_selected_clients_len - len(epsilon_selected_clients_ids), replace=False)
                    else:
                        curr_device_epsilon_selected_clients_ids = tmp_epsilon_selected_clients_ids
                    epsilon_selected_clients_ids = [*epsilon_selected_clients_ids, *curr_device_epsilon_selected_clients_ids]
                    device_idx += 1
                
                if len(epsilon_selected_clients_ids) < epsilon_selected_clients_len:
                    if self.cfg.behav_hete:
                        if min(int(num_clients - len(epsilon_selected_clients_ids)), len(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids)) > 0:
                            selected_clients_ids = np.random.choice(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids, min(int(num_clients - len(epsilon_selected_clients_ids)), len(intersection_btw_possible_clients_and_c_id_over_cutoff_loss_ids)), replace=False, p=intersection_btw_possible_clients_and_c_id_over_cutoff_loss_probs)
                        if len(selected_clients_ids) + len(epsilon_selected_clients_ids) < num_clients:
                            remaining_sampled_clients_ids = np.random.choice(possible_clients_not_in_c_id_over_cutoff_loss_ids, num_clients - (len(selected_clients_ids) + len(epsilon_selected_clients_ids)), replace=False)
                            selected_clients_ids = [*selected_clients_ids, *remaining_sampled_clients_ids]
                    else:
                        selected_clients_ids = np.random.choice(c_id_over_cutoff_loss_ids, min(int(num_clients - len(epsilon_selected_clients_ids)), len(c_id_over_cutoff_loss_ids)), replace=False, p=c_id_over_cutoff_loss_probs)
                selected_clients_ids = [*selected_clients_ids, *epsilon_selected_clients_ids]

            #Update client_last_selected_round of each selected clients
            for c_id in selected_clients_ids:
                self.client_last_selected_round[c_id] = self.current_round

            #Retrieve clients from ids, and add statistical utility for oort pacer
            selected_clients = []
            oort_pacer_utility_sum = 0
            for c_id in selected_clients_ids:
                if self.cfg.oort_pacer:
                    if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                        oort_pacer_utility_sum += self.client_utility[str(c_id)]
                for p_c in possible_clients:
                    if c_id == str(p_c.id):
                        selected_clients.append(p_c)
            
            # Save the exploited utility of this round for further calculation of oort_pacer
            if self.cfg.oort_pacer:
                self.round_exploited_utility.append(oort_pacer_utility_sum)
            
            # Change the epsilon value as in Oort paper
            self.selected_clients = selected_clients
            if self.epsilon > 0.2:
                logger.info('epsilon {}'.format(self.epsilon))
                self.epsilon = self.epsilon * 0.98

        # Randomly sample clients if Oort, Oortbalancer, or FedBalancer is not being used
        else:
            self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        
        if self.cfg.oort_blacklist:
            for c in self.selected_clients:
                self.client_selected_count[str(c.id)] += 1

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]
    
    def filter_if_more_than_ratio_is_explored(self, client_one_epoch_train_time_dict, ratio):
        client_time_values = list(client_one_epoch_train_time_dict.values())
        client_num = len(client_time_values)
        return client_time_values.count(-1) < client_num * (1 - ratio)
    
    def filter_if_more_than_number_is_explored(self, client_one_epoch_train_time_dict, number):
        client_time_values = list(client_one_epoch_train_time_dict.values())
        client_num = len(client_time_values)
        return (client_num - client_time_values.count(-1)) > number
    
    def findPeakDDLE(self, selected_clients, num_epochs):
        t_max = sys.maxsize
        total_user_count = len(selected_clients)
        
        complete_user_counts_per_time = []
        max_complete_user_counts_per_time = -1
        max_complete_user_counts_per_time_idx = -1

        client_complete_time = {}

        for c in selected_clients:
            if self.cfg.fb_inference_pipelining:
                client_complete_time[str(c.id)] = (c.device.get_expected_download_time()) + (c.device.get_expected_upload_time()) + ((np.mean(c.trained_num_of_samples[-5:])-1)//self.cfg.batch_size + 1) * np.mean(c.per_batch_train_times) * num_epochs + self.guard_time
            else:
                client_complete_time[str(c.id)] = (c.device.get_expected_download_time()) + (c.device.get_expected_upload_time()) + np.mean(c.inference_times_per_sample) * (c.num_train_samples) +  ((np.mean(c.trained_num_of_samples[-5:])-1)//self.cfg.batch_size + 1) * np.mean(c.per_batch_train_times) * num_epochs + self.guard_time
        
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

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0,
                   'acc': {},
                   'loss': {},
                   'ori_d_t': 0,
                   'ori_t_t': 0,
                   'ori_u_t': 0,
                   'act_d_t': 0,
                   'act_t_t': 0,
                   'act_u_t': 0} for c in clients}

        simulate_time = 0
        accs = []
        losses = []

        self.updates = []
        self.gradiants = []
        self.deltas = []
        self.hs = []

        # Fedbalancer variables
        sorted_loss_sum = 0
        num_of_samples = 0
        max_loss = 0
        min_loss = sys.maxsize

        # Fedbalancer loss threshold and deadline update based on the loss threshold percentage (ratio) and the deadline percentage (ratio)
        if (self.cfg.fedbalancer or self.cfg.oortbalancer) and self.if_any_client_sent_response_for_current_round:
            if self.loss_threshold == 0 and self.loss_threshold_percentage == 0:
                self.loss_threshold = 0
                logger.info('loss_threshold {}'.format(self.loss_threshold))
            else:
                loss_low = np.min(self.current_round_loss_min)
                loss_high = np.mean(self.current_round_loss_max)
                self.loss_threshold = loss_low + (loss_high - loss_low) * self.loss_threshold_percentage
                            
                logger.info('loss_low {}, loss_high {}, loss_threshold {}'.format(loss_low, loss_high, self.loss_threshold))

            # Deadline is only updated when the round completion time of more than half clients (or more than 200 clients) are explored
            if self.cfg.fb_simple_control_ddl and not self.deadline_just_updated and (self.filter_if_more_than_ratio_is_explored(self.client_one_epoch_train_time_dict, 0.5) or self.filter_if_more_than_number_is_explored(self.client_one_epoch_train_time_dict, 200)):
                deadline_low = self.findPeakDDLE(self.selected_clients, 1)
                deadline_high = self.findPeakDDLE(self.selected_clients, self.cfg.num_epochs)
                self.deadline = deadline_low + (deadline_high - deadline_low) * self.deadline_percentage

                logger.info('deadline_low {}, deadline_high {}, deadline {}'.format(deadline_low, deadline_high, self.deadline))

            if self.deadline_just_updated:
                self.past_deadlines.append(self.deadline)
                self.deadline_just_updated = False
        
        # These are the two cases which the round does not end with deadline; it ends when predefined number of clients succeed in a round
        # Thus, these two cases are handled separately from other cases
        if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
            client_tmp_info = {}
            client_tmp_info = {
                c.id: {
                    'simulate_time_c': 0,
                    'num_samples': 0,
                    'update': 0,
                    'acc': 0,
                    'loss': 0,
                    'update_size': 0,
                    'seed': 0,
                    'sorted_loss': 0,
                    'download_time': 0,
                    'upload_time': 0,
                    'train_time': 0,
                    'inference_time': 0,
                    'completed_epochs': 0,
                    'c_model_size': 0,
                    'c_before_comp_upload_time': 0,
                    'c_ori_download_time': 0,
                    'c_ori_train_time': 0,
                    'c_ori_upload_time': 0,
                    'c_act_download_time': 0,
                    'c_act_train_time': 0,
                    'c_act_upload_time': 0,
                    'client_simulate_time': 0} for c in clients}
            client_simulate_times = []

        if self.cfg.fedbalancer or self.cfg.oortbalancer:
            logger.info('this round deadline {}, loss_threshold {}'.format(self.deadline, self.loss_threshold))
            logger.info('this round deadline percentage {}, loss_threshold percentage {}'.format(self.deadline_percentage, self.loss_threshold_percentage))
        else:
            logger.info('this round deadline {}'.format(self.deadline))
        
        curr_round_exploited_utility = 0.0

        for c in clients:
            c.model = copy.deepcopy(self.model)
        
        for c in clients:
            try:
                if self.cfg.fedbalancer or self.cfg.oortbalancer:
                    c.set_deadline(self.deadline)
                    c.set_loss_threshold(self.loss_threshold, self.upper_loss_threshold)
                else:
                    c.set_deadline(self.deadline)

                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.get_cur_time()
                
                # train on the client
                simulate_time_c, num_samples, update, acc, loss, update_size, sorted_loss, download_time, upload_time, train_time, inference_time, completed_epochs = c.train(start_t, num_epochs, batch_size, minibatch)

                # These are the two cases which the round does not end with deadline; it ends when predefined number of clients succeed in a round
                # Thus, these two cases are handled separately from other cases
                if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
                    client_tmp_info[c.id]['simulate_time_c'] = simulate_time_c
                    client_tmp_info[c.id]['num_samples'] = num_samples
                    client_tmp_info[c.id]['update'] = update
                    client_tmp_info[c.id]['acc'] = acc
                    client_tmp_info[c.id]['loss'] = loss
                    client_tmp_info[c.id]['update_size'] = update_size
                    client_tmp_info[c.id]['sorted_loss'] = sorted_loss
                    client_tmp_info[c.id]['download_time'] = download_time
                    client_tmp_info[c.id]['upload_time'] = upload_time
                    client_tmp_info[c.id]['train_time'] = train_time
                    client_tmp_info[c.id]['inference_time'] = inference_time
                    client_tmp_info[c.id]['completed_epochs'] = completed_epochs
                    client_tmp_info[c.id]['c_model_size'] = c.model.size
                    client_tmp_info[c.id]['c_before_comp_upload_time'] = c.before_comp_upload_time
                    client_tmp_info[c.id]['c_ori_download_time'] = c.ori_download_time
                    client_tmp_info[c.id]['c_ori_inference_time'] = c.ori_inference_time
                    client_tmp_info[c.id]['c_ori_train_time'] = c.ori_train_time
                    client_tmp_info[c.id]['c_ori_upload_time'] = c.ori_upload_time
                    client_tmp_info[c.id]['c_act_download_time'] = c.act_download_time
                    client_tmp_info[c.id]['c_act_inference_time'] = c.act_inference_time
                    client_tmp_info[c.id]['c_act_train_time'] = c.act_train_time
                    client_tmp_info[c.id]['c_act_upload_time'] = c.act_upload_time
                    client_tmp_info[c.id]['client_simulate_time'] = max(simulate_time_c, download_time + upload_time + train_time + inference_time)
                    client_simulate_times.append((c.id, max(simulate_time_c, download_time + upload_time + train_time + inference_time)))
                else:
                    self.client_download_time[str(c.id)].append(download_time)
                    self.client_upload_time[str(c.id)].append(upload_time)

                    self.client_one_epoch_train_time_dict[str(c.id)] = train_time/completed_epochs

                    # In case of using Oort-based client selection, when we do not use pacer, the round ends with the deadline.
                    # We calculate the curr_round_exploited_utility from the successful clients at the round.
                    if (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) and not self.cfg.oort_pacer:
                        curr_round_exploited_utility += self.client_utility[str(c.id)]
                    
                    # If everyone is at least once selected for a round, we set epsilon as zero, which is the exploration-exploitation parameter of Oort
                    if (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) and self.epsilon != 0:
                        is_everyone_explored = True
                        for value in self.client_one_epoch_train_time_dict.values():
                            if value == -1:
                                is_everyone_explored = False
                                break
                        if is_everyone_explored:
                            self.epsilon = 0.0

                    # calculate succeeded client's round duration for oort pacer
                    if self.cfg.oort_pacer and (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) :
                        self.client_last_selected_round_duration[str(c.id)] = download_time + train_time + upload_time + inference_time
                    
                    # in FedBalancer algorithm, this is done in client-side.
                    # doing this here makes no difference in performance and privacy (because we do not use sample-level information other than calculating the overthreshold sum and count)
                    # but we will patch this soon to move this to client
                    if len(sorted_loss) > 0:
                        sorted_loss_sum += sum(sorted_loss)
                        num_of_samples += len(sorted_loss)
                        if sorted_loss[0] < min_loss:
                            min_loss = sorted_loss[0]
                        if sorted_loss[-1] > max_loss:
                            max_loss = sorted_loss[-1]
                        if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                            summ = 0
                            overthreshold_loss_count = 0
                            for loss_idx in range(len(sorted_loss)):
                                if sorted_loss[len(sorted_loss)-1-loss_idx] > self.loss_threshold:
                                    summ += sorted_loss[len(sorted_loss)-1-loss_idx]*sorted_loss[len(sorted_loss)-1-loss_idx]
                                    overthreshold_loss_count += 1
                            self.client_overthreshold_loss_sum[str(c.id)] = summ
                            self.client_overthreshold_loss_count[str(c.id)] = overthreshold_loss_count
                        if self.cfg.fedbalancer or self.cfg.oortbalancer:
                            self.if_any_client_sent_response_for_current_round = True
                            noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                            noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                            self.current_round_loss_min.append(np.min(sorted_loss)+noise1)
                            self.current_round_loss_max.append(np.percentile(sorted_loss, 80)+noise2)

                    logger.debug('client {} simulate_time: {}'.format(c.id, simulate_time_c))
                    logger.debug('client {} num_samples: {}'.format(c.id, num_samples))
                    logger.debug('client {} acc: {}, loss: {}'.format(c.id, acc, loss))
                    accs.append(acc)
                    losses.append(loss)
                    
                    simulate_time = min(self.deadline, max(simulate_time, simulate_time_c))

                    sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                    sys_metrics[c.id][BYTES_WRITTEN_KEY] += update_size
                    sys_metrics[c.id]['acc'] = acc
                    sys_metrics[c.id]['loss'] = loss
                    # uploading 
                    self.updates.append((c.id, num_samples, update))

                    logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                logger.debug('client {} failed: {}'.format(c.id, e))
                
                sys_metrics[c.id]['acc'] = -1
                sys_metrics[c.id]['loss'] = -1
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.update_size
                
                if self.cfg.fedbalancer or self.cfg.oortbalancer:
                    simulate_time = self.deadline
                    # in FedBalancer algorithm, this is done in client-side.
                    # doing this here makes no difference in performance and privacy (because we do not use sample-level information other than calculating the overthreshold sum and count)
                    # but we will patch this soon to move this to client
                    if len(c.sorted_loss) != 0:
                        # self.client_loss_count[str(c.id)] = len(c.sorted_loss)
                        summ = 0
                        overthreshold_loss_count = 0

                        for loss_idx in range(len(c.sorted_loss)):
                            if c.sorted_loss[len(c.sorted_loss)-1-loss_idx] > self.loss_threshold:
                                summ += c.sorted_loss[len(c.sorted_loss)-1-loss_idx]*c.sorted_loss[len(c.sorted_loss)-1-loss_idx]
                                overthreshold_loss_count += 1
                        if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                            self.client_overthreshold_loss_sum[str(c.id)] = summ
                            self.client_overthreshold_loss_count[str(c.id)] = overthreshold_loss_count
                        
                        noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        self.current_round_loss_min.append(np.min(c.sorted_loss)+noise1)
                        self.current_round_loss_max.append(np.percentile(c.sorted_loss, 80)+noise2)
                else:
                    simulate_time = self.deadline
                
                if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
                    assert(False)
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                traceback.print_exc()
            finally:
                if self.cfg.compress_algo:
                    sys_metrics[c.id]['before_cprs_u_t'] = round(c.before_comp_upload_time, 3)
                sys_metrics[c.id]['ori_d_t'] = round(c.ori_download_time, 3)
                sys_metrics[c.id]['ori_i_t'] = round(c.ori_inference_time, 3)
                sys_metrics[c.id]['ori_t_t'] = round(c.ori_train_time, 3)
                sys_metrics[c.id]['ori_u_t'] = round(c.ori_upload_time, 3)
                
                sys_metrics[c.id]['act_d_t'] = round(c.act_download_time, 3)
                sys_metrics[c.id]['act_i_t'] = round(c.act_inference_time, 3)
                sys_metrics[c.id]['act_t_t'] = round(c.act_train_time, 3)
                sys_metrics[c.id]['act_u_t'] = round(c.act_upload_time, 3)

        # These are the two cases which the round does not end with deadline; it ends when predefined number of clients succeed in a round
        # Thus, these two cases are handled separately from other cases
        # We sort the client round completion time, and aggregate the result of faster clients within the predefined number of succeeded clients in a round
        if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
            client_simulate_times = sorted(client_simulate_times, key=lambda tup: tup[1])
            for_loop_until = 0
            if self.cfg.oort_pacer:
                for_loop_until = min(self.cfg.clients_per_round, len(client_simulate_times))
            elif self.cfg.ddl_baseline_smartpc:
                for_loop_until = int(min(self.cfg.clients_per_round, len(client_simulate_times)) * self.cfg.ddl_baseline_smartpc_percentage)
            for c_idx in range(for_loop_until):
                c_id = client_simulate_times[c_idx][0]
                self.client_download_time[str(c_id)].append(client_tmp_info[c_id]['download_time'])
                self.client_upload_time[str(c_id)].append(client_tmp_info[c_id]['upload_time'])

                self.client_one_epoch_train_time_dict[str(c_id)] = client_tmp_info[c_id]['train_time']/client_tmp_info[c_id]['completed_epochs']
                if (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) and self.epsilon != 0:
                    is_everyone_explored = True
                    for value in self.client_one_epoch_train_time_dict.values():
                        if value == -1:
                            is_everyone_explored = False
                            break
                    if is_everyone_explored:
                        self.epsilon = 0.0

                # calculate succeeded client's round duration for oort pacer
                if self.cfg.oort_pacer and (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer):
                    self.client_last_selected_round_duration[str(c.id)] = download_time + train_time + upload_time + inference_time
                
                # in FedBalancer algorithm, this is done in client-side.
                # doing this here makes no difference in performance and privacy (because we do not use sample-level information other than calculating the overthreshold sum and count)
                # but we will patch this soon to move this to client

                if len(client_tmp_info[c_id]['sorted_loss']) > 0:
                    sorted_loss_sum += sum(client_tmp_info[c_id]['sorted_loss'])
                    num_of_samples += len(client_tmp_info[c_id]['sorted_loss'])
                    if client_tmp_info[c_id]['sorted_loss'][0] < min_loss:
                        min_loss = client_tmp_info[c_id]['sorted_loss'][0]
                    if client_tmp_info[c_id]['sorted_loss'][-1] > max_loss:
                        max_loss = client_tmp_info[c_id]['sorted_loss'][-1]
                    if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                        #self.client_loss[c_id] = client_tmp_info[c_id]['sorted_loss']
                        # self.client_loss_count[c_id] = len(client_tmp_info[c_id]['sorted_loss'])
                        summ = 0
                        overthreshold_loss_count = 0
                        for loss_idx in range(len(client_tmp_info[c_id]['sorted_loss'])):
                            if client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx] > self.loss_threshold:
                                summ += client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx]*client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx]
                                overthreshold_loss_count += 1
                        self.client_overthreshold_loss_sum[c_id] = summ
                        self.client_overthreshold_loss_count[c_id] = overthreshold_loss_count

                    if self.cfg.fedbalancer or self.cfg.oortbalancer:
                        # self.current_round_losses = self.current_round_losses + client_tmp_info[c_id]['sorted_loss']
                        self.if_any_client_sent_response_for_current_round = True
                        noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        self.current_round_loss_min.append(np.min(client_tmp_info[c_id]['sorted_loss'])+noise1)
                        self.current_round_loss_max.append(np.percentile(client_tmp_info[c_id]['sorted_loss'], 80)+noise2)

                logger.debug('client {} simulate_time: {}'.format(c_id, client_tmp_info[c_id]['simulate_time_c']))
                logger.debug('client {} num_samples: {}'.format(c_id, client_tmp_info[c_id]['num_samples']))
                logger.debug('client {} acc: {}, loss: {}'.format(c_id, client_tmp_info[c_id]['acc'], client_tmp_info[c_id]['loss']))
                accs.append(client_tmp_info[c_id]['acc'])
                losses.append(client_tmp_info[c_id]['loss'])

                if self.cfg.fedbalancer or self.cfg.oortbalancer:
                    simulate_time = min(self.deadline, max(simulate_time, client_tmp_info[c_id]['simulate_time_c']))
                elif self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
                    simulate_time = max(simulate_time, client_tmp_info[c_id]['client_simulate_time'])
                else:
                    simulate_time = min(self.deadline, max(simulate_time, client_tmp_info[c_id]['simulate_time_c']))

                sys_metrics[c_id][BYTES_READ_KEY] += client_tmp_info[c_id]['c_model_size']
                sys_metrics[c_id][BYTES_WRITTEN_KEY] += client_tmp_info[c_id]['update_size']
                sys_metrics[c_id]['acc'] = client_tmp_info[c_id]['acc']
                sys_metrics[c_id]['loss'] = client_tmp_info[c_id]['loss']
                # uploading 
                self.updates.append((c_id, client_tmp_info[c_id]['num_samples'], client_tmp_info[c_id]['update']))

                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c_id,client_tmp_info[c_id]['acc'], client_tmp_info[c_id]['loss']))

                if self.cfg.compress_algo:
                    sys_metrics[c_id]['before_cprs_u_t'] = round(client_tmp_info[c_id]['c_before_comp_upload_time'], 3)
                sys_metrics[c_id]['ori_d_t'] = round(client_tmp_info[c_id]['c_ori_download_time'], 3)
                sys_metrics[c_id]['ori_i_t'] = round(client_tmp_info[c_id]['c_ori_inference_time'], 3)
                sys_metrics[c_id]['ori_t_t'] = round(client_tmp_info[c_id]['c_ori_train_time'], 3)
                sys_metrics[c_id]['ori_u_t'] = round(client_tmp_info[c_id]['c_ori_upload_time'], 3)
                
                sys_metrics[c_id]['act_d_t'] = round(client_tmp_info[c_id]['c_act_download_time'], 3)
                sys_metrics[c_id]['act_i_t'] = round(client_tmp_info[c_id]['c_act_inference_time'], 3)
                sys_metrics[c_id]['act_t_t'] = round(client_tmp_info[c_id]['c_act_train_time'], 3)
                sys_metrics[c_id]['act_u_t'] = round(client_tmp_info[c_id]['c_ori_upload_time'], 3)
        try:
            # logger.info('simulation time: {}'.format(simulate_time))
            sys_metrics['configuration_time'] = simulate_time
            avg_acc = sum(accs)/len(accs)
            avg_loss = sum(losses)/len(losses)
            logger.info('average acc: {}, average loss: {}'.format(avg_acc, avg_loss))
            logger.info('configuration and update stage simulation time: {}'.format(simulate_time))

            # In case of using Oort-based client selection, when we do not use pacer, the round ends with the deadline.
            # We calculate the curr_round_exploited_utility from the successful clients at the round.
            if not self.cfg.oort_pacer:
                if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                    self.round_exploited_utility.append(curr_round_exploited_utility)
            
            # Update the loss threshold percentage (ratio) and the deadline percentage (ratio) 
            # according to the Algorithm 3 in FedBalancer paper
            if self.cfg.fedbalancer or self.cfg.oortbalancer:
                current_round_loss = (sorted_loss_sum / num_of_samples) / (self.deadline)
                self.prev_train_losses.append(current_round_loss)
                logger.info('current_round_loss: {}'.format(current_round_loss))
                if self.current_round % int(self.cfg.fb_w) == int(self.cfg.fb_w) - 1 :
                    if len(self.prev_train_losses) >= 2 * self.cfg.fb_w:
                        if self.cfg.fb_simple_control_lt or self.cfg.fb_simple_control_ddl:
                            nonscaled_reward = (np.mean(self.prev_train_losses[-(self.cfg.fb_w*2):-(self.cfg.fb_w)]) - np.mean(self.prev_train_losses[-(self.cfg.fb_w):]))
                        
                        if self.cfg.fb_simple_control_lt:
                            if nonscaled_reward > 0:
                                if self.loss_threshold_percentage + self.cfg.fb_simple_control_lt_stepsize <= 1:
                                    self.loss_threshold_percentage += self.cfg.fb_simple_control_lt_stepsize
                            else:
                                if self.loss_threshold_percentage - self.cfg.fb_simple_control_lt_stepsize >= 0:
                                    self.loss_threshold_percentage -= self.cfg.fb_simple_control_lt_stepsize
                        if self.cfg.fb_simple_control_ddl:
                            if nonscaled_reward > 0:
                                if self.deadline_percentage - self.cfg.fb_simple_control_ddl_stepsize >= 0:
                                    self.deadline_percentage -= self.cfg.fb_simple_control_ddl_stepsize
                            else:
                                if self.deadline_percentage + self.cfg.fb_simple_control_ddl_stepsize <= 1:
                                    self.deadline_percentage += self.cfg.fb_simple_control_ddl_stepsize
                    # else:
                    #     self.loss_threshold = min_loss
                self.current_round += 1
                logger.info('min sample loss: {}, max sample loss: {}'.format(min_loss, max_loss))
            # logger.info('losses: {}'.format(losses))
        except ZeroDivisionError as e:
            logger.error('training time window is too short to train!')
            # assert False
        except Exception as e:
            logger.error('failed reason: {}'.format(e))
            traceback.print_exc()
            assert False
        return sys_metrics

    def update_model(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
                self.deltas = []
                self.hs = []
                return
            if self.cfg.aggregate_algorithm == 'SucFedAvg':
                # aggregate the successfully uploaded clients
                logger.info('Aggragate with SucFedAvg')
                total_weight = 0.


                total_data_size = sum([client_num_samples for (cid, client_num_samples, client_model_state) in self.updates])
                aggregation_weights = [client_num_samples / total_data_size for (cid, client_num_samples, client_model_state) in self.updates]

                update_state = OrderedDict()
                for k, (cid, client_samples, client_model) in enumerate(self.updates):
                    for key in self.model.net.state_dict().keys():
                        if k == 0:
                            update_state[key] = client_model[key] * aggregation_weights[k]
                        else:
                            update_state[key] += client_model[key] * aggregation_weights[k]
                self.model.net.load_state_dict(update_state)
            else:
                # not supported aggregating algorithm
                logger.error('not supported aggregating algorithm: {}'.format(self.cfg.aggregate_algorithm))
                assert False
            self.guard_time = 0
        else:
            logger.info('round failed, global model maintained.')
            self.guard_time += 10
            
        
        self.updates = []
        self.deltas = []
        self.hs = []
        
    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients
            assert False

        for client in clients_to_test:
            client.model = copy.deepcopy(self.model)
            c_metrics = client.test(set_to_use)
            # logger.info('client {} metrics: {}'.format(client.id, c_metrics))
            metrics[client.id] = c_metrics
            if isinstance(c_metrics['accuracy'], np.ndarray):
                self.clients_info[client.id]['acc'] = c_metrics['accuracy'].tolist()
            else:
                self.clients_info[client.id]['acc'] = c_metrics['accuracy']
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.all_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples
    
    def get_cur_time(self):
        return self._cur_time

    def pass_time(self, sec):
        self._cur_time += sec
    
    def get_time_window(self):
        tw =  np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        while tw < 0:
            tw =  np.random.normal(self.cfg.time_window[0], self.cfg.time_window[1])
        return tw
