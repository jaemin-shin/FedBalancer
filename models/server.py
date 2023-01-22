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

from oort import Oort
from fedbalancer import FedBalancer

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

L = Logger()
logger = L.get_logger()

class Server:
    
    def __init__(self, model, clients=[], cfg=None, deadline=0):
        self._cur_time = 0      # simulation time
        self.model = model
        self.all_clients = clients
        self.cfg = cfg
        self.deadline = deadline

        self.selected_clients = []
        self.updates = []
        self.clients_info = defaultdict(dict)
        self.test_clients_info = defaultdict(dict)

        self.failed_clients = []

        self.current_round = 0

        self.oort = None
        self.fedbalancer = None

        if self.cfg.oort or self.cfg.fb_client_selection or self.cfg.oortbalancer:
            self.oort = Oort(self.all_clients, deadline, oort_pacer = self.cfg.oort_pacer, pacer_delta = self.cfg.oort_pacer_delta, oort_blacklist = self.cfg.oort_blacklist)
        
        if self.cfg.fedbalancer or self.cfg.oortbalancer or self.cfg.ss_baseline:
            self.fedbalancer = FedBalancer(self.cfg.fb_inference_pipelining, self.cfg.fb_p, self.cfg.fb_w, self.cfg.fb_simple_control_lt_stepsize, self.cfg.fb_simple_control_ddl_stepsize)

        for c in self.all_clients:
            self.clients_info[str(c.id)]["acc"] = 0.0
            self.clients_info[str(c.id)]["device"] = c.device.device_model
            self.clients_info[str(c.id)]["sample_num"] = len(c.train_data['y'])

            self.clients_info[str(c.id)]["download_times"] = []
            self.clients_info[str(c.id)]["upload_times"] = []

            self.clients_info[str(c.id)]["one_epoch_train_time"] = -1
            self.clients_info[str(c.id)]["overthreshold_loss_count"] = -1
            self.clients_info[str(c.id)]["overthreshold_loss_sum"] = -1
            self.clients_info[str(c.id)]["utility"] = 0.0
            self.clients_info[str(c.id)]["last_selected_round"] = -1

            if self.cfg.oort_pacer:
                self.clients_info[str(c.id)]["last_selected_round_duration"] = -1
            if self.cfg.oort_blacklist:
                self.clients_info[str(c.id)]["selected_count"] = -1
            
            c.fedbalancer = self.fedbalancer
            c.oort = self.oort

    def select_clients(self, possible_clients, num_clients=20, batch_size=10):
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
        # np.random.seed(self.current_round)

        if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
            if self.cfg.oort_pacer:
                self.deadline = self.oort.pacer_deadline_update(self.deadline, self.current_round)
            self.selected_clients, self.clients_info = self.oort.select_clients(self.all_clients, possible_clients, num_clients, self.clients_info, self.current_round, self.deadline, self.cfg.batch_size, self.cfg.num_epochs,
                                                                            self.cfg.behav_hete, self.cfg.fb_client_selection, self.cfg.fb_inference_pipelining, self.cfg.oortbalancer)

        # Randomly sample clients if Oort, Oortbalancer, or FedBalancer is not being used
        else:
            self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """

        simulate_time = 0
        accs = []
        losses = []

        self.updates = []

        # Fedbalancer variables
        sorted_loss_sum = 0
        num_of_samples = 0
        max_loss = 0
        min_loss = sys.maxsize

        # Fedbalancer loss threshold and deadline update based on the loss threshold percentage (ratio) and the deadline percentage (ratio)
        if (self.cfg.fedbalancer or self.cfg.oortbalancer) and self.fedbalancer.if_any_client_sent_response_for_current_round:
            self.fedbalancer.loss_threshold_selection()

            if self.cfg.fb_simple_control_ddl_stepsize != 0.0:
                self.deadline = self.fedbalancer.deadline_selection(self.selected_clients, self.clients_info, self.cfg.num_epochs, self.deadline, self.cfg.batch_size, self.current_round)
        
            logger.info('this round deadline {}, loss_threshold {}'.format(self.deadline, self.fedbalancer.loss_threshold))
            logger.info('this round deadline ratio {}, loss_threshold ratio {}'.format(self.fedbalancer.deadline_ratio, self.fedbalancer.loss_threshold_ratio))
        else:
            logger.info('this round deadline {}'.format(self.deadline))

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
                    'client_simulate_time': 0} for c in self.selected_clients}
            client_simulate_times = []
        
        if self.oort != None:
            self.oort.curr_round_exploited_utility = 0.0

        server_current_model = copy.deepcopy(self.model)
        # for c in clients:
        #     c.model = copy.deepcopy(self.model)

        round_failed_clients = []
        
        for c in self.selected_clients:
            c.model = None
            c._model = None
            c.model = copy.deepcopy(server_current_model)

            try:
                c.set_deadline(self.deadline)
                if self.cfg.fedbalancer or self.cfg.oortbalancer:
                    c.set_loss_threshold(self.fedbalancer.loss_threshold)

                # training
                logger.debug('client {} starts training...'.format(c.id))
                start_t = self.get_cur_time()
                
                # train on the client
                simulate_time_c, num_samples, update, acc, loss, update_size, sorted_loss, download_time, upload_time, train_time, inference_time, completed_epochs = c.train(start_t, num_epochs, batch_size)

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
                    self.clients_info[str(c.id)]["download_times"].append(download_time)
                    self.clients_info[str(c.id)]["upload_times"].append(upload_time)
                    self.clients_info[str(c.id)]["one_epoch_train_time"] = train_time/completed_epochs

                    # In case of using Oort-based client selection, when we do not use pacer, the round ends with the deadline.
                    # We calculate the curr_round_exploited_utility from the successful clients at the round.
                    if (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) and not self.cfg.oort_pacer:
                        self.oort.curr_round_exploited_utility += self.clients_info[str(c.id)]["utility"]

                    # calculate succeeded client's round duration for oort pacer
                    if self.cfg.oort_pacer and (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer) :
                        self.clients_info[str(c.id)]["last_selected_round_duration"] = download_time + train_time + upload_time + inference_time
                    
                    # TODO: MOVE THIS TO CLIENT
                    # TODO: MOVE THIS TO CLIENT
                    # TODO: MOVE THIS TO CLIENT
                    # TODO: MOVE THIS TO CLIENT
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
                                if sorted_loss[len(sorted_loss)-1-loss_idx] > c.loss_threshold:
                                    summ += sorted_loss[len(sorted_loss)-1-loss_idx]*sorted_loss[len(sorted_loss)-1-loss_idx]
                                    overthreshold_loss_count += 1
                            self.clients_info[str(c.id)]["overthreshold_loss_sum"] = summ
                            self.clients_info[str(c.id)]["overthreshold_loss_count"] = overthreshold_loss_count
                        if self.cfg.fedbalancer or self.cfg.oortbalancer:
                            self.fedbalancer.if_any_client_sent_response_for_current_round = True
                            noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                            noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                            self.fedbalancer.current_round_loss_min.append(np.min(sorted_loss)+noise1)
                            self.fedbalancer.current_round_loss_max.append(np.percentile(sorted_loss, 80)+noise2)

                    logger.debug('client {} simulate_time: {}'.format(c.id, simulate_time_c))
                    logger.debug('client {} num_samples: {}'.format(c.id, num_samples))
                    logger.debug('client {} acc: {}, loss: {}'.format(c.id, acc, loss))
                    accs.append(acc)
                    losses.append(loss)
                    
                    simulate_time = min(self.deadline, max(simulate_time, simulate_time_c))
                    # uploading 
                    self.updates.append((c.id, num_samples, update))

                    logger.debug('client {} upload successfully with acc {}, loss {}'.format(c.id,acc,loss))
            except timeout_decorator.timeout_decorator.TimeoutError as e:
                logger.debug('client {} failed: {}'.format(c.id, e))
                round_failed_clients.append(c.id)
                
                # if self.cfg.fedbalancer or self.cfg.oortbalancer:
                #     # in FedBalancer algorithm, this is done in client-side.
                #     # doing this here makes no difference in performance and privacy (because we do not use sample-level information other than calculating the overthreshold sum and count)
                #     # but we will patch this soon to move this to client
                #     if len(c.sorted_loss) != 0:
                #         # self.client_loss_count[str(c.id)] = len(c.sorted_loss)
                #         summ = 0
                #         overthreshold_loss_count = 0

                #         for loss_idx in range(len(c.sorted_loss)):
                #             if c.sorted_loss[len(c.sorted_loss)-1-loss_idx] > c.loss_threshold:
                #                 summ += c.sorted_loss[len(c.sorted_loss)-1-loss_idx]*c.sorted_loss[len(c.sorted_loss)-1-loss_idx]
                #                 overthreshold_loss_count += 1
                #         if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                #             self.clients_info[str(c.id)]["overthreshold_loss_sum"] = summ
                #             self.clients_info[str(c.id)]["overthreshold_loss_count"] = overthreshold_loss_count
                        
                #         noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                #         noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                #         self.fedbalancer.current_round_loss_min.append(np.min(c.sorted_loss)+noise1)
                #         self.fedbalancer.current_round_loss_max.append(np.percentile(c.sorted_loss, 80)+noise2)
                simulate_time = self.deadline
                
                if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
                    assert(False)
            except Exception as e:
                logger.error('client {} failed: {}'.format(c.id, e))
                traceback.print_exc()
            
            c.model = None
            c._model = None

        # These are the two cases which the round does not end with deadline; it ends when predefined number of clients succeed in a round
        # Thus, these two cases are handled separately from other cases
        # We sort the client round completion time, and aggregate the result of faster clients within the predefined number of succeeded clients in a round
        if self.cfg.oort_pacer or self.cfg.ddl_baseline_smartpc:
            client_simulate_times = sorted(client_simulate_times, key=lambda tup: tup[1])
            # for i, client_simulate_time in enumerate(client_simulate_times):
            #     print(i, client_simulate_time)
            for_loop_until = 0
            if self.cfg.oort_pacer:
                for_loop_until = min(self.cfg.clients_per_round, len(client_simulate_times))
            elif self.cfg.ddl_baseline_smartpc:
                for_loop_until = int(min(self.cfg.clients_per_round, len(client_simulate_times)) * self.cfg.ddl_baseline_smartpc_percentage)
            for c_idx in range(for_loop_until):
                c_id = client_simulate_times[c_idx][0]
                self.clients_info[str(c.id)]["download_times"].append(client_tmp_info[c_id]['download_time'])
                self.clients_info[str(c.id)]["upload_times"].append(client_tmp_info[c_id]['upload_time'])
                self.clients_info[str(c.id)]["one_epoch_train_time"] = client_tmp_info[c_id]['train_time']/client_tmp_info[c_id]['completed_epochs']

                # calculate succeeded client's round duration for oort pacer
                if self.cfg.oort_pacer and (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer):
                    self.clients_info[str(c.id)]["last_selected_round_duration"] = download_time + train_time + upload_time + inference_time
                
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
                            loss_threshold = 0
                            if self.cfg.fb_client_selection or self.cfg.oortbalancer:
                                loss_threshold = self.fedbalancer.loss_threshold
                            else:
                                loss_threshold = 0
                            if client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx] > loss_threshold:
                                summ += client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx]*client_tmp_info[c_id]['sorted_loss'][len(client_tmp_info[c_id]['sorted_loss'])-1-loss_idx]
                                overthreshold_loss_count += 1
                        self.clients_info[c_id]["overthreshold_loss_sum"] = summ
                        self.clients_info[c_id]["overthreshold_loss_count"] = overthreshold_loss_count

                    if self.cfg.fedbalancer or self.cfg.oortbalancer:
                        # self.current_round_losses = self.current_round_losses + client_tmp_info[c_id]['sorted_loss']
                        self.fedbalancer.if_any_client_sent_response_for_current_round = True
                        noise1 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        noise2 = np.random.normal(0, self.cfg.noise_factor, 1)[0]
                        self.fedbalancer.current_round_loss_min.append(np.min(client_tmp_info[c_id]['sorted_loss'])+noise1)
                        self.fedbalancer.current_round_loss_max.append(np.percentile(client_tmp_info[c_id]['sorted_loss'], 80)+noise2)

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

                # uploading 
                self.updates.append((c_id, client_tmp_info[c_id]['num_samples'], client_tmp_info[c_id]['update']))

                logger.debug('client {} upload successfully with acc {}, loss {}'.format(c_id,client_tmp_info[c_id]['acc'], client_tmp_info[c_id]['loss']))
        try:
            # logger.info('simulation time: {}'.format(simulate_time))
            # sys_metrics['configuration_time'] = simulate_time
            avg_acc = sum(accs)/len(accs)
            avg_loss = sum(losses)/len(losses)
            logger.info('average acc: {}, average loss: {}'.format(avg_acc, avg_loss))
            logger.info('configuration and update stage simulation time: {}'.format(simulate_time))

            # In case of using Oort-based client selection, when we do not use pacer, the round ends with the deadline.
            # We calculate the curr_round_exploited_utility from the successful clients at the round.
            if not self.cfg.oort_pacer:
                if self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer:
                    self.oort.round_exploited_utility.append(self.oort.curr_round_exploited_utility)
            
            # Update the loss threshold percentage (ratio) and the deadline percentage (ratio) 
            # according to the Algorithm 3 in FedBalancer paper
            if self.cfg.fedbalancer or self.cfg.oortbalancer:
                current_round_loss = (sorted_loss_sum / num_of_samples) / (self.deadline)
                self.fedbalancer.prev_train_losses.append(current_round_loss)
                logger.info('current_round_loss: {}'.format(current_round_loss))
                    # else:
                    #     self.loss_threshold = min_loss
                
                self.fedbalancer.ratio_control(self.fedbalancer.prev_train_losses, self.current_round)

                logger.info('min sample loss: {}, max sample loss: {}'.format(min_loss, max_loss))
            
            self.current_round += 1
            # logger.info('losses: {}'.format(losses))
        except ZeroDivisionError as e:
            logger.error('training time window is too short to train!')
            # assert False
        except Exception as e:
            logger.error('failed reason: {}'.format(e))
            traceback.print_exc()
            assert False

        # If everyone is at least once selected for a round, we set epsilon as zero, which is the exploration-exploitation parameter of Oort
        if (self.cfg.fb_client_selection or self.cfg.oort or self.cfg.oortbalancer):
            
            if len(self.failed_clients) != 0:
                again_failed_clients = 0
                for fc_id in self.failed_clients[-1]:
                    for r_fc_id in round_failed_clients:
                        if fc_id == r_fc_id:
                            again_failed_clients += 1
                logger.info("AGAIN FAILED CLIENTS:" + str(again_failed_clients))
                if again_failed_clients > self.cfg.clients_per_round * 0.1:
                    self.fedbalancer.guard_time += 10
                else:
                    self.fedbalancer.guard_time = 0
            self.failed_clients.append(round_failed_clients)

            if self.oort.epsilon != 0:
                self.oort.zero_epsilon(self.all_clients, self.clients_info)

        return simulate_time
        
    def update_model(self, update_frac):
        logger.info('{} of {} clients upload successfully'.format(len(self.updates), len(self.selected_clients)))
        if len(self.updates) / len(self.selected_clients) >= update_frac:        
            logger.info('round succeed, updating global model...')
            if self.cfg.no_training:
                logger.info('pseduo-update because of no_training setting.')
                self.updates = []
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
            # if self.fedbalancer != None:
            #     self.fedbalancer.guard_time = 0
        else:
            logger.info('round failed, global model maintained.')
            if self.fedbalancer != None:
                self.fedbalancer.guard_time += 10
        self.updates = []
        
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
                self.test_clients_info[client.id]['acc'] = c_metrics['accuracy'].tolist()
            else:
                self.test_clients_info[client.id]['acc'] = c_metrics['accuracy']
            client.model = None
            client._model = None
        
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
