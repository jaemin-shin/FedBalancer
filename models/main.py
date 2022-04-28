"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import time
import eventlet
import signal
import json
import traceback
import tensorflow as tf
from collections import defaultdict

# args
from utils.args import parse_args
eventlet.monkey_patch()
args = parse_args()
config_name = args.config

print(config_name)

# logger
from utils.logger import Logger
L = Logger()
L.set_log_name(config_name)
logger = L.get_logger()

from baseline_constants import MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.model_utils import read_data_return_all
from utils.config import Config
from device import Device

def main():

    # read config from file
    cfg = Config(config_name)

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + cfg.seed)
    np.random.seed(12 + cfg.seed)
    tf.compat.v1.set_random_seed(123 + cfg.seed)

    model_path = '%s/%s.py' % (cfg.dataset, cfg.model)
    if not os.path.exists(model_path):
        logger.error('Please specify a valid dataset and a valid model.')
        assert False
    model_path = '%s.%s' % (cfg.dataset, cfg.model)
    
    logger.info('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    
    num_rounds = cfg.num_rounds
    eval_every = cfg.eval_every
    clients_per_round = cfg.clients_per_round

    current_test_accuracy = 0.0

    # Oort (Fan Lai et al., OSDI'21) samples 1.3K clients per round and accepts until K clients send results
    if cfg.oort_pacer:
        clients_per_round = int(clients_per_round * 1.3)
    
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if cfg.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = cfg.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(cfg.seed, *model_params, cfg=cfg)
    logger.info('model size: {}'.format(client_model.size))

    # Create clients
    logger.info('======================Setup Clients==========================')
    train_clients, test_clients = setup_clients(cfg, client_model)

    # Calculate the initial deadline based on the mean of sampled round_duration
    # This deadline is fixed to be used for FedAvg or FedProx + 1T / 2T baselines
    # Include inference time when calculating the round duration time for FedBalancer and OortBalancer, as clients do full pass on their data when they are first selected
    fixed_deadline = 0
    oort_initial_deadline = 0
    fb_initial_deadline = 0
    if cfg.ddl_baseline_fixed or cfg.fb_client_selection or cfg.realoort or cfg.realoortbalancer or cfg.fedbalancer:
        round_duration_summ_list = []
        for c in train_clients:
            if cfg.realoortbalancer or cfg.fedbalancer:
                round_duration_summ_list.append(c.device.get_expected_download_time() + c.device.get_expected_upload_time(client_model.size) + np.mean(c.inference_times) + np.mean(c.per_epoch_train_times)*cfg.num_epochs)
            else:
                round_duration_summ_list.append(c.device.get_expected_download_time() + c.device.get_expected_upload_time(client_model.size) + np.mean(c.per_epoch_train_times)*cfg.num_epochs)
            if c.num_train_samples <= 0:
                assert(False)
        if cfg.ddl_baseline_fixed:
            fixed_deadline = int(np.mean(round_duration_summ_list) * cfg.ddl_baseline_fixed_value_multiplied_at_mean)
        if cfg.fb_client_selection or cfg.realoort:
            oort_initial_deadline = int(np.mean(round_duration_summ_list))
        if cfg.fedbalancer or cfg.realoortbalancer:
            fb_initial_deadline = int(np.mean(round_duration_summ_list))

    attended_clients = set()
    
    # Create server
    if cfg.ddl_baseline_fixed:
        server = Server(client_model, clients=train_clients, cfg = cfg, deadline=fixed_deadline)
    elif cfg.fedbalancer or cfg.realoortbalancer:
        server = Server(client_model, clients=train_clients, cfg = cfg, deadline=fb_initial_deadline)
    elif cfg.fb_client_selection or cfg.realoort:
        server = Server(client_model, clients=train_clients, cfg = cfg, deadline=oort_initial_deadline)
    else:
        server = Server(client_model, clients=train_clients, cfg = cfg, deadline=int(cfg.round_ddl[0]))
    
    client_ids, client_groups, client_num_samples = server.get_clients_info(train_clients)
    
    # Initial status
    logger.info('===================== Random Initialization =====================')

    # Simulate training
    if num_rounds == -1:
        import sys
        num_rounds = sys.maxsize
        
    def timeout_handler(signum, frame):
        raise Exception
    
    def exit_handler(signum, frame):
        os._exit(0)
    
    for i in range(num_rounds):
        logger.info('===================== Round {} of {} ====================='.format(i+1, num_rounds))
        
        if cfg.oort_pacer:
            if server.current_round > server.pacer_window * 2:
                if sum(server.round_exploited_utility[-(2*server.pacer_window):-(server.pacer_window)]) > sum(server.round_exploited_utility[-(server.pacer_window):]):
                    logger.info('by oort pacer, the deadline is changed from {} to {}'.format(server.deadline, server.deadline+cfg.oort_pacer_delta))
                    server.deadline += cfg.oort_pacer_delta
        else:
            if server.current_round > server.pacer_window * 2:
                if sum(server.round_exploited_utility[-(2*server.pacer_window):-(server.pacer_window)]) > sum(server.round_exploited_utility[-(server.pacer_window):]):
                    logger.info('by oort, the oort_non_pacer_deadline is changed from {} to {}'.format(server.oort_non_pacer_deadline, server.oort_non_pacer_deadline+10))
                    server.deadline += 10
                    

        # 1. selection stage
        logger.info('--------------------- selection stage ---------------------')
        # 1.1 select clients
        cur_time = server.get_cur_time()
        time_window = server.get_time_window() 
        logger.info('current time: {}\ttime window: {}\t'.format(cur_time, time_window))
        
        if cfg.global_final_time != 0 and cur_time > cfg.global_final_time: 
            break

        if type(current_test_accuracy) is np.ndarray:
            if cfg.global_final_test_accuracy != 0.0 and current_test_accuracy[0] > cfg.global_final_test_accuracy: 
                break
        else:
            if cfg.global_final_test_accuracy != 0.0 and current_test_accuracy > cfg.global_final_test_accuracy: 
                break
            
        online_clients = online(train_clients, cur_time, time_window)
        if not server.select_clients(i, 
                              online_clients, 
                              num_clients=clients_per_round,
                              batch_size=cfg.batch_size):
            # insufficient clients to select, round failed
            logger.info('round failed in selection stage!')
            server.pass_time(time_window)
            continue
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        attended_clients.update(c_ids)
        c_ids.sort()
        logger.info("selected num: {}".format(len(c_ids)))
        logger.debug("selected client_ids: {}".format(c_ids))
        
        # 1.3 update simulation time
        server.pass_time(time_window)
        
        # 2. configuration stage
        logger.info('--------------------- configuration stage ---------------------')
        # 2.1 train(no parallel implementation)
        sys_metrics = server.train_model(num_epochs=cfg.num_epochs, batch_size=cfg.batch_size, minibatch=cfg.minibatch)
        
        # 2.2 update simulation time
        server.pass_time(sys_metrics['configuration_time'])
        
        # 3. update stage
        logger.info('--------------------- report stage ---------------------')
        # 3.1 update global model
        if cfg.compress_algo:
            logger.info('update using compressed grads')
            server.update_using_compressed_grad(cfg.update_frac)
        elif cfg.qffl:
            server.update_using_qffl(cfg.update_frac)
            logger.info('round success by using qffl')
        else:
            server.update_model(cfg.update_frac)

        # 4. Test model(if necessary)
        if eval_every == -1:
            continue
        
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            if cfg.no_training:
                continue
            logger.info('--------------------- test result ---------------------')
            test_num = len(test_clients)
            if (i + 1) % (10*eval_every) == 0 or (i + 1) == num_rounds:
                test_num = len(test_clients)
                config_name_split = config_name.split('/')
                with open(cfg.output_path+'/attended_clients/'+config_name_split[-1][:-4]+'_attended_clients.json', 'w') as fp:
                    json.dump(list(attended_clients), fp)
                    logger.info('save attended_clients.json')
                
                # Save server model
                ckpt_path = os.path.join('../models/checkpoints', cfg.dataset)
                ckpt_path = os.path.join(ckpt_path, config_name_split[-1][:-4])
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(config_name_split[-1][:-4])))
                logger.info('Model saved in path: %s' % save_path)
                
            test_clients = random.sample(test_clients, test_num) 
            sc_ids, sc_groups, sc_num_samples = server.get_clients_info(test_clients)

            test_stat_metrics = server.test_model(test_clients, set_to_use='test')
            current_test_accuracy = print_metrics(test_stat_metrics, sc_num_samples, prefix='test_')
            
            if (i + 1) % (10*eval_every) == 0 or (i + 1) == num_rounds:
                server.save_clients_info()
    
    logger.info('--------------------- FINAL test result ---------------------')
    test_num = len(test_clients)
    test_num = len(test_clients)
    config_name_split = config_name.split('/')
    with open(cfg.output_path+'/attended_clients/'+config_name_split[-1][:-4]+'_attended_clients.json', 'w') as fp:
        json.dump(list(attended_clients), fp)
        logger.info('save attended_clients.json')
    
    # Save server model
    ckpt_path = os.path.join('../models/checkpoints', cfg.dataset)
    ckpt_path = os.path.join(ckpt_path, config_name_split[-1][:-4])
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(config_name_split[-1][:-4])))
    logger.info('Model saved in path: %s' % save_path)
        
    test_clients = random.sample(test_clients, test_num) 
    sc_ids, sc_groups, sc_num_samples = server.get_clients_info(test_clients)

    test_stat_metrics = server.test_model(test_clients, set_to_use='test')
    current_test_accuracy = print_metrics(test_stat_metrics, sc_num_samples, prefix='test_')
    
    if (i + 1) % (10*eval_every) == 0 or (i + 1) == num_rounds:
        server.save_clients_info()
            
    # Close models
    server.close_model()

def online(clients, cur_time, time_window):
    # """We assume all users are always online."""
    # return online client according to client's timer
    online_clients = []
    for c in clients:
        try:
            if c.timer.ready(cur_time, time_window):
                online_clients.append(c)
        except Exception as e:
            traceback.print_exc()
    L = Logger()
    logger = L.get_logger()
    logger.info('{} of {} clients online'.format(len(online_clients), len(clients)))
    return online_clients

def create_train_and_test_clients(train_users, train_groups, train_data, test_users, test_groups, test_data, model, cfg):
    L = Logger()
    logger = L.get_logger()
    client_num = min(cfg.max_client_num, len(train_users))
    train_users = random.sample(train_users, client_num)
    
    logger.info('Train Clients in Total: %d' % (len(train_users)))
    logger.info('Test Clients in Total: %d' % (len(test_users)))

    if len(train_groups) == 0:
        train_groups = [[] for _ in train_users]
    if len(test_groups) == 0:
        test_groups = [[] for _ in test_users]

    cnt = 0
    train_clients = []
    for u, g in zip(train_users, train_groups):
        c = Client(u, g, train_data[u], test_data[u], model, Device(cfg, model_size=model.size), cfg)
        if len(c.train_data["x"]) == 0:
            continue
        # if len(c.train_data["x"]) < 10:
        #     continue
        train_clients.append(c)
        cnt += 1
        if cnt % 10 == 0:
            logger.info('set up {} clients'.format(cnt))

    test_clients = []
    for u, g in zip(test_users, test_groups):
        c = Client(u, g, train_data[u], test_data[u], model, Device(cfg, model_size=model.size), cfg)
        if len(c.eval_data["x"]) == 0:
            continue
        test_clients.append(c)

    from timer import Timer
    Timer.save_cache()
    model2cnt = defaultdict(int)
    for c in train_clients:
        model2cnt[c.get_device_model()] += 1
    logger.info('device setup result: {}'.format(model2cnt))
    return train_clients, test_clients

def setup_clients(cfg, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    #train_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', 'train')
    #test_data_dir = os.path.join('..', 'data', cfg.dataset, 'data', eval_set)
    train_data_dir = os.path.join('/mnt/sting/jmshin/FedBalancer/FLASH_jm/', 'data', cfg.dataset, 'data', 'train')
    test_data_dir = os.path.join('/mnt/sting/jmshin/FedBalancer/FLASH_jm/', 'data', cfg.dataset, 'data', eval_set)

    train_users, train_groups, train_data, test_users, test_groups, test_data = read_data_return_all(train_data_dir, test_data_dir)

    train_clients, test_clients = create_train_and_test_clients(train_users, train_groups, train_data, test_users, test_groups, test_data, model, cfg)

    #return clients
    return train_clients, test_clients

def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    client_ids = [c for c in sorted(metrics.keys())]
    ordered_weights = [weights[c] for c in client_ids]

    if len(metrics) == 0:
        metric_names = []
    else:
        metrics_dict = next(iter(metrics.values()))
        metric_names =  list(metrics_dict.keys())

    to_ret = None
    L = Logger()
    logger = L.get_logger()
    test_accuracy = 0.0
    for metric in metric_names:
        if metric == 'loss_list':
            continue
        ordered_metric = [metrics[c][metric] for c in client_ids]
        logger.info('{}: {}, 10th percentile: {}, 50th percentile: {}, 90th percentile {}'.format
                (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights, axis = 0),
                 np.percentile(ordered_metric, 10, axis=0),
                 np.percentile(ordered_metric, 50, axis=0),
                 np.percentile(ordered_metric, 90, axis=0)))
        if metric == 'accuracy':
            test_accuracy = np.average(ordered_metric, weights=ordered_weights, axis = 0)
    return test_accuracy

def output_current_round_deadline(selected_clients):
    t_max = sys.maxsize
    total_user_count = len(selected_clients)
    
    complete_user_counts_per_time = []
    max_complete_user_counts_per_time = -1
    max_complete_user_counts_per_time_idx = -1
    
    for i in range(1, t_max):
        complete_user_count = 0
        for c in selected_clients:
            if len(c.per_epoch_train_times) > 5:
                if np.mean(c.download_times) + np.mean(c.upload_times) + np.mean(c.per_epoch_train_times[-5:]) <= i:
                    complete_user_count += 1
            else:
                if np.mean(c.download_times) + np.mean(c.upload_times) + np.mean(c.per_epoch_train_times) <= i:
                    complete_user_count += 1
        complete_user_counts_per_time.append(complete_user_count/(i))
        
        if max_complete_user_counts_per_time < complete_user_count/(i):
            max_complete_user_counts_per_time = complete_user_count/(i)
            max_complete_user_counts_per_time_idx = i
        
        if complete_user_count == total_user_count:
            break
        
    return max_complete_user_counts_per_time_idx

if __name__ == '__main__':
    # nohup python main.py -dataset shakespeare -model stacked_lstm &
    start_time=time.time()
    main()
    # logger.info("used time = {}s".format(time.time() - start_time))