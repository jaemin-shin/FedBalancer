import json
import numpy as np
import os
from collections import defaultdict
import torch
import importlib
from utils.logger import Logger

L = Logger()
logger = L.get_logger()

def batch_data(data, data_idx, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''

    # randomly shuffle data
    # np.random.seed(seed)
    data_y = data['y'].detach().numpy()
    data_x = data['x'].detach().numpy()

    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)
    np.random.set_state(rng_state)
    np.random.shuffle(data_idx)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        batched_idx = data_idx[i:i+batch_size]
        yield (batched_x, batched_y, batched_idx)

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data_return_all(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    return train_clients, train_groups, train_data, test_clients, test_groups, test_data

def build_net(dataset,model_name,num_classes):
    model_file="%s/%s.py" %(dataset,model_name)
    if not os.path.exists(model_file):
        print("Please specify a valid model")
    model_path="%s.%s" %(dataset,model_name)
    #build net
    mod=importlib.import_module(model_path)
    build_net_op=getattr(mod,"build_net")
    net=build_net_op(num_classes)
    return net