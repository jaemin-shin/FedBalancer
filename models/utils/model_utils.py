import json
import numpy as np
import os
from collections import defaultdict
import torch
import importlib


def batch_data(data, data_idx, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_y = data['y'].detach().numpy()
    data_x = data['x'].detach().numpy()

    # data_y = data['y']
    # data_x = data['x']

    # randomly shuffle data
    np.random.seed(seed)
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

# def batch_data_multiple_iters(data, batch_size, num_iters):
#     data_x = data['x']
#     data_y = data['y']

#     np.random.seed(100)
#     rng_state = np.random.get_state()
#     np.random.shuffle(data_x)
#     np.random.set_state(rng_state)
#     np.random.shuffle(data_y)

#     idx = 0

#     for i in range(num_iters):
#         if idx+batch_size >= len(data_x):
#             idx = 0
#             rng_state = np.random.get_state()
#             np.random.shuffle(data_x)
#             np.random.set_state(rng_state)
#             np.random.shuffle(data_y)
#         batched_x = data_x[idx: idx+batch_size]
#         batched_y = data_y[idx: idx+batch_size]
#         idx += batch_size
#         yield (batched_x, batched_y)

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

def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if cuda:
        m_parameter = torch.Tensor([0]).cuda()
    else:
        m_parameter = torch.Tensor([0])
    for parameter in list(model.cpu().parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]
    
def unravel_model_params(model, parameter_update):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in grad_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from grad_update
    for p in model.parameters():
        numel = p.data.numel()
        size = p.data.size()
        p.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel