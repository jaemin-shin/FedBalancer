import numpy as np
import torch

def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0] # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i]) # output a flattened array

    return np.sum(np.square(client_grads))


def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads
