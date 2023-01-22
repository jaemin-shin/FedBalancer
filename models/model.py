"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import math

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.logger import Logger

import torch

L = Logger()
logger = L.get_logger()

class Model(ABC):
    init_cnt = 0

    def __init__(self, seed, lr):

        self.lr = lr
        self.seed = seed
        
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.DEVICE)
        
        self.net, self.losses, self.optimizer, self.optimizer_args = self.create_model()

        self.size = sum(p.numel() for p in self.net.parameters()) * 4 # 4byte per parameter

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None

    def train(self, data, data_idx, num_epochs=1, batch_size=10):
        """
        Trains the client model.
        """

        data_loss_list_and_idx = []
        
        for i in range(num_epochs):
            if i == 0:
                epoch_result = self.run_epoch(data, data_idx, batch_size, True)
                data_loss_list_and_idx = epoch_result['data_loss_list_and_idx']
            else:
                self.run_epoch(data, data_idx, batch_size, False)

        train_reslt = self.test(data)
        acc = train_reslt[ACCURACY_KEY]
        loss = train_reslt['loss']
        
        update = self.net.state_dict()

        return update, acc, loss, data_loss_list_and_idx
    
    def oorttrain(self, data_idx, xss, yss, num_epochs=1, batch_size=10, oortbalancer=False):
        """
        Trains the client model.
        """
        
        representative_batched_data = {}

        data_loss_list_and_idx = []
        
        if oortbalancer:
            with torch.no_grad():
                order = np.array(range(len(xss)))
                np.random.shuffle(order)
                xss[np.array(range(len(xss)))] = xss[order]
                yss[np.array(range(len(xss)))] = yss[order]

        for i in range(num_epochs):
            xs = xss[i*(int(len(xss)/num_epochs)):(i+1)*(int(len(xss)/num_epochs))]
            ys = yss[i*(int(len(xss)/num_epochs)):(i+1)*(int(len(xss)/num_epochs))]
            data = {'x': xs, 'y': ys}
            if i == 0:
                representative_batched_data = data
            # self.run_epoch(data, 1, False)
            epoch_result = self.run_epoch(data, data_idx[i*(int(len(xss)/num_epochs)):(i+1)*(int(len(xss)/num_epochs))], 1, True)
            data_loss_list_and_idx += epoch_result['data_loss_list_and_idx']

        train_reslt = self.test(representative_batched_data)
        acc = train_reslt[ACCURACY_KEY]
        loss = train_reslt['loss']
        
        update = self.net.state_dict()

        return update, acc, loss, data_loss_list_and_idx

    def run_epoch(self, data, data_idx, batch_size, require_loss_list):

        if require_loss_list:
            data_loss_list = np.zeros((len(data_idx)), dtype=np.float32)
            data_loss_list_idx = np.zeros((len(data_idx)), dtype=np.int32)

        current_data_loss_list_len = 0

        self.net.train()
        self.net = self.net.to(self.DEVICE)
        
        optimizer = self.optimizer(self.net.parameters(), **self.optimizer_args)

        for batched_x, batched_y, batched_idx in batch_data(data, data_idx, batch_size, seed=self.seed):

            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            # THIS IS ONLY FOR THE MOTIVATION EXPERIMENT
            # if len(input_data) < 10:
            #     continue

            self.last_features = input_data
            self.last_labels = target_data

            input_data = input_data.to(self.DEVICE).float()
            target_data = target_data.to(self.DEVICE).long()

            optimizer.zero_grad()
            y_hats = self.net(input_data)
            loss_list = self.losses(y_hats, target_data)
            loss = loss_list.mean()
            loss.backward()
            optimizer.step()

            loss_list = loss_list.cpu().detach().numpy()
            
            if require_loss_list:
                data_loss_list[current_data_loss_list_len:current_data_loss_list_len+len(loss_list)] = loss_list
                data_loss_list_idx[current_data_loss_list_len:current_data_loss_list_len+len(loss_list)] = batched_idx

                current_data_loss_list_len += len(loss_list)
        
        self.net = self.net.to("cpu")

        if require_loss_list:
            return {'data_loss_list_and_idx': list(zip(data_loss_list, data_loss_list_idx))}
        else:
            return {'data_loss_list_and_idx': []}
    
    @abstractmethod
    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        return None
    
    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass
