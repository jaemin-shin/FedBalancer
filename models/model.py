"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf
import math

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size
from utils.logger import Logger

L = Logger()
logger = L.get_logger()


class Model(ABC):
    init_cnt = 0

    def __init__(self, seed, lr, optimizer=None):

        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + self.seed)
            self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss, self.loss_list, self.flag_training, self.batch_gradients = self.create_model()
            self.saver = tf.train.Saver()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth=True
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(graph=self.graph, config=config)

        self.size = graph_size(self.graph)  

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

        np.random.seed(self.seed)
    
    def load_params(self, meta_path, ckpt_path):
        self.saver = tf.train.import_meta_graph(meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_path))

    def set_params(self, model_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params
    
    def get_gradients(self):
        logger.error('???')
        with self.graph.as_default():
            gradient_paras = tf.gradients(self.loss, tf.trainable_variables()[1:])
            if 'lstm' in self.model_name:
                gradients = self.sess.run(gradient_paras,
                                            feed_dict={
                                                self.features: self.last_features,
                                                self.labels: self.last_labels,
                                                self.sequence_length_ph: self.last_sequence_length_ph,
                                                self.sequence_mask_ph: self.last_sequence_mask_ph
                                            })
            else:
                gradients = self.sess.run(gradient_paras,
                                            feed_dict={
                                                self.features: self.last_features,
                                                self.labels: self.last_labels
                                            })
        return gradients
    
    def update_with_gradiant(self, gradients):
        params = self.get_params()
        for i in range(len(gradients)):
            params[i] -= gradients[i]*self.lr
        self.set_params(params)
        return params

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        return self._optimizer

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
        params_old= self.get_params()
        loss_old = self.test(data)['loss']

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
        
        update = self.get_params()
        comp = num_epochs * math.ceil(len(data['y'])/batch_size) * batch_size * self.flops

        grad = []
        for i in range(len(update)):
            grad.append((params_old[i] - update[i]) / self.lr)
        return comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx
    
    def realoorttrain(self, data, data_idx, xss, yss, num_epochs=1, batch_size=10):
        """
        Trains the client model.
        """

        params_old = self.get_params()
        loss_old = self.test(data)['loss']
        
        representative_batched_data = {}

        data_loss_list_and_idx = []

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
        
        update = self.get_params()
        comp = num_epochs * math.ceil(len(data['y'])/batch_size) * batch_size * self.flops

        grad = []
        for i in range(len(update)):
            grad.append((params_old[i] - update[i]) / self.lr)
        return comp, update, acc, loss, grad, loss_old, data_loss_list_and_idx

    def run_epoch(self, data, data_idx, batch_size, require_loss_list):

        if require_loss_list:
            data_loss_list = np.zeros((len(data_idx)), dtype=np.float32)
            data_loss_list_idx = np.zeros((len(data_idx)), dtype=np.int32)

        current_data_loss_list_len = 0

        for batched_x, batched_y, batched_idx in batch_data(data, data_idx, batch_size, seed=self.seed):

            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            # THIS IS ONLY FOR THE MOTIVATION EXPERIMENT
            # if len(input_data) < 10:
            #     continue

            self.last_features = input_data
            self.last_labels = target_data
            
            with self.graph.as_default():
                _, tot_acc, loss, loss_list, batch_gradients = self.sess.run([self.train_op, self.eval_metric_ops, self.loss, self.loss_list, self.batch_gradients],
                feed_dict={
                    self.features: input_data,
                    self.labels: target_data,
                    self.flag_training: True
                })
            
            if require_loss_list:
                data_loss_list[current_data_loss_list_len:current_data_loss_list_len+len(loss_list)] = loss_list
                data_loss_list_idx[current_data_loss_list_len:current_data_loss_list_len+len(loss_list)] = batched_idx

                current_data_loss_list_len += len(loss_list)

        acc = float(tot_acc) / input_data.shape[0]

        if require_loss_list:
            return {'acc': acc, 'loss': loss, 'data_loss_list_and_idx': list(zip(data_loss_list, data_loss_list_idx))}
        else:
            return {'acc': acc, 'loss': loss, 'data_loss_list_and_idx': []}

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, loss, loss_list = self.sess.run(
                [self.eval_metric_ops, self.loss, self.loss_list],
                feed_dict={self.features: x_vecs, self.labels: labels, self.flag_training: False}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, 'loss': loss, 'loss_list': loss_list}

    def close(self):
        self.sess.close()
    
    def get_comp(self, data, num_epochs=1, batch_size=10):
        comp = num_epochs * math.ceil(len(data['y'])/batch_size) * batch_size * self.flops
        return comp
        
    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
    
