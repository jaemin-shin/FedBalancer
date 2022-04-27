import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.contrib import rnn

from model import Model
from fedprox import PerturbedGradientDescent
from utils.language_utils import letter_to_vec, word_to_indices

class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, num_classes, n_hidden, gpu_fraction=0.2, cfg=None):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden

        self.model_name = os.path.abspath(__file__)
        
        if cfg.fedprox:
            super(ClientModel, self).__init__(seed, lr, optimizer=PerturbedGradientDescent(lr, cfg.fedprox_mu))
        else:
            super(ClientModel, self).__init__(seed, lr)


    def create_model(self):
        features = tf.placeholder(tf.int32, [None, self.seq_len])
        embedding = tf.get_variable("embedding", [self.num_classes, 8])
        flag_training = tf.placeholder(tf.bool)
        x = tf.nn.embedding_lookup(embedding, features)
        labels = tf.placeholder(tf.int32, [None, self.num_classes])
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.layers.dense(inputs=outputs[:,-1,:], units=self.num_classes)
        
        loss_list = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.count_nonzero(correct_pred)
        batch_gradients = []

        # batch_gradients = [tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[0])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[1])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[2])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[3])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[4])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[5])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[6])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[7])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[8])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[9])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[10])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[11])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[12])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[13])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[14])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[15])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[16])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[17])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[18])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[19])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[20])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[21])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[22])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[23])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[24])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[25])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[26])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[27])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[28])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[29])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[30])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[31])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[32])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[33])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[34])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[35])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[36])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[37])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[38])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[39])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[40])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[41])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[42])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[43])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[44])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[45])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[46])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[47])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[48])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[49])])
        # ]

        return features, labels, train_op, eval_metric_ops, loss, loss_list, flag_training, batch_gradients

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch
