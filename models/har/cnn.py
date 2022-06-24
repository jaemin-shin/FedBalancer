import tensorflow as tf

from model import Model
from fedprox import PerturbedGradientDescent
import numpy as np
import os


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, cfg=None):
        self.num_classes = num_classes
        self.model_name = os.path.abspath(__file__)

        if cfg.fedprox:
            super(ClientModel, self).__init__(seed, lr, optimizer=PerturbedGradientDescent(lr, cfg.fedprox_mu))
        else:
            super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, 128 * 9], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        flag_training = tf.placeholder(tf.bool)
        input_layer = tf.reshape(features, [-1, 128, 9])
        # conv1 = tf.layers.conv1d(
        #   inputs=input_layer,
        #   filters=64,
        #   kernel_size=[3],
        #   padding="same",
        #   activation=tf.nn.relu)
        # conv2 = tf.layers.conv1d(
        #     inputs=conv1,
        #     filters=64,
        #     kernel_size=[3],
        #     padding="same",
        #     activation=tf.nn.relu)
        # pool1 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[2], strides=2)
        # pool1_flat = tf.reshape(pool1, [-1, 64 * 64])
        conv1 = tf.layers.conv1d(
          inputs=input_layer,
          filters=192,
          kernel_size=[16],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[4], strides=4)
        pool1_flat = tf.reshape(pool1, [-1, 32 * 192])
        # dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
        dense = tf.layers.dense(inputs=pool1_flat, units=256, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        # loss_list = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.NONE)
        # sum_loss = tf.reduce_sum(loss_list)
        loss_list = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.NONE)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        batch_gradients = []

        # if self.cfg.ss_baseline:
        #     batch_gradients
        #     for l_idx in range(loss_list.shape[0].value):
        #         batch_gradients.append(tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[l_idx])]))
        
        # TODO: Confirm that opt initialized once is ok?
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))

        # TODO: Only enable when experimenting sample selection baseline
        # print(loss_list.shape)



        # if self.cfg.ss_baseline:
        #     batch_gradients = tf.map_fn(fn=lambda t: tf.global_norm([self.optimizer.compute_gradients(t)[0][0]]), elems=loss_list)
            # [self.optimizer.compute_gradients(elem)[0] for elem in loss_list]
            # for l_idx in range(loss_list.shape[0].value):
            #     batch_gradients.append(tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[l_idx])]))
        
        # batch_gradients = [tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[0])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[1])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[2])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[3])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[4])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[5])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[6])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[7])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[8])]),
        # tf.global_norm([grad for grad, variable in self.optimizer.compute_gradients(loss_list[9])])]

        return features, labels, train_op, eval_metric_ops, loss, loss_list, flag_training, batch_gradients

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)