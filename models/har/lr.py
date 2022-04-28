import tensorflow as tf

from model import Model
from fedprox import PerturbedGradientDescent
import numpy as np
import os


INPUT_SIZE = 561


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, cfg=None):
        self.num_classes = num_classes
        self.model_name = os.path.abspath(__file__)
        if cfg.fedprox:
            super(ClientModel, self).__init__(seed, lr, optimizer=PerturbedGradientDescent(lr, cfg.fedprox_mu))
        else:
            super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for LR."""
        features = tf.placeholder(
            tf.float32, shape=[None, INPUT_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        flag_training = tf.placeholder(tf.bool)

        logits = tf.layers.dense(inputs=features, units=self.num_classes)
        
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss_list = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.NONE)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))

        batch_gradients = []

        return features, labels, train_op, eval_metric_ops, loss, loss_list, flag_training, batch_gradients

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
