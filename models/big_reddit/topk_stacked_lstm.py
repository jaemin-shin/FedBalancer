import collections
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import copy

from tensorflow.contrib import rnn

from model import Model
from fedprox import PerturbedGradientDescent

VOCABULARY_PATH = '/mnt/sting/jmshin/FedBalancer/FLASH_jm/data/big_reddit/vocab/reddit_vocab.pck'

# Code adapted from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
# and https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
class ClientModel(Model):
    def __init__(self, seed, lr, seq_len, n_hidden, num_layers,
        keep_prob=1.0, max_grad_norm=5, init_scale=0.1, cfg=None):

        self.seq_len = seq_len
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.max_grad_norm = max_grad_norm

        self.model_name = os.path.abspath(__file__)

        # initialize vocabulary
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self.load_vocab()
        print('vocab_size: {}'.format(self.vocab_size))

        self.initializer = tf.random_uniform_initializer(-init_scale, init_scale)

        self.cfg = cfg

        if cfg.fedprox:
            super(ClientModel, self).__init__(seed, lr, optimizer=PerturbedGradientDescent(lr, cfg.fedprox_mu))
        else:
            super(ClientModel, self).__init__(seed, lr)

    def create_model(self):

        with tf.variable_scope('language_model', reuse=None, initializer=self.initializer):
            features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
            labels = tf.placeholder(tf.int32, [None, self.seq_len], name='labels')
            flag_training = tf.placeholder(tf.bool)
            self.sequence_length_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.sequence_mask_ph = tf.placeholder(tf.float32, [None, self.seq_len], name='seq_mask_ph')

            self.batch_size = tf.shape(features)[0]

            # word embedding
            embedding = tf.get_variable(
                'embedding', [self.vocab_size, self.n_hidden], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, features)

            # LSTM
            output, state = self._build_rnn_graph(inputs) # TODO: check!

            # softmax
            with tf.variable_scope('softmax'):
                softmax_w = tf.get_variable(
                    'softmax_w', [self.n_hidden, self.vocab_size], dtype=tf.float32)
                softmax_b = tf.get_variable('softmax_b', [self.vocab_size], dtype=tf.float32)
            
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            
            # to calculate top 1, 3, 5 acc
            self.labels_reshaped = tf.reshape(labels, [-1])
            self.top_5_indices = tf.nn.top_k(logits, k = 5).indices
            
            '''
            unk_tensor = tf.fill(tf.shape(labels_reshaped) , self.unk_symbol)
            pred_unk = tf.cast(tf.equal(labels_reshaped, unk_tensor), tf.int32)
            labels_rm_unk = tf.add(labels_reshaped, tf.multiply(pred_unk, [tf.shape(logits)[1]]))
            pad_tensor = tf.fill(tf.shape(labels_reshaped) , self.pad_symbol)
            pred_pad = tf.cast(tf.equal(labels_reshaped, pad_tensor), tf.int32)
            labels_rm_pad = tf.add(labels_rm_unk, tf.multiply(pred_pad, [tf.shape(logits)[1]]))
            # self.labels_rm_pad = tf.add(labels_rm_unk, tf.multiply(10*pred_pad, [tf.shape(logits)[1]]))
            
            
            # self.logits_shape = tf.shape(logits) # [50, 10000]
            # self.labels_reshaped_shape = tf.shape(labels_reshaped) [50]

            # pred = tf.cast(tf.argmax(logits, 1), tf.int32)
            # self.w_pred = tf.nn.in_top_k(logits, labels_rm_pad, k = 1)
            # self.ww_pred = tf.nn.in_top_k(logits, labels_reshaped, k = 1)
            w_pred = tf.nn.in_top_k(logits, labels_rm_pad, k = 1)
            top3_pred = tf.nn.in_top_k(logits, labels_rm_pad, k = 3)
            top5_pred = tf.nn.in_top_k(logits, labels_rm_pad, k = 5)
            self.w_correct_pred = tf.cast(w_pred, tf.int32)
            w_correct_pred = tf.cast(w_pred, tf.int32)
            top3_correct_pred = tf.cast(top3_pred, tf.int32)
            top5_correct_pred = tf.cast(top5_pred, tf.int32)
            '''

            # correct predictions
            labels_reshaped = tf.reshape(labels, [-1])
            pred = tf.cast(tf.argmax(logits, 1), tf.int32)
            correct_pred = tf.cast(tf.equal(pred, labels_reshaped), tf.int32)
            
            # predicting unknown is always considered wrong
            unk_tensor = tf.fill(tf.shape(labels_reshaped), self.unk_symbol)
            pred_unk = tf.cast(tf.equal(pred, unk_tensor), tf.int32)
            correct_unk = tf.multiply(pred_unk, correct_pred)

            # predicting padding is always considered wrong
            pad_tensor = tf.fill(tf.shape(labels_reshaped), self.pad_symbol)
            pred_pad = tf.cast(tf.equal(pred, pad_tensor), tf.int32)
            correct_pad = tf.multiply(pred_pad, correct_pred)
            

            # Reshape logits to be a 3-D tensor for sequence loss
            logits = tf.reshape(logits, [-1, self.seq_len, self.vocab_size])

            # Use the contrib sequence loss and average over the batches
            loss_list = tf.contrib.seq2seq.sequence_loss(
                logits,
                labels,
                weights=self.sequence_mask_ph,
                average_across_timesteps=True,
                average_across_batch=False)
            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                labels,
                weights=self.sequence_mask_ph,
                average_across_timesteps=False,
                average_across_batch=True)

            # Update the cost
            #self.cost = tf.reduce_sum(loss)
            self.cost = tf.reduce_mean(loss)
            self.final_state = state

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            train_op = self.optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())
            
            batch_gradients = []

            # eval_metric_ops = [ tf.count_nonzero(correct_pred) - tf.count_nonzero(correct_unk) - tf.count_nonzero(correct_pad),
            #                     tf.count_nonzero(top3_correct_pred) - tf.count_nonzero(top3_correct_unk) - tf.count_nonzero(top3_correct_pad),
            #                    tf.count_nonzero(top5_correct_pred) - tf.count_nonzero(top5_correct_unk) - tf.count_nonzero(top5_correct_pad) ]

            # eval_metric_ops = [tf.count_nonzero(correct_pred), tf.count_nonzero(top3_correct_pred), tf.count_nonzero(top5_correct_pred)]
            # eval_metric_ops = [tf.count_nonzero(w_correct_pred), tf.count_nonzero(correct_pred) - tf.count_nonzero(correct_unk) - tf.count_nonzero(correct_pad)]
            eval_metric_ops = tf.count_nonzero(correct_pred) - tf.count_nonzero(correct_unk) - tf.count_nonzero(correct_pad)
        
        return features, labels, train_op, eval_metric_ops, self.cost, loss_list, flag_training, batch_gradients

    def _build_rnn_graph(self, inputs):
        def make_cell():
            cell = tf.contrib.rnn.LSTMBlockCell(self.n_hidden, forget_bias=0.0)
            if self.keep_prob < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [make_cell() for _ in range(self.num_layers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=self.initial_state, sequence_length=self.sequence_length_ph)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])
        return output, state

    def process_x(self, raw_x_batch):
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        lengths = np.sum(tokens != self.pad_symbol, axis=1)
        return tokens, lengths

    def process_y(self, raw_y_batch):
        tokens = self._tokens_to_ids([s for s in raw_y_batch])
        return tokens

    def _tokens_to_ids(self, raw_batch):
        def tokens_to_word_ids(tokens, vocab):
            return [vocab[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, self.vocab) for seq in raw_batch]
        return np.array(to_ret)

    def batch_data(self, data, data_idx, batch_size):
        data_x = data['x']
        data_y = data['y']

        perm = np.random.permutation(len(data['x']))
        data_x = [data_x[i] for i in perm]
        data_y = [data_y[i] for i in perm]
        data_idx_shuffled = [data_idx[i] for i in perm]

        # print("data_x:", len(data_x))
        # print(data_x)
        # print("data_y:", len(data_y))
        # print(data_y)

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            if len(data_x_by_seq) % batch_size != 0:
                dummy_tokens = [self.pad_symbol for _ in range(self.seq_len)]
                dummy_mask = [0 for _ in range(self.seq_len)]
                num_dummy = batch_size - len(data_x_by_seq) % batch_size

                data_x_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                data_y_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                mask_by_seq.extend([dummy_mask for _ in range(num_dummy)])

            return data_x_by_seq, data_y_by_seq, mask_by_seq
        
        data_x, data_y, data_mask = flatten_lists(data_x, data_y)

        # print("data_x:", len(data_x))
        # print(data_x)
        # print("data_y:", len(data_y))
        # print(data_y)

        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i+batch_size]
            batched_y = data_y[i:i+batch_size]
            batched_mask = data_mask[i:i+batch_size]
            batched_idx = data_idx_shuffled[i:i+batch_size]

            input_data, input_lengths = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            yield (input_data, target_data, input_lengths, batched_mask, batched_idx)
    
    def batch_data_for_test(self, data, batch_size):
        data_x = data['x']
        data_y = data['y']

        # perm = np.random.permutation(len(data['x']))
        # data_x = [data_x[i] for i in perm]
        # data_y = [data_y[i] for i in perm]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            if len(data_x_by_seq) % batch_size != 0:
                dummy_tokens = [self.pad_symbol for _ in range(self.seq_len)]
                dummy_mask = [0 for _ in range(self.seq_len)]
                num_dummy = batch_size - len(data_x_by_seq) % batch_size

                data_x_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                data_y_by_seq.extend([dummy_tokens for _ in range(num_dummy)])
                mask_by_seq.extend([dummy_mask for _ in range(num_dummy)])

            return data_x_by_seq, data_y_by_seq, mask_by_seq
        
        data_x, data_y, data_mask = flatten_lists(data_x, data_y)

        for i in range(0, len(data_x), batch_size):
            batched_x = data_x[i:i+batch_size]
            batched_y = data_y[i:i+batch_size]
            batched_mask = data_mask[i:i+batch_size]

            input_data, input_lengths = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            yield (input_data, target_data, input_lengths, batched_mask)

    def run_epoch(self, data, data_idx, batch_size, require_loss_list):
        if require_loss_list:
            # data_len = len(data)
            data_loss_list = np.zeros((len(data_idx)), dtype=np.float32)
            data_loss_list_idx = np.zeros((len(data_idx)), dtype=np.int32)
        
        current_data_loss_list_len = 0

        state = None

        fetches = {
            'cost': self.cost,
            'final_state': self.final_state,
        }

        for input_data, target_data, input_lengths, input_mask, batched_idx in self.batch_data(data, data_idx, batch_size):

            feed_dict = {
                self.features: input_data,
                self.labels: target_data,
                self.sequence_length_ph: input_lengths,
                self.sequence_mask_ph: input_mask,
            }

            self.last_features = input_data
            self.last_labels = target_data

            # We need to feed the input data so that the batch size can be inferred.
            if state is None:
                state = self.sess.run(self.initial_state, feed_dict=feed_dict)

            for i, (c, h) in enumerate(self.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            with self.graph.as_default():
                _, loss_list, vals = self.sess.run([self.train_op, self.loss_list, fetches], feed_dict=feed_dict)
            
            if require_loss_list:
                data_loss_list[current_data_loss_list_len:min(len(data_loss_list), current_data_loss_list_len+len(loss_list))] = loss_list[:min(len(data_loss_list), current_data_loss_list_len+len(loss_list)) - current_data_loss_list_len]
                data_loss_list_idx[current_data_loss_list_len:min(len(data_loss_list), current_data_loss_list_len+len(loss_list))] = batched_idx[:min(len(data_loss_list), current_data_loss_list_len+len(loss_list)) - current_data_loss_list_len]
                current_data_loss_list_len += min(len(data_loss_list), current_data_loss_list_len+len(loss_list)) - current_data_loss_list_len

            state = vals['final_state']
        
        if require_loss_list:
            return {'data_loss_list_and_idx': list(zip(data_loss_list, data_loss_list_idx))}
        else:
            return {'data_loss_list_and_idx': []}
    
    def zeropadding_count(self, input_data):
        seq_len = len(input_data[0])
        sum_data = np.sum(input_data, axis=1)
        return np.count_nonzero(sum_data == seq_len)
    
    def fedbalancer_xy_processing(self, data):
        result_x = []
        result_y = []
        train_data_x = data["x"]
        train_data_y = data["y"]

        for idx in range(len(train_data_y)):
            x_data = train_data_x[idx]
            chunk = train_data_y[idx]
            for i in range(len(chunk["target_tokens"])):
                result_x.append([x_data[i]])
                chunk_new = copy.deepcopy(chunk)
                chunk_new["target_tokens"] = [chunk["target_tokens"][i]]
                chunk_new["count_tokens"] = [chunk["count_tokens"][i]]
                result_y.append(chunk_new)
        
        return {"x": tuple(result_x), "y": tuple(result_y)}

    def test(self, data, batch_size=5):
        # tot_acc, tot_samples = 0, 0
        tot_acc = np.array([0.0] * 6,dtype=float)
        tot_samples = 0
        tot_loss, tot_batches = 0, 0
        sample_loss = []
        # zp_count = 0
        sample_len = len(data['x'])

        if self.cfg.fedbalancer or (self.cfg.ss_baseline):
            for input_data, target_data, input_lengths, input_mask in self.batch_data_for_test(data, batch_size):
                # zp_count += self.zeropadding_count(input_data)
                with self.graph.as_default():
                    acc, targets, top_5_indices, loss, loss_list = self.sess.run(
                        [self.eval_metric_ops, self.labels_reshaped, self.top_5_indices, self.loss, self.loss_list], 
                        feed_dict={
                            self.features: input_data,
                            self.labels: target_data,
                            self.sequence_length_ph: input_lengths, 
                            self.sequence_mask_ph: input_mask,
                        })
                    ks = [1,3,5]
                    for k in ks:
                        for i in range(len(targets)):
                            target = targets[i]
                            if target in top_5_indices[i][:k] and target != self.unk_symbol and target != self.pad_symbol:
                                tot_acc[k] += 1
                tot_acc[0] += acc
                tot_samples += np.sum(input_lengths)

                tot_loss += loss
                tot_batches += 1
                sample_loss.extend(loss_list)
        else:
            for input_data, target_data, input_lengths, input_mask in self.batch_data_for_test(data, batch_size):
                # zp_count += self.zeropadding_count(input_data)
                with self.graph.as_default():
                    acc, targets, top_5_indices, loss, loss_list = self.sess.run(
                        [self.eval_metric_ops, self.labels_reshaped, self.top_5_indices, self.loss, self.loss_list], 
                        feed_dict={
                            self.features: input_data,
                            self.labels: target_data,
                            self.sequence_length_ph: input_lengths, 
                            self.sequence_mask_ph: input_mask,
                        })
                    ks = [1,3,5]
                    for k in ks:
                        for i in range(len(targets)):
                            target = targets[i]
                            if target in top_5_indices[i][:k] and target != self.unk_symbol and target != self.pad_symbol:
                                tot_acc[k] += 1
                tot_acc[0] += acc
                tot_samples += np.sum(input_lengths)

                tot_loss += loss
                tot_batches += 1
                sample_loss.extend(loss_list)
        acc = tot_acc / tot_samples # this top 1 accuracy considers every pred. of unknown and padding as wrong
        loss = tot_loss / tot_batches # the loss is already averaged over samples
        return {'accuracy': acc, 'loss': loss, 'loss_list': sample_loss[:sample_len]}
        # if zp_count > 0:
        #     return {'accuracy': acc, 'loss': loss, 'loss_list': sample_loss[:-zp_count]}
        # else:
        #     return {'accuracy': acc, 'loss': loss, 'loss_list': sample_loss}


    def load_vocab(self):
        vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

