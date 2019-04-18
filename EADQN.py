#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
# Project:  Extracting Action Sequences Based on Deep Reinforcement Learning
# Module:   EADQN
# Author:   Wenfeng Feng 
# Time:     2017.12
################################################################################

import os
import ipdb
import numpy as np
import tensorflow as tf
from functools import reduce
from utils import save_pkl, load_pkl
from tensorflow.contrib.layers.python.layers import initializers

class DeepQLearner:
    """
    Deep Q-Network, in tensorflow
    """
    def __init__(self, args, sess, agent_mode):
        print('Initializing the DQN...')
        self.sess = sess
        self.use_pos = args.use_pos
        self.num_pos = args.num_pos
        self.pos_dim = args.pos_dim
        self.tag_dim = args.tag_dim
        self.num_filters = args.num_filters
        self.num_actions = args.num_actions
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        if agent_mode == 'act':
            self.num_words = args.num_words
            self.word_dim = args.word_dim
        else: # agent_mode == 'arg'
            self.num_words = args.context_len
            self.word_dim = args.word_dim + args.dis_dim
        with tf.variable_scope(agent_mode):
            self.build_dqn()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer, 
                activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        """
        Convolutional layer
        """
        with tf.variable_scope(name):
            # default data_format == 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
            
            w = tf.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)
            out = activation_fn(out)
            return out, w, b


    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        """
        Max pooling layer
        """
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)


    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        """
        Dense layer
        """
        with tf.variable_scope(name):
            w = tf.get_variable('w', [x.get_shape()[1], output_dim], tf.float32, 
                initializer=tf.truncated_normal_initializer(0, 0.1))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def build_dqn(self):
        """
        Build Deep Q-Network
        """
        # prepare repeat-representation for operations
        self.init_tag_emb = np.zeros([self.num_actions + 1, self.tag_dim], dtype=np.float32)
        for i in range(self.num_actions + 1):
            self.init_tag_emb[i] = i
        conv_init = tf.contrib.layers.xavier_initializer_conv2d()
        activation_fn = tf.nn.relu

        # training network
        def build_nn(name, w, s_t):
            """
            Build Text-CNN
            """
            fn = self.num_filters  # number of filters
            fw = s_t.shape[2]      # width of kernels
            with tf.variable_scope(name):
                print('Initializing %s network ...' % name)
                bi_gram,   w['bi_gram_w'],   w['bi_gram_b']   = self.conv2d(s_t, fn, [2, fw], [1, 1], conv_init, name='bi_gram')
                tri_gram,  w['tri_gram_w'],  w['tri_gram_b']  = self.conv2d(s_t, fn, [3, fw], [1, 1], conv_init, name='tri_gram')
                four_gram, w['four_gram_w'], w['four_gram_b'] = self.conv2d(s_t, fn, [4, fw], [1, 1], conv_init, name='four_gram')
                five_gram, w['five_gram_w'], w['five_gram_b'] = self.conv2d(s_t, fn, [5, fw], [1, 1], conv_init, name='five_gram')
                bi_gram_pooled   = self.max_pooling(bi_gram,   kernel_size = [self.num_words - 1, 1], stride = [1, 1], name='bi_gram_pooled')
                tri_gram_pooled  = self.max_pooling(tri_gram,  kernel_size = [self.num_words - 2, 1], stride = [1, 1], name='tri_gram_pooled')
                four_gram_pooled = self.max_pooling(four_gram, kernel_size = [self.num_words - 3, 1], stride = [1, 1], name='four_gram_pooled')
                five_gram_pooled = self.max_pooling(five_gram, kernel_size = [self.num_words - 4, 1], stride = [1, 1], name='five_gram_pooled')

                concat = tf.concat([bi_gram_pooled, tri_gram_pooled, four_gram_pooled, five_gram_pooled], axis=3)
                flatten = tf.reshape(concat, [-1, reduce(lambda x, y: x * y, concat.get_shape().as_list()[1:])])
                dense, w['dense_w'], w['dense_b'] = self.linear(flatten, flatten.shape[-1], activation_fn=tf.nn.relu, name='dense')
                out_layer, w['q_w'], w['q_b'] = self.linear(dense, self.num_actions, name='q')
                for layer in [s_t, bi_gram, bi_gram_pooled, concat, dense, out_layer]:
                    print(layer.shape)
                return out_layer

        #ipdb.set_trace()
        self.w, self.t_w = {}, {}
        self.tag_ind = tf.placeholder(tf.int32, [None, self.num_words], 'tag_ind')
        self.pos_ind = tf.placeholder(tf.int32, [None, self.num_words], 'pos_ind')
        self.word_emb = tf.placeholder(tf.float32, [None, self.num_words, self.word_dim], 'word_emb')
        self.tag_emb_table = tf.placeholder(tf.float32, [self.num_actions + 1, self.tag_dim], 'tag_emb_table')
        self.pos_emb_table = tf.get_variable('pos_emb_table', [self.num_pos, self.pos_dim], tf.float32)
        self.tag_emb = tf.nn.embedding_lookup(self.tag_emb_table, self.tag_ind)
        self.pos_emb = tf.nn.embedding_lookup(self.pos_emb_table, self.pos_ind)
        # s_t is the state representation at t
        if self.use_pos:
            self.s_t = tf.expand_dims(tf.concat([self.word_emb, self.pos_emb, self.tag_emb], axis=2), -1)
        else:
            self.s_t = tf.expand_dims(tf.concat([self.word_emb, self.tag_emb], axis=2), -1)
        
        # q is the currently trained DQN and target_q is a copy of it every target steps
        self.q = build_nn('prediction', self.w, self.s_t)
        self.target_q = build_nn('target', self.t_w, self.s_t)

        with tf.variable_scope('pred_to_target'):
            print('Initializing pred_to_target...')
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder(tf.float32, self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            print('Initializing optimizer...')
            self.target_q_t = tf.placeholder(tf.float32, [None, self.num_actions], name='target_q_t')
            self.delta = self.target_q_t - self.q
            self.loss = tf.reduce_sum(tf.square(self.delta), name='loss')
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()


    def update_target_network(self):
        """
        Update target DQN
        """
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})


    def train(self, minibatch):
        """
        Train DQN with a mini-batch of samples
        """
        #ipdb.set_trace()
        prestates, actions, rewards, poststates, terminals = minibatch
        pre_tag_ind = prestates[:, :, -1]
        pre_pos_ind = prestates[:, :, -2]
        pre_word_emb = prestates[:, :, :-2]
        post_tag_ind = poststates[:, :, -1]
        post_pos_ind = poststates[:, :, -2]
        post_word_emb = poststates[:, :, :-2]
        
        if self.use_pos:
            postq = self.target_q.eval({self.word_emb: post_word_emb, 
                                        self.tag_emb_table: self.init_tag_emb,
                                        self.tag_ind: post_tag_ind,
                                        self.pos_ind: post_pos_ind})
            targets = self.q.eval({ self.word_emb: pre_word_emb,
                                    self.tag_emb_table: self.init_tag_emb,
                                    self.tag_ind: pre_tag_ind,
                                    self.pos_ind: pre_pos_ind})

            maxpostq = np.max(postq, axis=1)
            # update Q-value targets for actions taken  
            for i, action in enumerate(actions):
                if terminals[i]:  
                    targets[i, action] = rewards[i]
                else:  
                    targets[i, action] = rewards[i] + self.gamma * maxpostq[i]

            _, delta, loss = self.sess.run([self.optim, self.delta, self.loss], 
                                           {self.word_emb: pre_word_emb,
                                            self.tag_emb_table: self.init_tag_emb,
                                            self.tag_ind: pre_tag_ind,
                                            self.pos_ind: pre_pos_ind,
                                            self.target_q_t: targets})  
        else:
            postq = self.target_q.eval({self.word_emb: post_word_emb, 
                                        self.tag_emb_table: self.init_tag_emb,
                                        self.tag_ind: post_tag_ind})
            targets = self.q.eval({ self.word_emb: pre_word_emb,
                                    self.tag_emb_table: self.init_tag_emb,
                                    self.tag_ind: pre_tag_ind})

            maxpostq = np.max(postq, axis=1)
            # update Q-value targets for actions taken  
            for i, action in enumerate(actions):
                if terminals[i]:  
                    targets[i, action] = rewards[i]
                else:  
                    targets[i, action] = rewards[i] + self.gamma * maxpostq[i]

            _, delta, loss = self.sess.run([self.optim, self.delta, self.loss], 
                                           {self.word_emb: pre_word_emb,
                                            self.tag_emb_table: self.init_tag_emb,
                                            self.tag_ind: pre_tag_ind,
                                            self.target_q_t: targets})   
        return delta, loss


    def predict(self, current_state):
        """
        Predict Q-values
        """
        #ipdb.set_trace()
        tag_ind = current_state[:, -1][np.newaxis, :]
        pos_ind = current_state[:, -2][np.newaxis, :]
        word_emb = current_state[: ,:-2][np.newaxis, :]
        qvalues = self.q.eval({ self.word_emb: word_emb,
                                self.tag_emb_table: self.init_tag_emb,
                                self.tag_ind: tag_ind,
                                self.pos_ind: pos_ind})
        return qvalues


    def save_weights(self, weight_dir):
        """
        Save weights
        """
        print('Saving weights to %s ...' % weight_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(weight_dir, "%s.pkl" % name))


    def load_weights(self, weight_dir):
        """
        Load weights
        """
        print('Loading weights from %s ...' % weight_dir)
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(weight_dir, "%s.pkl" % name))})

        self.update_target_network()

