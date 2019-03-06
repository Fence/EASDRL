import os
import ipdb
import numpy as np
import tensorflow as tf
from functools import reduce
from utils import save_pkl, load_pkl
from tensorflow.contrib.layers.python.layers import initializers

class DeepQLearner:
    def __init__(self, args, sess, agent_mode):
        print('Initializing the DQN...')
        self.sess = sess
        self.args = args
        self.tag_dim = args.tag_dim
        self.dense_dim = args.dense_dim
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


    def conv2d(self, x, output_dim, kernel_size, stride, initializer, activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        with tf.variable_scope(name):
            # data_format = 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
            
            w = tf.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)


    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [x.get_shape()[1], output_dim], tf.float32, 
                initializer=tf.truncated_normal_initializer(0, 0.1))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def build_dqn(self):
        # prepare repeat-representation for operations
        self.init_tag_emb = np.zeros([self.num_actions + 1, self.tag_dim], dtype=np.float32)
        for i in range(self.num_actions + 1):
            self.init_tag_emb[i] = i
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        activation_fn = tf.nn.relu

        # training network
        def build_nn(name, weight, s_t):
            fn = self.args.num_filters  #filter num
            fw = s_t.shape[2]  # filter width
            with tf.variable_scope(name):
                print('Initializing %s network ...' % name)
                self.l1, weight['l1_w'], weight['l1_b'] = self.conv2d(s_t,
                    fn, [2, fw], [1, 1], initializer, activation_fn, name='l1')
                self.l3, weight['l3_w'], weight['l3_b'] = self.conv2d(s_t,
                    fn, [3, fw], [1, 1], initializer, activation_fn, name='l3')
                self.l5, weight['l5_w'], weight['l5_b'] = self.conv2d(s_t,
                    fn, [4, fw], [1, 1], initializer, activation_fn, name='l5')
                self.l7, weight['l7_w'], weight['l7_b'] = self.conv2d(s_t,
                    fn, [5, fw], [1, 1], initializer, activation_fn, name='l7')
                self.l2 = self.max_pooling(
                    self.l1, kernel_size = [self.num_words-1, 1], stride = [1, 1], name='l2')
                self.l4 = self.max_pooling(
                    self.l3, kernel_size = [self.num_words-2, 1], stride = [1, 1], name='l4')
                self.l6 = self.max_pooling(
                    self.l5, kernel_size = [self.num_words-3, 1], stride = [1, 1], name='l6')
                self.l8 = self.max_pooling(
                    self.l7, kernel_size = [self.num_words-4, 1], stride = [1, 1], name='l8')

                self.l9 = tf.concat([self.l2, self.l4, self.l6, self.l8], axis=3)
                l9_shape = self.l9.get_shape().as_list()
                self.l9_flat = tf.reshape(self.l9, [-1, reduce(lambda x, y: x * y, l9_shape[1:])])
                self.l10, weight['l10_w'], weight['l10_b'] = self.linear(self.l9_flat, self.l9_flat.shape[-1], activation_fn=activation_fn, name='l10')
                out_layer, weight['q_w'], weight['q_b'] = self.linear(self.l10, self.num_actions, name='q')
                for layer in [s_t, self.l1, self.l2, self.l9, self.l10, out_layer]:
                    print(layer.shape)

                return out_layer

        #self.cnn_format == 'NCHW'
        #ipdb.set_trace()
        self.w, self.t_w = {}, {}
        self.word_emb = tf.placeholder(tf.float32, [None, self.num_words, self.word_dim], 'word_emb')
        self.tag_emb = tf.placeholder(tf.float32, [self.num_actions + 1, self.tag_dim], 'tag_emb')
        self.tag_ind = tf.placeholder(tf.int32, [None, self.num_words], 'tag_ind')
        self.tags = tf.nn.embedding_lookup(self.tag_emb, self.tag_ind)
        self.s_t = tf.expand_dims(tf.concat([self.word_emb, self.tags], axis=2), -1)
        
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
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})


    def train(self, minibatch):
        # expand components of minibatch
        #ipdb.set_trace()
        prestates, actions, rewards, poststates, terminals = minibatch
        pre_tag_ind = prestates[:, :, -1]
        post_tag_ind = poststates[:, :, -1]
        pre_word_emb = prestates[:, :, :-1]
        post_word_emb = poststates[:, :, :-1]
        
        postq = self.target_q.eval({self.word_emb: post_word_emb, 
                                    self.tag_emb: self.init_tag_emb,
                                    self.tag_ind: post_tag_ind})
        maxpostq = np.max(postq, axis=1)
        targets = self.q.eval({ self.word_emb: pre_word_emb,
                                self.tag_emb: self.init_tag_emb,
                                self.tag_ind: pre_tag_ind})

        # update Q-value targets for actions taken  
        for i, action in enumerate(actions):
            if terminals[i]:  
                targets[i, action] = rewards[i]
            else:  
                targets[i, action] = rewards[i] + self.gamma * maxpostq[i]

        _, delta, loss = self.sess.run([self.optim, self.delta, self.loss], 
                                       {self.word_emb: pre_word_emb,
                                        self.tag_emb: self.init_tag_emb,
                                        self.tag_ind: pre_tag_ind,
                                        self.target_q_t: targets})     
        return delta, loss


    def predict(self, current_state):
        #ipdb.set_trace()
        tag_ind = current_state[:, -1][np.newaxis, :]
        word_emb = current_state[: ,:-1][np.newaxis, :]
        qvalues = self.q.eval({ self.word_emb: word_emb,
                                self.tag_emb: self.init_tag_emb,
                                self.tag_ind: tag_ind})

        return qvalues


    def save_weights(self, weight_dir):
        print('Saving weights to %s ...' % weight_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(weight_dir, "%s.pkl" % name))


    def load_weights(self, weight_dir, cpu_mode=False):
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

