import ipdb
import keras
import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import Model
from keras.layers.normalization import BatchNormalization


class DeepQLearner:
    def __init__(self, args, agent_name):
        print('Initializing the DQN...')
        self.word_dim = args.word_dim
        self.dropout = args.dropout
        self.optimizer = args.optimizer
        self.dense_dim = args.dense_dim
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate
        self.num_actions = args.num_actions
        self.num_filters = args.num_filters
        self.agent_mode = args.agent_mode
        if agent_name == 'act':
            self.num_words = args.num_words
            self.emb_dim = args.word_dim + args.tag_dim
        elif agent_name == 'arg':
            self.num_words = args.context_len
            self.emb_dim = args.word_dim + args.dis_dim + args.tag_dim
        self.build_dqn()


    def build_dqn(self):
        #ipdb.set_trace()
        fw = self.emb_dim  #filter width
        fn = self.num_filters  #filter num
        inputs = Input(shape=(self.num_words, self.emb_dim, 1))

        bi_gram = Conv2D(fn, (2, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #bi_gram = BatchNormalization()(bi_gram)
        bi_gram = Activation(activation='relu')(bi_gram)
        bi_gram = MaxPooling2D((self.num_words - 1, 1), strides=(1, 1), padding='valid')(bi_gram)

        tri_gram = Conv2D(fn, (3, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #tri_gram = BatchNormalization()(tri_gram)
        tri_gram = Activation(activation='relu')(tri_gram)
        tri_gram = MaxPooling2D((self.num_words - 2, 1), strides=(1, 1), padding='valid')(tri_gram)

        four_gram = Conv2D(fn, (4, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #four_gram = BatchNormalization()(four_gram)
        four_gram = Activation(activation='relu')(four_gram)
        four_gram = MaxPooling2D((self.num_words - 3, 1), strides=(1, 1), padding='valid')(four_gram)

        five_gram = Conv2D(fn, (5, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #five_gram = BatchNormalization()(five_gram)
        five_gram = Activation(activation='relu')(five_gram)
        five_gram = MaxPooling2D((self.num_words - 4, 1), strides=(1, 1), padding='valid')(five_gram)

        # concates.shape = [None, 1, 8, 32]
        concate = kl.concatenate([bi_gram, tri_gram, four_gram, five_gram], axis=2)
        flat = Flatten()(concate)

        full_con = Dense(self.dense_dim, activation='relu', kernel_initializer='truncated_normal')(flat)
        out = Dense(self.num_actions, kernel_initializer='truncated_normal')(full_con)

        self.model = Model(inputs, out)
        self.target_model = Model(inputs, out)
        self.compile_model()


    def compile_model(self):
        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=0.9, nesterov=True)
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=self.learning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
        else:
            opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)

        self.model.compile(optimizer=opt, loss='mse')
        self.target_model.compile(optimizer=opt, loss='mse')
        print(self.model.summary())


    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self, minibatch):
        # expand components of minibatch
        # channel_first
        prestates, actions, rewards, poststates, terminals = minibatch
        
        post_input = poststates[:, :, :, np.newaxis] #np.reshape(poststates, [-1, self.num_words, self.emb_dim, 1])
        postq = self.target_model.predict_on_batch(post_input)

        pre_input = prestates[:, :, :, np.newaxis] #np.reshape(prestates, [-1, self.num_words, self.emb_dim, 1])
        targets = self.model.predict_on_batch(pre_input)
        # calculate max Q-value for each poststate  
        maxpostq = np.max(postq, axis=1)

        # update Q-value targets for actions taken  
        for i, action in enumerate(actions):
            if terminals[i]:  
                targets[i, action] = float(rewards[i])
            else:  
                targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        self.model.train_on_batch(pre_input, targets)


    def predict(self, current_state):
        state_input = current_state[np.newaxis, :, :, np.newaxis] 
        qvalues = self.model.predict_on_batch(state_input)
        return qvalues


    def save_weights(self, weight_dir):
        self.model.save_weights(weight_dir)
        print('Saved weights to %s ...' % weight_dir)


    def load_weights(self, weight_dir):
        self.model.load_weights(weight_dir)
        print('Loaded weights from %s ...' % weight_dir)
