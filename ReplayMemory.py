#coding:utf-8
import ipdb
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, args, agent_mode):
        print('Initializing ReplayMemory...')
        self.size = args.replay_size
        if agent_mode == 'act':
            self.word_dim = args.word_dim
            self.num_words = args.num_words
        elif agent_mode == 'arg':
            self.word_dim = args.word_dim + args.dis_dim
            self.num_words = args.context_len

        self.actions = np.zeros(self.size, dtype = np.uint8)
        self.rewards = np.zeros(self.size, dtype = np.float16)
        self.states = np.zeros([self.size, self.num_words, self.word_dim + 1], dtype=np.float16)
        self.terminals = np.zeros(self.size, dtype = np.bool)
        self.priority = args.priority
        self.positive_rate = args.positive_rate
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        if args.load_replay:
            self.load(args.save_replay_name)


    def reset(self):
        print('Reset the replay memory')
        self.actions *= 0
        self.rewards *= 0.0
        self.states *= 0.0
        self.terminals *= False
        self.count = 0
        self.current = 0

        
    def add(self, action, reward, state, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size


    def getMinibatch(self):
        """
        Memory must include poststate, prestate and history
        Sample random indexes or with priority
        """
        prestates = np.zeros([self.batch_size, self.num_words, self.word_dim + 1])
        poststates = np.zeros([self.batch_size, self.num_words, self.word_dim + 1])
        if self.priority:
            pos_amount =  int(self.positive_rate*self.batch_size) 

        indexes = []
        count_pos = 0
        count_neg = 0
        count_circle = 0 
        max_circles = 10*self.batch_size # max times for choosing positive samples or nagative samples
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over) 
                index = np.random.randint(1, self.count - 1)
                # NB! poststate (last state) can be terminal state!
                if self.terminals[index - 1]:
                    continue
                # use prioritized replay trick
                if self.priority:
                    if count_circle < max_circles:
                        # if num_pos is already enough but current ind is also pos sample, continue
                        if (count_pos >= pos_amount) and (self.rewards[index] > 0):
                            count_circle += 1
                            continue
                        # elif num_nag is already enough but current ind is also nag sample, continue
                        elif (count_neg >= self.batch_size - pos_amount) and (self.rewards[index] < 0): 
                            count_circle += 1
                            continue
                    if self.rewards[index] > 0:
                        count_pos += 1
                    else:
                        count_neg += 1
                break
            
            prestates[len(indexes)] = self.states[index - 1]
            indexes.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indexes]  
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        poststates = self.states[indexes]
        return prestates, actions, rewards, poststates, terminals


    def save(self, fname, size):
        if size > self.size:
            size = self.size
        databag = {}
        databag['actions'] = self.actions[: size]
        databag['rewards'] = self.rewards[: size]
        databag['states'] = self.states[: size]
        databag['terminals'] = self.terminals[: size]
        with open(fname, 'wb') as f:
            print('Try to save replay memory ...')
            pickle.dump(databag, f)
            print('Replay memory is successfully saved as %s' % fname)


    def load(self, fname):
        if not os.path.exists(fname):
            print("%s doesn't exist!" % fname)
            return
        with open(fname, 'rb') as f:
            print('Loading replay memory from %s ...' % fname)
            databag = pickle.load(f)
            size = len(databag['states'])
            self.states[: size] = databag['states']
            self.actions[: size] = databag['actions']
            self.rewards[: size] = databag['rewards']
            self.terminals[: size] = databag['terminals']
            self.count = size
            self.current = size