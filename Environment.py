# coding:utf-8
import re
import ipdb
import time
import pickle
import numpy as np
from utils import ten_fold_split_ind, index2data, save_pkl, load_pkl


class Environment:
    def __init__(self, args, agent_mode):
        print('Initializing the Environment...')  
        self.domain = args.domain
        self.dis_dim = args.dis_dim
        self.tag_dim = args.tag_dim
        self.word_dim = args.word_dim
        self.num_words = args.num_words
        self.action_rate = args.action_rate
        self.use_act_rate = args.use_act_rate
        self.use_act_att = args.use_act_att
        self.reward_base = args.reward_base
        self.ra = args.reward_assign

        self.word2vec = args.word2vec
        self.fold_id = args.fold_id
        self.k_fold = args.k_fold
        self.k_fold_indices = args.k_fold_indices

        self.terminal_flag = False
        self.train_epoch_end_flag = False
        self.valid_epoch_end_flag = False
        self.max_data_char_len = 0
        self.max_data_sent_len = 0
        self.agent_mode = agent_mode # args.agent_mode
        self.context_len = args.context_len
        if not args.gui_mode:
            if self.agent_mode == 'arg':
                self.read_arg_sents()
            else:
                self.read_act_texts()
            args.train_steps = self.train_steps
            args.valid_steps = self.valid_steps
        

    def init_predict_act_text(self, raw_text):
        #ipdb.set_trace()
        #raw_text = re.sub(r'\n|\r|\(|\)|,|;', ' ', raw_text)
        #raw_text = re.split(r'\. |\? |\! ', raw_text)
        text = {'tokens': [], 'sents': [], 'word2sent': {}}
        for s in raw_text:
            words = s.split()
            if len(words) > 0:
                for i in range(len(words)):
                    text['word2sent'][i + len(text['tokens'])] = [len(text['sents']), i]
                text['tokens'].extend(words)
                text['sents'].append(words)

        sent_vec = np.zeros([self.num_words, self.word_dim + self.tag_dim])
        for i, w in enumerate(text['tokens']):
            if i >= self.num_words:
                break
            if w in self.word2vec.vocab:
                sent_vec[i][: self.word_dim] = self.word2vec[w]

        self.state = sent_vec
        self.terminal_flag = False
        self.current_text = text


    def init_predict_arg_text(self, act_idx, text):
        self.terminal_flag = False
        sents = text['sents']
        word2sent = text['word2sent']
        sent_idx = word2sent[act_idx][0]
        word_ids = []
        this_sent = sents[sent_idx]
        if sent_idx > 0: # use the former sentence and current one
            last_sent = sents[sent_idx - 1]
            for k, v in word2sent.iteritems():
                if v[0] == sent_idx or v[0] == sent_idx - 1:
                    word_ids.append(k)
        else:
            last_sent = []
            for k, v in word2sent.iteritems():
                if v[0] == sent_idx:
                    word_ids.append(k)
        words = last_sent + this_sent + ['UNK']
        end_idx = max(word_ids) # the last index of words of these two sents
        start_idx = min(word_ids)
        sent_len = len(words)

        position = np.zeros(sent_len, dtype=np.int32)
        position.fill(act_idx - start_idx)
        distance = np.abs(np.arange(sent_len) - position)
        sent_vec = np.zeros([self.context_len, self.word_dim + self.dis_dim + self.tag_dim])
        for i, w in enumerate(words):
            if i >= self.context_len:
                break
            if w in self.word2vec.vocab:
                sent_vec[i][: self.word_dim] = self.word2vec[w]
            sent_vec[i][self.word_dim: self.word_dim + self.dis_dim] = distance[i]
        self.state = sent_vec
        self.current_text = {'tokens': words, 'word2sent': word2sent, 'distance': distance}
        return last_sent, this_sent


    def act_online(self, action, word_ind):
        #self.state[word_ind, -1] = action + 1
        self.state[word_ind, -self.tag_dim:] = action + 1
        if word_ind + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True


    def read_act_texts(self):
        #ipdb.set_trace()
        if self.domain == 'all':
            indata = []
            for domain in ['win2k', 'cooking', 'wikihow']:
                tmp_data = load_pkl('data/%s_labeled_text_data.pkl' % domain)
                indata.extend(tmp_data)
        else:
            indata = load_pkl('data/%s_labeled_text_data.pkl' % self.domain)
        
        act_texts = []
        for i in range(len(indata)):
            if len(indata[i]['words']) == 0:
                continue
            act_text = {}
            act_text['tokens'] = indata[i]['words']
            act_text['sents'] = indata[i]['sents']
            act_text['acts'] = indata[i]['acts']
            act_text['sent_acts'] = indata[i]['sent_acts']
            act_text['word2sent'] = indata[i]['word2sent']
            act_text['tags'] = np.ones(len(indata[i]['words']), dtype=np.int32)
            act_text['act2related'] = {}
            for acts in indata[i]['acts']:
                act_text['act2related'][acts['act_idx']] = acts['related_acts']
                act_text['tags'][acts['act_idx']] = acts['act_type'] + 1 # 2, 3, 4
            self.create_matrix(act_text)
            act_texts.append(act_text)

        act_indices = ten_fold_split_ind(len(act_texts), self.k_fold_indices, self.k_fold)
        act_folds = index2data(act_indices, act_texts)
        self.train_data = act_folds['train'][self.fold_id]
        self.valid_data = act_folds['valid'][self.fold_id]
        self.train_steps = len(self.train_data) * self.num_words
        self.valid_steps = len(self.valid_data) * self.num_words

        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def read_arg_sents(self):
        if self.domain == 'all':
            indata = []
            for domain in ['win2k', 'cooking', 'wikihow']:
                tmp_data = load_pkl('data/refined_%s_data.pkl' % domain)[-1]
                indata.extend(tmp_data)
        else:
            indata = load_pkl('data/refined_%s_data.pkl' % self.domain)[-1]
        arg_sents = []
        #ipdb.set_trace()
        for i in range(len(indata)):
            for j in range(len(indata[i])):
                if len(indata[i][j]) == 0:
                    continue
                # -1 obj_ind refer to UNK
                words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK'] 
                sent_len = len(words)
                act_inds = [a['act_idx'] for a in indata[i][j]['acts'] if a['act_idx'] < self.num_words]
                for k in range(len(indata[i][j]['acts'])):
                    act_ind = indata[i][j]['acts'][k]['act_idx']
                    obj_inds = indata[i][j]['acts'][k]['obj_idxs']
                    arg_sent = {}
                    arg_tags = np.ones(sent_len, dtype=np.int32)
                    if len(obj_inds[1]) == 0:
                        arg_tags[obj_inds[0]] = 2 # essential objects
                    else:
                        arg_tags[obj_inds[0]] = 4 # exclusive objects
                        arg_tags[obj_inds[1]] = 4 # exclusive objects
                    position = np.zeros(sent_len, dtype=np.int32)
                    position.fill(act_ind)
                    distance = np.abs(np.arange(sent_len) - position)
                    
                    arg_sent['tokens'] = words
                    arg_sent['tags'] = arg_tags
                    arg_sent['act_ind'] = act_ind
                    arg_sent['distance'] = distance
                    arg_sent['act_inds'] = act_inds
                    arg_sent['obj_inds'] = obj_inds
                    self.create_matrix(arg_sent)
                    arg_sents.append(arg_sent)

        arg_indices = ten_fold_split_ind(len(arg_sents), self.k_fold_indices, self.k_fold)
        arg_folds = index2data(arg_indices, arg_sents)
        self.train_data = arg_folds['train'][self.fold_id]
        self.valid_data = arg_folds['valid'][self.fold_id]
        self.train_steps = len(self.train_data) * self.num_words
        self.valid_steps = len(self.valid_data) * self.num_words

        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))



    def create_matrix(self, sentence):
        #ipdb.set_trace()
        sent_vec = []
        for w in sentence['tokens']:
            if len(w) > self.max_data_char_len:
                self.max_data_char_len = len(w)
            if w in self.word2vec.vocab:
                sent_vec.append(self.word2vec[w])
            else:
                sent_vec.append(np.zeros(self.word_dim))

        sent_vec = np.array(sent_vec)
        pad_len = self.num_words - len(sent_vec)
        if len(sent_vec) > self.max_data_sent_len:
            self.max_data_sent_len = len(sent_vec)
        
        if self.agent_mode == 'act':
            if pad_len > 0:
                sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
                sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
            else:
                sent_vec = sent_vec[: self.num_words]
                sentence['tokens'] = sentence['tokens'][: self.num_words]
                sentence['tags'] = np.array(sentence['tags'])[: self.num_words]

        else: # self.agent_mode == 'arg':
            #ipdb.set_trace()
            distance = np.zeros([self.num_words, self.dis_dim])
            act_vec = sent_vec[sentence['act_ind']]  # word vector of the input action 
            attention = np.sum(sent_vec * act_vec, axis=1)  # attention between the input action and its context 
            attention = np.exp(attention)
            attention /= sum(attention)
            if pad_len > 0:
                sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
                sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
                attention = np.concatenate((attention, np.zeros(pad_len)))
                for d in range(len(sentence['distance'])):
                    distance[d] = sentence['distance'][d]
            else:
                sent_vec = sent_vec[: self.num_words]
                sentence['tokens'] = sentence['tokens'][: self.num_words]
                sentence['tags'] = np.array(sentence['tags'])[: self.num_words]
                attention = attention[: self.num_words]
                for d in range(self.num_words):
                    distance[d] = sentence['distance'][d]
            #ipdb.set_trace()
            if self.use_act_att: # apply attention to word embedding
                sent_vec = attention.reshape(-1, 1) * sent_vec
            sent_vec = np.concatenate((sent_vec, distance), axis=1)
        
        sentence['sent_vec'] = sent_vec
        sentence['tags'].shape = (self.num_words, 1)


    def restart(self, train_flag, init=False):
        #ipdb.set_trace()
        if train_flag:
            if init:
                self.train_text_ind = -1
                self.train_epoch_end_flag = False
            self.train_text_ind += 1
            if self.train_text_ind >= len(self.train_data):
                self.train_epoch_end_flag = True
                print('\n\n-----train_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.train_data[self.train_text_ind%self.num_train]
            print('\ntrain_text_ind: %d of %d' % (self.train_text_ind, len(self.train_data)))
        else:
            if init:
                self.valid_text_ind = -1
                self.valid_epoch_end_flag = False
            self.valid_text_ind += 1
            if self.valid_text_ind >= len(self.valid_data):
                self.valid_epoch_end_flag = True
                print('\n\n-----valid_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.valid_data[self.valid_text_ind]
            print('\nvalid_text_ind: %d of %d' % (self.valid_text_ind, len(self.valid_data)))
        
        self.text_vec = np.concatenate((self.current_text['sent_vec'], self.current_text['tags']), axis=1)
        self.state = self.text_vec.copy() 
        self.state[:, -1] = 0
        self.terminal_flag = False

        
    def act(self, action, word_ind):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        self.state[word_ind, -1] = action + 1
        #t_a_count = 0  #amount of tagged actions 
        t_a_count = sum(self.state[: word_ind + 1, -1]) - (word_ind + 1)
        t_a_rate = float(t_a_count) / self.num_words

        label = self.text_vec[word_ind,-1]
        self.real_action_flag = False
        if self.agent_mode == 'arg':
            # text_vec is labelled data
            if label >= 2:
                self.real_action_flag = True
            if label == 2:
                if action == 1:
                    reward = self.ra[1] * self.reward_base
                else:
                    reward = -self.ra[1] * self.reward_base
            elif label == 4:
                right_flag = True
                if word_ind in self.current_text['obj_inds'][0]:
                    exc_objs = self.current_text['obj_inds'][1]
                else:
                    exc_objs = self.current_text['obj_inds'][0]
                for oi in exc_objs: # exclusive objs
                    if self.state[oi, -1] == 2:
                        right_flag = False
                        break
                if action == 1 and right_flag:
                    reward = self.ra[2] * self.reward_base
                elif action == 2 and not right_flag:
                    reward = self.ra[2] * self.reward_base
                elif action == 2 and word_ind != self.current_text['obj_inds'][1][-1]:
                    reward = self.ra[2] * self.reward_base
                else:
                    reward = -self.ra[2] * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = -self.ra[0] * self.reward_base

        else: # self.agent_mode == 'act'
            if label >= 2:
                self.real_action_flag = True 
            if label == 2: #required action
                if action == 1: # extracted as action
                    reward = self.ra[1] * self.reward_base
                else: # filtered out
                    reward = -self.ra[1] * self.reward_base
            elif label == 3: #optional action
                if action == 1:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = 0.0
            elif label == 4: # exclusive action
                #ipdb.set_trace()
                assert word_ind in self.current_text['act2related']
                exclusive_act_inds = self.current_text['act2related'][word_ind]
                exclusive_flag = False
                not_biggest_flag = False
                for ind in exclusive_act_inds:
                    if self.state[ind, -1] == 2: # extracted as action
                        exclusive_flag = True
                    if ind > word_ind:
                        not_biggest_flag = True
                if action == 1 and not exclusive_flag:
                # extract current word and no former exclusive action was extracted
                    reward = self.ra[2] * self.reward_base
                elif action == 0 and exclusive_flag:
                # filtered out current word because one former exclusive action was extracted
                    reward = self.ra[2] * self.reward_base
                elif action == 0 and not_biggest_flag:
                # filtered out current word and at least one exclusive action left 
                    reward = self.ra[2] * self.reward_base
                else:
                    reward = -self.ra[2] * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = -self.ra[0] * self.reward_base
        
        if self.use_act_rate and reward != 0:
            if t_a_rate <= self.action_rate and reward > 0:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
        # all words of current text are tagged, break
        if word_ind + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True
        
        return reward


    def getState(self):
        '''
        Gets current text state
        '''
        return self.state


    def isTerminal(self):
        '''
        Returns if tag_actions is done
        if all the words of a text have been tagged, then terminate
        '''
        return self.terminal_flag
