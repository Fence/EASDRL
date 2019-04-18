#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
# Project:  Extracting Action Sequences Based on Deep Reinforcement Learning
# Module:   utils
# Author:   Wenfeng Feng 
# Time:     2017.12
################################################################################

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg') # do not require GUI


def str2bool(v):
    """
    Transfrom string command/argument to bool
    """
    return v.lower() in ("yes", "true", "t", "1")


def timeit(f):
    """
    Return time cost of function f
    """
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed


def get_time():
    """
    Get global time as string
    """
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


def save_pkl(obj, path):
    """
    Save pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    """
    Load pickle file
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj


def print_args(args, output_file=''):
    """
    Print all arguments of an argparse instance
    """
    print('\n Arguments:')
    for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
        print('{}: {}'.format(k, v))
    if output_file:
        output_file.write('\n Arguments:\n')
        for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
            output_file.write('{}: {}\n'.format(k, v))
        output_file.write('\n')


def pos_tagging(agent_mode, domain):
    """
    Part-of-speech tagging, using StanfordPOSTagger
    """
    # import ipdb
    from tqdm import tqdm
    from nltk.tag import StanfordPOSTagger
    # NB! you should download the Stanford postagger and set the following two directories
    # e.g. pos_model = 'stanford/postagger/models/english-bidirectional-distsim.tagger'
    # pos_jar = 'stanford/postagger/stanford-postagger.jar'
    pos_model = ''
    pos_jar = ''
    pos_tagger = StanfordPOSTagger(pos_model, pos_jar)
    
    if agent_mode == 'act':
        indata = pickle.load(open('data/%s_labeled_text_data.pkl' % domain, 'r'))
    else:
        _, __, indata = pickle.load(open('data/refined_%s_data.pkl' % domain, 'r'))

    # ipdb.set_trace()
    # pos_data = []
    pos_data = load_pkl('data/%s_%s_pos.pkl' % (domain, agent_mode))
    try:
        for i in range(118, len(indata)):
            pos_text = []
            for j in range(len(indata[i])):
                print('Text %d/%d Sent %d/%d' % (i+1, len(indata), j+1, len(indata[i])))
                if len(indata[i][j]) == 0:
                    continue
                last_sent = [w.lower() for w in indata[i][j]['last_sent']]
                this_sent = [w.lower() for w in indata[i][j]['this_sent']]
                pos_sent = [pos_tagger.tag(last_sent), pos_tagger.tag(this_sent)]
                pos_text.append(pos_sent)
            pos_data.append(pos_text)
    except:
        print('Error! i=%d, j=%d' % (i, j))
    save_pkl(pos_data, 'data/%s_%s_pos.pkl' % (domain, agent_mode))


def transfer(infile, outfile):
    """
    Transfer labeled data for action name extractor. 
    The infile is original labeled data which can be use directly by action argument extractor
    e.g. infile = 'data/online_test/online_labeled_text.pkl' # is online-labeled data
         outfile = 'data/online_test/online_labeled_text_data.pkl'
    PS: It's an a long
    """
    indata = load_pkl(infile)[-1]
    data = []
    log = {'wrong_last_sent': 0, 'act_reference_1': 0, 'related_act_reference_1': 0,
            'obj_reference_1': 0, 'non-obj_reference_1': 0}
    for i in range(len(indata)):
        words = []  # all words of a text
        sents = []  # all sentences of a text
        word2sent = {}  # transfer from a word index to its sentence index
        text_acts = []  # all actions of a text
        sent_acts = []  # actions of each sentence
        reference_related_acts = False
        for j in range(len(indata[i])):
            # labeling error: empty sentence
            if len(indata[i][j]) == 0:  
                print('%s, len(indata[%d][%d]) == 0'%(self.domain, i, j))
                continue
            last_sent = indata[i][j]['last_sent']
            this_sent = indata[i][j]['this_sent']
            acts = indata[i][j]['acts']
            
            # labeling error: mis-matched sentences
            if j > 0 and len(last_sent) != len(indata[i][j-1]['this_sent']):
                b1 = len(last_sent)
                b2 = len(indata[i][j-1]['this_sent'])
                for k in range(len(acts)):
                    ai = acts[k]['act_idx']
                    new_act_type = acts[k]['act_type']
                    new_act_idx = ai - b1 + b2
                    new_obj_idxs = [[],[]]
                    for l in range(2):
                        for oi in acts[k]['obj_idxs'][l]:
                            if oi == -1:
                                new_obj_idxs[l].append(oi)
                            else:
                                new_obj_idxs[l].append(oi - b1 + b2)
                        assert len(new_obj_idxs[l]) == len(acts[k]['obj_idxs'][l])
                    new_related_acts = []
                    acts[k] = {'act_idx': new_act_idx, 'obj_idxs': new_obj_idxs,
                            'act_type': new_act_type, 'related_acts': new_related_acts}
                last_sent = indata[i][j-1]['this_sent']
                log['wrong_last_sent'] += 1

            sent = last_sent + this_sent
            last_sent_bias = len(last_sent)
            # pronoun resolution, find the source noun of a pronoun
            reference_obj_flag = False  
            tmp_acts = []
            for k in range(len(acts)):
                act_idx = acts[k]['act_idx']
                obj_idxs = acts[k]['obj_idxs']
                tmp_act_idx = act_idx - last_sent_bias
                if tmp_act_idx < 0:
                    log['act_reference_1'] += 1
                
                tmp_obj_idxs = [[],[]]
                for l in range(2):
                    for oi in obj_idxs[l]:
                        if oi == -1:
                            tmp_obj_idxs[l].append(oi)
                        else:
                            tmp_obj_idxs[l].append(oi - last_sent_bias)
                            if oi - last_sent_bias < 0:
                                reference_obj_flag = True
                    assert len(tmp_obj_idxs[l]) == len(obj_idxs[l])
                tmp_act_type = acts[k]['act_type']
                tmp_related_acts = []
                if len(acts[k]['related_acts']) > 0:
                    for idx in acts[k]['related_acts']:
                        tmp_related_acts.append(idx - last_sent_bias)
                        if idx - last_sent_bias < 0:
                            reference_related_acts = True
                            log['related_act_reference_1'] += 1
                    assert len(tmp_related_acts) == len(acts[k]['related_acts'])
                tmp_acts.append({'act_idx': tmp_act_idx, 'obj_idxs': tmp_obj_idxs,
                            'act_type': tmp_act_type, 'related_acts': tmp_related_acts})
            # assert len(tmp_acts) == len(acts)
            # labeling error: wrong word index in the first sentence 
            if j == 0:
                if reference_obj_flag:
                    log['obj_reference_1'] += 1
                    for ii in range(len(words), len(words)+len(last_sent)):
                        word2sent[ii] = len(sents)
                    words.extend(last_sent)
                    sents.append(last_sent)
                    sent_acts.append({})
                else:
                    if len(last_sent) > 0:
                        log['non-obj_reference_1'] += 1
                        last_sent = []
                        last_sent_bias = len(last_sent)
                        sent = last_sent + this_sent
                        acts = tmp_acts

            
            for ii in range(len(words), len(words)+len(this_sent)):
                word2sent[ii] = len(sents)
            all_word_bias = len(words)
            words.extend(this_sent)
            sents.append(this_sent)
            sent_acts.append(acts)
            all_acts_of_cur_sent = update_acts(words, sent, last_sent_bias, all_word_bias, tmp_acts)
            text_acts.extend(all_acts_of_cur_sent)

        # assert len(word2sent) == len(words)
        # assert len(sents) == len(sent_acts)
        data.append({'words': words, 'acts': text_acts, 'sent_acts': sent_acts,
                    'sents': sents, 'word2sent': word2sent})
    upper_bound, lower_bound = compute_context_len(data)
    print('\nupper_bound: {}\tlower_bound: {}\nlog history: {}\n'.format(upper_bound, lower_bound, log))
    save_pkl(data, outfile)



def update_acts(words, sent, last_sent_bias, all_word_bias, tmp_acts):
    """
    Add all actions of the current sentence to text_acts
    """
    # all indices of the words in the current sentences need to add a last_sent_bias
    all_acts_of_cur_sent = []
    for k in range(len(tmp_acts)):
        act_idx = tmp_acts[k]['act_idx']
        obj_idxs = tmp_acts[k]['obj_idxs']
        text_act_idx = act_idx + all_word_bias
        # labeling error: mis-matched word index
        if sent[act_idx + last_sent_bias] != words[act_idx + all_word_bias]:
            print(sent[act_idx + last_sent_bias], words[act_idx + all_word_bias])
        text_obj_idxs = [[],[]]
        for l in range(2):
            for oi in obj_idxs[l]:
                if oi == -1:
                    text_obj_idxs[l].append(-1)
                else:
                    text_obj_idxs[l].append(oi + all_word_bias)
                    if sent[oi + last_sent_bias] != words[oi + all_word_bias]:
                        ipdb.set_trace()
                        print(sent[oi + last_sent_bias], words[oi + all_word_bias])
            # assert len(text_obj_idxs[l]) == len(obj_idxs[l])
        text_act_type = tmp_acts[k]['act_type']
        text_related_acts = []
        if len(tmp_acts[k]['related_acts']) > 0:
            for idx in tmp_acts[k]['related_acts']:
                text_related_acts.append(idx + all_word_bias)
            # assert len(text_related_acts) == len(tmp_acts[k]['related_acts'])
        acts = {'act_idx': text_act_idx, 'obj_idxs': text_obj_idxs,
                'act_type': text_act_type, 'related_acts': text_related_acts}
        all_acts_of_cur_sent.append(acts)
    return all_acts_of_cur_sent



def compute_context_len(data):
    """
    Compute the length of context for action argument extractor
    the upper_bound/lower_bound indicate how far/near between the action name and its arguments
    the difference between them is used to control the context_len
    e.g. context_len = 2 * upper_bound
    """
    upper_bound = 0
    lower_bound = 0
    for d in data:
        for n in range(len(d['acts'])):
            act = d['acts'][n]['act_idx']
            objs = d['acts'][n]['obj_idxs']
            for l in range(2):
                for obj in objs[l]:
                    if obj == -1:
                        continue
                    if obj - act < lower_bound:
                        lower_bound = obj - act
                    if obj - act > upper_bound:
                        upper_bound = obj - act
    return upper_bound, lower_bound



def plot_results(results, domain, filename):
    """
    Plot training results, called by main
    """
    print('\nSave results to %s' % filename)
    fontsize = 20
    if isinstance(results, list):
        plt.figure()
        plt.plot(range(len(results)), results, label='loss')
        plt.title('domain: %s' % domain)
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 
        plt.savefig(filename, format='pdf')
        print('Success\n')

    else:
        plt.figure(figsize=(16, 20)) # , dpi=300
        plt.subplot(311)
        x = range(len(results['rec']))
        plt.plot(x, results['rec'], label='rec')
        plt.plot(x, results['pre'], label='pre')
        plt.plot(x, results['f1'], label='f1')
        plt.title('domain: %s' % domain, fontsize=fontsize)
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 

        plt.subplot(312)
        plt.plot(range(len(results['rw'])), results['rw'], label='reward')
        plt.xlabel('episodes', fontsize=fontsize)
        plt.legend(loc='best', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)  
        plt.yticks(fontsize=fontsize) 

        if 'loss' in results:
            plt.subplot(313)
            plt.plot(range(len(results['loss'])), results['loss'], label='loss')
            plt.xlabel('episodes', fontsize=fontsize)
            plt.legend(loc='best', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)  
            plt.yticks(fontsize=fontsize) 
        
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.savefig(filename, format='pdf')
        print('Success\n')



def ten_fold_split_ind(num_data, fname, k, random=True):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        tmp_inds = np.arange(num_data)
        np.random.shuffle(tmp_inds)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_inds[i*n: ])
            else:
                indices.append(tmp_inds[i*n: (i+1)*n])
    else:
        for i in range(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices



def index2data(indices, data):
    """
    Obtain k-fold data according to given indices
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
        num_data = len(data[keys[0]])
        for i in range(len(indices)):
            valid_data = {}
            train_data = {}
            for k in keys:
                valid_data[k] = []
                train_data[k] = []
            for ind in range(num_data):
                for k in keys:
                    if ind in indices[i]:
                        valid_data[k].append(data[k][ind])
                    else:
                        train_data[k].append(data[k][ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)
    else:
        num_data = len(data)
        for i in range(len(indices)):
            valid_data = []
            train_data = []
            for ind in range(num_data):
                if ind in indices[i]:
                    valid_data.append(data[ind])
                else:
                    train_data.append(data[ind])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds



if __name__ == '__main__':
    import ipdb
    infile = 'data/online_test/online_labeled_text.pkl' # is online-labeled data
    outfile = 'online_labeled_text_data.pkl'
    transfer(infile, outfile)
