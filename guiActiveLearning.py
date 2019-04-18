#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
# Project:  Extracting Action Sequences Based on Deep Reinforcement Learning
# Module:   guiActiveLearning
# Author:   Wenfeng Feng 
# Time:     2019.03
################################################################################

import wx, re, os
import time
import ipdb
import pickle
import argparse
import random
import wx.lib.buttons as buttons
import tensorflow as tf
import numpy as np

from copy import deepcopy
from utils import get_time, load_pkl, save_pkl
from main import preset_args, args_init
from EADQN import DeepQLearner
# from KerasEADQN import DeepQLearner
from Environment import Environment
from ReplayMemory import ReplayMemory
from gensim.models import KeyedVectors
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from Clustering import text_classification


def EASDRL_init(args, sess):
    """
    Initial EASDRL model, load weights and prepare RL Agent
    """
    args.gui_mode = True
    args.fold_id = 0
    args.domain = 'wikihow'
    args.replay_size = 100
    args.load_weights = 'weights'
    args.use_pos = False
    args = args_init(args)
    
    # ipdb.set_trace()
    agent = Agent(args, sess)
    if args.load_weights:
        print('Loading weights ...')
        filename = 'weights/online_test/%s_act_%d_fold%d.h5' % (args.domain, args.k_fold, args.fold_id)
        agent.net_act.load_weights(filename)
        filename = 'weights/online_test/%s_arg_%d_fold%d.h5' % (args.domain, args.k_fold, args.fold_id)
        agent.net_arg.load_weights(filename)
    return agent


class Agent(object):
    """
    RL Agent for online Active Learning
    """
    def __init__(self, args, sess):
        self.env_act = Environment(args, 'act')
        # self.net_act = DeepQLearner(args, 'act', 'channels_first')
        self.net_act = DeepQLearner(args, sess, 'act') # for tensorflow
        self.env_arg = Environment(args, 'arg')
        # self.net_arg = DeepQLearner(args, 'arg', 'channels_first')
        self.net_arg = DeepQLearner(args, sess, 'arg') # for tensorflow
        self.num_words = args.num_words
        self.context_len = args.context_len
        self.gamma = args.gamma
        self.uncertainty_mode = 'cml' # or 'cml'


    def load_data(self):
        """
        Load all unlabeled texts.
        PS: the file 'home_and_garden_500_words_with_title.pkl' contains more than 15k 
            unlabeled texts from wikihow home and garden category.
        """
        print('Loading texts from data/home_and_garden_500_words_with_title.pkl ...')
        self.texts = load_pkl('data/home_and_garden_500_words_with_title.pkl')
        self.label2text = text_classification()
        self.history_texts = []
        self.sort_ind = 1 if self.uncertainty_mode == 'diff' else 0
        self.category = 0 # category of the currently chosen text
        self.max_category = len(self.label2text) - 1


    def choose_unlabeled_texts(self, num_texts, dialog=None):
        """
        Apply Active Learning. 
        Choose texts from each class and sort them by cumulative reward.
        """
        chosen_texts = []
        while len(chosen_texts) < num_texts:
            # text_ind = np.random.randint(len(self.texts))
            # text = self.texts[text_ind]
            text_ind = random.sample(self.label2text[self.category], 1)[0]
            if text_ind in self.history_texts: # or len(text['title'].split()) < 2:
                continue

            # print('textID: {:<10}  category: {}'.format(text_ind, self.category))
            # traverse all categories, choose texts from each category
            self.category = self.category + 1 if self.category < self.max_category else 0
            # predict Q-values, compute cumulative reward
            text = self.texts[text_ind]
            sents, word2sent, R_t = self.predict(text['sent'])
            r_t = R_t[: -1] - self.gamma * R_t[1: ] # deduced from R_t = r_t + gamma * R_{t+1}
            cml_rwd = sum(r_t) + self.gamma * R_t[-1]
            delta_r = abs(R_t[0] - cml_rwd) # difference between predicted and real cml_rwd
            text['sents'] = sents
            text['reward'] = (cml_rwd, delta_r) # 
            text['r_t'] = r_t # len(r_t) = len(words) - 1
            text['text_ind'] = text_ind
            text['word2sent'] = word2sent
            chosen_texts.append(text)
            if dialog:
                dialog.Update(len(chosen_texts), 'Progress: %d/%d'%(len(chosen_texts), num_texts))
        # sort the texts by cumulative reward
        sorted_texts = sorted(chosen_texts, key=lambda x:x['reward'][self.sort_ind])
        # for t in sorted_texts:
        #     print(t['text_ind'], t['reward'][self.sort_ind])
        # print('\n')
        return sorted_texts


    def predict(self, text):
        """
        Call EASDRL model to generate output actions for an input text
        e.g. text = ['Cook the rice the day before.', 'Use leftover rice.']
        """
        self.env_act.init_predict_act_text(text)
        # act_seq = []
        sents = []
        for i in range(len(self.env_act.current_text['sents'])):
            if i > 0:
                last_sent = self.env_act.current_text['sents'][i - 1]
                # last_pos = self.env_act.current_text['sent_pos'][i - 1]
            else:
                last_sent = []
                # last_pos = []
            this_sent = self.env_act.current_text['sents'][i]
            # this_pos = self.env_act.current_text['sent_pos'][i]
            sents.append({'last_sent': last_sent, 'this_sent': this_sent, 'acts': []})#, 
                        #'last_pos': last_pos, 'this_pos': this_pos})
        word2sent = self.env_act.current_text['word2sent']
        # ipdb.set_trace()
        R_t = []
        for i in range(self.num_words):
            state_act = self.env_act.getState()
            qvalues_act = self.net_act.predict(state_act)
            R_t.append(max(qvalues_act[0]))
            action_act = np.argmax(qvalues_act[0])
            self.env_act.act_online(action_act, i)
            if action_act == 1:
                last_sent, this_sent = self.env_arg.init_predict_arg_text(i, self.env_act.current_text)
                for j in range(self.context_len):
                    state_arg = self.env_arg.getState()
                    qvalues_arg = self.net_arg.predict(state_arg)
                    action_arg = np.argmax(qvalues_arg[0])
                    self.env_arg.act_online(action_arg, j)
                    if self.env_arg.terminal_flag:
                        break
                # act_name = self.env_act.current_text['tokens'][i]
                # act_arg = [act_name]
                act_idx = i
                obj_idxs = []
                sent_words = self.env_arg.current_text['tokens']
                tmp_num = self.context_len if len(sent_words) >= self.context_len else len(sent_words)
                for j in range(tmp_num):
                    if self.env_arg.state[j, -1] == 2:
                        #act_arg.append(sent_words[j])
                        if j == len(sent_words) - 1:
                            j = -1
                        obj_idxs.append(j)
                if len(obj_idxs) == 0:
                    # act_arg.append(sent_words[-1])
                    obj_idxs.append(-1)
                # ipdb.set_trace()
                si, ai = self.env_act.current_text['word2sent'][i]
                ai += len(sents[si]['last_sent'])
                sents[si]['acts'].append({'act_idx': ai, 'obj_idxs': [obj_idxs, []],
                                            'act_type': 1, 'related_acts': []})
                # act_seq.append(act_arg)
            if self.env_act.terminal_flag:
                break
        # for k, v in act_seq.iteritems():
        #     print(k, v)
        # ipdb.set_trace()
        return sents, word2sent, np.array(R_t)



class EASGUI(wx.Frame):
    """
    A human-robot interaction environment
    """
    def __init__(self, agent):
        wx.Frame.__init__(self, None, -1, "Action Sequence Extraction")
        self.panel = wx.Panel(self)
        # self.panel.SetBackgroundColour(colors[np.random.randint(len(colors))])
        self.font_size = 12
        self.create_io_text_ctrl() 
        self.create_layout()
        self.agent = agent # EASDRL_init()
        self.data = []
        self.cur_text = {'sents': [], 'text_ind': None, 'r_t': None}
        self.chosen_texts = []
        self.act2sent = {}
        self.load_data()


    def load_data(self):
        """
        Load last labeled data
        """
        self.output_file_name = 'data/online_test/online_labeled_text.pkl'
        if os.path.exists(self.output_file_name):
            print('Loading last saved data ...')
            try:
                self.agent.history_texts, self.data = load_pkl(self.output_file_name)
            except:
                self.agent.history_texts, self.data = [], []


    def create_layout(self):
        """
        Create APP's layout
        """
        img = wx.Image('data/online_test/planlab_logo.png', wx.BITMAP_TYPE_ANY)
        w = img.GetWidth()
        h = img.GetHeight()
        logo = wx.StaticBitmap(self.panel, -1, wx.BitmapFromImage(img.Scale(w/5, h/5)))
        self.candidate_list = wx.ListBox(self.panel, -1, pos=wx.DefaultPosition, size=(190, 600), 
                                choices=['1.', '2.', '3.', '4.' ,'5.', '6.'], style=wx.LB_SINGLE)
        self.candidate_list.SetSelection(0)
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.OnTextList, self.candidate_list)
        
        sel_button = self.create_button('Sampling', self.OnSampling)
        prs_button = self.create_button('AutoParse', self.OnParse)
        ext_button = self.create_button('Extract', self.OnExtract)
        del_button = self.create_button('Delete', self.OnDelete)
        rvs_button = self.create_button('Revise', self.OnRevise)
        ins_button = self.create_button('Insert', self.OnInsert)  
        sav_button = self.create_button('Save', self.OnSave)
        qit_button = self.create_button('Exit', self.OnQuit)
        button_box = self.create_boxsizer([prs_button, sav_button, ext_button, del_button, 
                            rvs_button, ins_button], direction=wx.HORIZONTAL)

        process_name = self.show_name('Progress:')
        text_index_name = self.show_name('TextID')
        cur_cml_rwd_name = self.show_name('CumReward')
        num_samples_name = self.show_name('NumSamples')
        self.process = self.show_name('0/0')
        self.text_index = self.show_name(' ')
        self.cur_cml_rwd = self.show_name('0.0')
        pro_box = self.create_boxsizer([process_name, self.process])
        ind_box = self.create_boxsizer([text_index_name, self.text_index])
        rwd_box = self.create_boxsizer([cur_cml_rwd_name, self.cur_cml_rwd])
        num_box = self.create_boxsizer([num_samples_name, self.num_samples], gap=5)
        but_box = self.create_boxsizer([sel_button, qit_button])
        sampling_box = self.create_static_sizer('DataSampling', 
                        [ind_box, pro_box, rwd_box, num_box, but_box],
                        expend=1, direction=wx.VERTICAL)
 
        in_box = self.create_static_sizer('Input Text', self.in_text)
        out_box = self.create_static_sizer('Output Actions', self.out_text)
        candidate_box = self.create_static_sizer('Sampled Texts', self.candidate_list, expend=0)
        upper_box = self.create_boxsizer([candidate_box, in_box, out_box], expend=0, gap=3)

        choice_box = self.create_boxsizer([self.show_name('Act/Arg:'), self.act_arg_choice])
        act_box = self.create_boxsizer([self.show_name('ActSeqNum:'), self.act_idx_in])
        type_box = self.create_boxsizer([self.show_name('ActType/ArgType:'), self.item_type])
        sent_box = self.create_boxsizer([self.show_name('SentId:'), self.sent_idx_in])
        word_box = self.create_boxsizer([self.show_name('ActId/ArgId:'), self.word_idx_in])
        related_si_box = self.create_boxsizer([self.show_name('ExSentId:'), self.related_sent_idx])
        related_it_box = self.create_boxsizer([self.show_name('ExActId/ExArgId:'), self.related_item])
        func_box1 = self.create_boxsizer([choice_box, act_box, type_box], expend=0, direction=wx.VERTICAL)
        func_box2 = self.create_boxsizer([sent_box, word_box, related_si_box, related_it_box], 
                                        direction=wx.VERTICAL)
        func_box3 = self.create_boxsizer([func_box1, func_box2])
        func_box4 = self.create_static_sizer('Manual Input', [func_box3, button_box], 
                                            expend=0, direction=wx.VERTICAL)
        #exit_box = self.create_boxsizer([logo, qit_button], expend=0, direction=wx.VERTICAL)
        lower_box = self.create_boxsizer([sampling_box, func_box4, logo], expend=0, gap=5)
        main_sizer = self.create_boxsizer([upper_box, lower_box], gap=5,
                                        expend=0, direction=wx.VERTICAL)
        #main_sizer = self.create_boxsizer([upper_box, sampling_box, button_box, func_box2, func_box1], 
        #    expend=0, gap=3, direction=wx.VERTICAL)
        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)


    def create_io_text_ctrl(self):
        """
        create text controlers, including all input-output textbox
        """
        self.in_text = self.create_textbox((400, 600), style=wx.TE_MULTILINE)
        self.out_text = self.create_textbox((500, 600), style=wx.TE_MULTILINE)# | wx.TE_READONLY)
        self.num_samples = self.create_textbox(style=wx.TE_LEFT)
        self.num_samples.SetValue('6')
        self.act_arg_choice = self.create_textbox(style=wx.TE_LEFT)
        self.item_type = self.create_textbox(style=wx.TE_LEFT)
        self.act_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.sent_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.word_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.related_sent_idx = self.create_textbox(style=wx.TE_LEFT)
        self.related_item = self.create_textbox(style=wx.TE_LEFT)


    def create_static_sizer(self, label, items, expend=1, direction=wx.HORIZONTAL):
        """
        create_static_sizer
        """
        box = wx.StaticBox(self.panel, -1, label) 
        sizer = wx.StaticBoxSizer(box, direction) 
        if type(items) == list:
            for item in items:
                sizer.Add(item, expend, wx.EXPAND|wx.ALL)
        else:
            sizer.Add(items, expend, wx.EXPAND|wx.ALL)
        return sizer


    def create_boxsizer(self, items, expend=1, gap=1, direction=wx.HORIZONTAL):
        """
        create_boxsizer
        """
        sizer = wx.BoxSizer(direction) 
        for item in items:
            sizer.Add(item, expend, wx.EXPAND|wx.ALL, gap)
        return sizer


    def show_name(self, label, pos=wx.DefaultPosition, size=wx.DefaultSize, font_style=wx.DEFAULT):
        """
        show static text
        """
        name = wx.StaticText(self.panel, -1, label, pos, size, style=wx.ALIGN_CENTER)
        font = wx.Font(self.font_size, font_style, wx.NORMAL, wx.NORMAL)
        name.SetFont(font)
        return name


    def create_button(self, label, func, pos=wx.DefaultPosition, size=wx.DefaultSize):
        """
        create a button, bind its click-event to a function
        """
        button = buttons.GenButton(self.panel, -1, label, pos, size)  
        self.Bind(wx.EVT_BUTTON, func, button)
        return button


    def create_textbox(self, size=wx.DefaultSize, pos=wx.DefaultPosition, style=wx.TE_MULTILINE):
        """
        create a text controler
        """
        textctrl = wx.TextCtrl(self.panel, -1, "", pos=pos, size=size, style=style)
        font = wx.Font(self.font_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        textctrl.SetFont(font)
        textctrl.SetInsertionPoint(0)
        return textctrl


    def change_font_style(self, text_ctrl, content, color):
        """
        change_font_style
        """
        start = text_ctrl.GetInsertionPoint()
        text_ctrl.AppendText(content)
        end = text_ctrl.GetInsertionPoint()
        text_ctrl.SetStyle(start, end, wx.TextAttr('black', color))


    def show_results(self, hightlight=False):
        """
        show results to output text controller
        """
        self.out_text.Clear()
        out1_start = self.out_text.GetInsertionPoint()
        count_act = 0
        act2sent = {}
        sents = self.cur_text['sents']
        show_ind = []
        if hightlight: # hightlight words with lower reward
            r_t = self.cur_text['r_t']
            sorted_r_t = sorted(zip(r_t, range(len(r_t))), key=lambda x:x[0])
            # print the top-20 words with least reward 
            for r, i in sorted_r_t[: 20]:
                if r < 50:
                    show_ind.append(self.cur_text['word2sent'][i])

        for i in range(len(sents)):
            words = sents[i]['last_sent'] + sents[i]['this_sent']
            self.out_text.AppendText('NO%d: ' % (i + 1))
            # self.out_text.AppendText(' '.join(sents[i]['this_sent']))
            for j, w in enumerate(sents[i]['this_sent']):
                if [i, j] in show_ind:
                    self.change_font_style(self.out_text, w+' ', 'yellow')
                else:
                    self.out_text.AppendText(w+' ')
            #     self.out_text.AppendText('%s(%d) '%(w, j + 1))
            self.out_text.AppendText('\n')
            # self.out_text.AppendText('NO%d: %s\n'%(i, ' '.join(sents[i]['this_sent'])))
            for k, act in enumerate(sents[i]['acts']):
                objs = []
                for oi in act['obj_idxs'][0] + act['obj_idxs'][1]:
                    if oi >= 0:
                        objs.append(words[oi])
                    else:
                        objs.append('UNK')
                act2sent[count_act] = [i, k]
                self.out_text.AppendText(
                    '-->  %s (%s)    '%(words[act['act_idx']], ', '.join(objs)))
                count_act += 1
            self.out_text.AppendText('\n\n')
        self.act2sent = act2sent
        self.out_text.ShowPosition(out1_start)
        

    def OnTextList(self, event):
        """
        show a text and its actions when the text is double-clicked in the list
        """
        if not self.chosen_texts:
            return
        ind = self.candidate_list.GetSelection()
        text = self.chosen_texts[ind]
        text_str = '.\n\n'.join([' '.join(s['this_sent']) for s in text['sents']])
        
        self.cur_cml_rwd.SetLabel('%d' % text['reward'][0])
        self.text_index.SetLabel('%d' % text['text_ind'])
        self.in_text.Clear()
        self.in_text.AppendText(text_str)
        self.in_text.SetInsertionPoint(0)
        self.cur_text = deepcopy(text)
        self.show_results(hightlight=True)


    def OnSampling(self, event):
        """
        create a progress dialog while sampling unlabeled texts
        """
        num_texts = self.num_samples.GetValue().strip()
        if not num_texts:
            num_texts = 5
        else:
            num_texts = int(num_texts)
        dialog = wx.ProgressDialog('Unlabeled Texts Sampling', 'Progress', num_texts,
                style=wx.PD_CAN_ABORT|wx.PD_ELAPSED_TIME|wx.PD_REMAINING_TIME)
        # sampling some texts for labeling
        self.chosen_texts = self.agent.choose_unlabeled_texts(num_texts, dialog)[: num_texts]
        dialog.Destroy()
        self.candidate_list.Clear()
        for i,t in enumerate(self.chosen_texts):
            self.candidate_list.Append('%d. %s' % (i + 1, t['title']))
        self.process.SetLabel('%d/%d' % (len(self.agent.history_texts), len(self.agent.texts)))
        self.clear_boxes()


    def OnParse(self, event):
        """
        Auto parse the user's annotations from the output text controller
        """
        lines = self.out_text.GetValue().strip()
        if not lines: return
        lines = lines.split('\n')
        sents = []
        parsed_data = []
        cur_sent_acts = []
        last_sent, this_sent = [], []
        all_act_types = {1: 0 ,2: 0, 3: 0}
        try:
            for line in lines:
                if not line: continue
                if line.startswith('NO'):
                    parsed_data.append({'last_sent': last_sent, 'this_sent': this_sent,
                                        'acts': cur_sent_acts})
                    cur_sent_acts = []
                    last_sent = deepcopy(this_sent)
                    this_sent = line.split()[1:]
                    sents.append(this_sent) # sent: a list of words
                else:
                    words = last_sent + this_sent
                    tmp_acts = line.split('--> ')
                    for act_obj in tmp_acts:
                        if not act_obj: continue
                        try:
                            act_obj = act_obj.strip().split(' (')
                            acts = act_obj[0].split('_')
                            objs = act_obj[1][:-1].split(', ')
                        except:
                            #ipdb.set_trace()
                            continue
                        # in case the word of action name is not unique in words
                        if acts[0].startswith('2'):
                            act_type = 2
                            related_acts = []
                            acts[0] = acts[0][1:] # skip the number of action type
                        elif acts[0].startswith('3'):
                            act_type = 3
                            ex_acts = acts[0].split('#') # exclude act_type and act
                            related_acts = [int(_) + len(last_sent) for _ in ex_acts[1:-1]]
                            acts[0] = ex_acts[-1]
                        else:
                            act_type = 1
                            related_acts = []
                        
                        if len(acts) > 1:
                            act_idxs = [i for i, w in enumerate(words) if w == acts[0]]
                            assert len(act_idxs) >= int(acts[1])
                            act_idx = act_idxs[int(acts[1]) - 1]
                        else:
                            act_idx = words.index(acts[0])

                        obj_idxs = [[], []]
                        for obj in objs:
                            # two kinds of objects (action arguments): exclusive (1) and essential (0)
                            if obj.startswith('3'):
                                obj_type = 1
                                obj = obj[1:]
                            else:
                                obj_type = 0
                            # in case the word of action arguments is not unique in words
                            tmp_objs = obj.split('_')
                            if len(tmp_objs) > 1:
                                tmp_idxs = [i for i, w in enumerate(words) if w == tmp_objs[0]]
                                assert len(tmp_idxs) >= int(tmp_objs[1])
                                obj_idx = tmp_idxs[int(tmp_objs[1]) - 1]
                            else:
                                if tmp_objs[0] == 'UNK':
                                    obj_idx = -1
                                else:
                                    obj_idx = words.index(tmp_objs[0])
                            obj_idxs[obj_type].append(obj_idx)
                        # save data, PS: auto-parsing could not distinguish exclusive items
                        all_act_types[act_type] += 1
                        cur_sent_acts.append({'act_idx': act_idx, 'obj_idxs': obj_idxs,
                                            'act_type': act_type, 'related_acts': related_acts})
            # save the last act
            parsed_data.append({'last_sent': last_sent, 'this_sent': this_sent,
                                        'acts': cur_sent_acts})
            # the first act is empty
            self.cur_text['sents'] = parsed_data[1:] 
            #ipdb.set_trace()
            if self.cur_text['text_ind'] in self.agent.history_texts:
                print('\nText%d exists and it is replaced now.\n' % self.cur_text['text_ind'])
                ind = self.agent.history_texts.index(self.cur_text['text_ind'])
                self.data[ind] = self.cur_text['sents']
            else:
                self.data.append(self.cur_text['sents']) 
                self.agent.history_texts.append(self.cur_text['text_ind'])
            # save file
            print('Saving data to %s' % self.output_file_name)
            save_pkl([self.agent.history_texts, self.data], self.output_file_name)
            print('Total labeled text: %d' % len(self.data))
            print(all_act_types)

            message = 'AutoParse finished! Currently annotated texts: %d' % len(self.data)
            dlg = wx.MessageDialog(None, message, caption='Reminder', style=wx.OK, pos=wx.DefaultPosition)
            dlg.ShowModal()
            # self.out_text.Clear()
            self.show_results() # update self.act2sent
        except Exception as e:
            dlg = wx.MessageDialog(None, str(e), caption='Errors', style=wx.OK, pos=wx.DefaultPosition)
            dlg.ShowModal()


    def OnExtract(self, event):
        """
        Extract all actions for the text in input text controller
        """
        if len(self.cur_text['sents']) > 0:
            if self.cur_text['text_ind'] not in self.agent.history_texts:
                self.data.append(self.cur_text['sents'])
                self.agent.history_texts.append(self.cur_text['text_ind'])
        raw_text = self.in_text.GetValue() + ' '
        if not raw_text or len(raw_text.split()) < 2:
            return
        raw_text = re.split(r'\n|\r', raw_text) # for text which has not punctuation
        text = []
        for t in raw_text:
            t = re.sub(r'\(|\)|,|;|:', ' ', t)
            for s in re.split(r'\. |\? |\! ', t):
                if len(s) > 1:
                    if s.isupper(): # all words are upper case
                        text.append(s[0]+s[1:].lower())
                    else:
                        text.append(s)
        self.cur_text['sents'], self.cur_text['word2sent'], R_t = self.agent.predict(text)
        r_t = R_t[: -1] - self.agent.gamma * R_t[1: ] # deduced from R_t = r_t + gamma * R_{t+1}
        self.cur_text['r_t'] = r_t
        # use the current time as the index of a text that given by a user
        self.cur_text['text_ind'] = int(time.time())
        self.show_results(hightlight=True)


    def OnDelete(self, event):
        """
        Delete an action (name/argument)
        """
        act_idx = self.act_idx_in.GetValue()
        if not act_idx:
            return
        act_idx = int(act_idx.split()[0])
        if act_idx >= len(self.act2sent):
            return
        # ipdb.set_trace()
        si, ai = self.act2sent[act_idx]
        self.cur_text['sents'][si]['acts'].pop(ai)
        self.show_results()


    def OnRevise(self, event):
        """
        Revise an action (name/argument)
        """
        choice = self.act_arg_choice.GetValue().strip()
        item_type = self.item_type.GetValue().strip()
        act_idx = self.act_idx_in.GetValue().strip()
        sent_idx = self.sent_idx_in.GetValue().strip()
        word_ids = self.word_idx_in.GetValue().split()
        related_sent_idx = self.related_sent_idx.GetValue().strip()
        related_item = self.related_item.GetValue().split()
        self.clear_boxes()
        if choice in ['n', 'N', '0']: # for act
        # labeling an action name
            act_idx = int(act_idx)
            item_type = int(item_type)
            assert item_type in [1, 2, 3]
            si, ai = self.act2sent[act_idx] # find the sent_idx and the act_idx of the sent_acts
            bias = len(self.cur_text['sents'][si]['last_sent'])
            self.cur_text['sents'][si]['acts'][ai]['act_idx'] = int(word_ids[0]) + bias
            self.cur_text['sents'][si]['acts'][ai]['act_type'] = item_type
            if item_type == 3:
                assert len(related_item) > 0
                if len(related_sent_idx.split()) == 1:
                    related_sent_idx = int(related_sent_idx.split()[0])
                    if related_sent_idx != si: # exclusive action names in last_sent
                        bias = 0
                    self.cur_text['sents'][si]['acts'][ai]['related_acts'] = []
                    for ra in related_item:
                        ra = int(ra)
                        self.cur_text['sents'][si]['acts'][ai]['related_acts'].append(ra + bias)
                else:
                    self.cur_text['sents'][si]['acts'][ai]['related_acts'] = []
                    for i, rsi in enumerate(related_sent_idx.split()):
                        # add the length of the last sentence if the related words are in the current sentence
                        bias = len(self.cur_text['sents'][si]['last_sent']) if int(rsi) == si else 0
                        ra = int(related_item[i])
                        self.cur_text['sents'][si]['acts'][ai]['related_acts'].append(ra + bias)
            self.show_results()
        elif choice in ['a', 'A', '1']: # for obj
            # ipdb.set_trace()
            act_idx = int(act_idx)
            # sent_idx = int(sent_idx)
            item_type = int(item_type)
            si, ai = self.act2sent[act_idx]
            if len(sent_idx.split()) == 1: # all arguments in this_sent
                sent_idx = int(sent_idx.split()[0])
                # add the length of the last sentence if the related words are in the current sentence
                bias = len(self.cur_text['sents'][si]['last_sent']) if sent_idx == si else 0
                self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][0] = [int(wi)+bias for wi in word_ids]
                if item_type == 3: # exclusive action arguments
                    self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1] = [int(rs)+bias for rs in related_item]
            else: # arguments in last_sent and this_sent
                sent_idx = [int(s) for s in sent_idx.split()]
                word_ids = [int(w) for w in word_ids]
                assert len(sent_idx) == len(word_ids)
                self.cur_text['sents'][si]['acts'][ai]['obj_idxs'] = [[], []]
                for i in range(len(sent_idx)):
                    bias = len(self.cur_text['sents'][si]['last_sent']) if sent_idx[i] == si else 0
                    self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][0].append(word_ids[i] + bias)
                if item_type == 3:
                    assert len(related_item) > 0
                    if len(related_sent_idx.split()) == 1:
                        related_sent_idx = int(related_sent_idx.split()[0])
                        bias = len(self.cur_text['sents'][si]['last_sent']) if related_sent_idx == si else 0
                        self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1] = [int(ra)+bias for ra in related_item]
                    else:
                        for i, rsi in enumerate(related_sent_idx.split()):
                            bias = len(self.cur_text['sents'][si]['last_sent']) if int(rsi) == si else 0
                            ra = int(related_item[i])
                            self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1].append(ra + bias)
            self.show_results()


    def clear_boxes(self):
        """
        Clear all index boxes
        """
        self.act_arg_choice.Clear()
        self.act_idx_in.Clear()
        self.sent_idx_in.Clear()
        self.word_idx_in.Clear()
        self.related_sent_idx.Clear()
        self.related_item.Clear()


    def OnInsert(self, event):
        """
        Insert an action (name/argument)
        """
        choice = self.act_arg_choice.GetValue().strip()
        item_type = self.item_type.GetValue().strip()
        act_idx = self.act_idx_in.GetValue().strip()
        sent_idx = self.sent_idx_in.GetValue().strip()
        word_ids = self.word_idx_in.GetValue().split()
        related_sent_idx = self.related_sent_idx.GetValue().strip()
        related_item = self.related_item.GetValue().split()
        self.clear_boxes()
        #ipdb.set_trace()
        if choice in ['a', '0']:
            sent_idx = int(sent_idx)
            word_ids = int(word_ids[0])
            item_type = int(item_type)
            bias = len(self.cur_text['sents'][sent_idx]['last_sent'])
            word_ids += bias
            self.cur_text['sents'][sent_idx]['acts'].append({'act_idx': word_ids, 'act_type': item_type,
                                                        'obj_idxs': [[-1], []], 'related_acts': []})
            self.cur_text['sents'][sent_idx]['acts'].sort(key=lambda x:x['act_idx'])
            if item_type == '3':
                ai = -1
                assert len(related_item) > 0
                if len(related_sent_idx.split()) == 1:
                    related_sent_idx = int(related_sent_idx.split()[0])
                    if related_sent_idx != si:
                        bias = 0
                    for ra in related_item:
                        ra = int(ra)
                        self.cur_text['sents'][si]['acts'][ai]['related_acts'].append(ra + bias)
                else:
                    self.cur_text['sents'][si]['acts'][ai]['related_acts'] = []
                    for i, rsi in enumerate(related_sent_idx.split()):
                        if int(rsi) != si:
                            bias = 0
                        else:
                            bias = len(self.cur_text['sents'][si]['last_sent'])
                        ra = int(related_item[i])
                        self.cur_text['sents'][si]['acts'][ai]['related_acts'].append(ra + bias)
        elif choice in ['o', '1']:
            act_idx = int(act_idx)
            item_type = int(item_type)
            si, ai = self.act2sent[act_idx]
            if len(sent_idx.split()) == 1:
                sent_idx = int(sent_idx.split()[0])
                # add the length of the last sentence if the related words are in the current sentence
                bias = len(self.cur_text['sents'][si]['last_sent']) if sent_idx == si else 0
                self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][0] = [int(wi)+bias for wi in word_ids]
                if item_type == 3:
                    self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1] = [int(rs)+bias for rs in related_item]
            else:
                sent_idx = [int(s) for s in sent_idx.split()]
                word_ids = [int(w) for w in word_ids]
                assert len(sent_idx) == len(word_ids)
                self.cur_text['sents'][si]['acts'][ai]['obj_idxs'] = [[], []]
                for i in range(len(sent_idx)):
                    # add the length of the last sentence if the related words are in the current sentence
                    bias = len(self.cur_text['sents'][si]['last_sent']) if sent_idx[i] == si else 0
                    self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][0].append(word_ids[i] + bias)
                if item_type == 3:
                    assert len(related_item) > 0
                    if len(related_sent_idx.split()) == 1:
                        related_sent_idx = int(related_sent_idx.split()[0])
                        bias = len(self.cur_text['sents'][si]['last_sent']) if related_sent_idx == si else 0
                        self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1] = [int(ra)+bias for ra in related_item]
                    else:
                        for i, rsi in enumerate(related_sent_idx.split()):
                            bias = len(self.cur_text['sents'][si]['last_sent']) if int(rsi) == si else 0
                            ra = int(related_item[i])
                            self.cur_text['sents'][si]['acts'][ai]['obj_idxs'][1].append(ra + bias)
        self.show_results()


    def OnSave(self, event):
        """
        Save labeled data
        """
        if len(self.cur_text['sents']) > 0:
            if self.cur_text['text_ind'] not in self.agent.history_texts:
                self.data.append(self.cur_text['sents'])
                self.agent.history_texts.append(self.cur_text['text_ind'])
            else:
                print('\nText%d exists and it is replaced now.\n' % self.cur_text['text_ind'])
                ind = self.agent.history_texts.index(self.cur_text['text_ind'])
                self.data[ind] = self.cur_text['sents']
        print('Saving data to %s' % self.output_file_name)
        save_pkl([self.agent.history_texts, self.data], self.output_file_name)


    def OnQuit(self, event):
        """
        Quit the APP
        """
        dlg = wx.TextEntryDialog(None, "Enter a name for saving the annotation data and press 'OK' ('Cancel' for no saving)!", 
            'Message Window', self.output_file_name, wx.OK | wx.CANCEL)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetValue()
            if filename:
                save_pkl([self.agent.history_texts, self.data], filename)
        #else:
        #    filename = 'data/online_test/labeled_data%s' % get_time()
        #    save_pkl([self.agent.history_texts, self.data], filename)
        wx.Exit()



class MyApp(wx.App):
    """
    A wxpython implemented GUI
    """
    def __init__(self, agent, redirect=True):
        self.agent = agent
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wx.App.__init__(self, redirect)
        
    def OnInit(self):
        self.frame = EASGUI(self.agent)
        self.frame.Show()
        return True
    
    def OnExit(self):
        return True



        
if __name__ == '__main__':
    args = preset_args()
    # K.set_image_data_format('channels_first')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    # set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))) # for keras 
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess: # for tensorflow
        agent = EASDRL_init(args, sess) 
        agent.load_data()
        app = MyApp(agent, redirect=False)
        app.MainLoop()  
