import wx
import re
import time
import ipdb
import pickle
import argparse
import wx.lib.buttons as buttons
from utils import get_time


def EASDRL_init():
    import tensorflow as tf
    from main import preset_args, args_init
    from Agent import Agent
    from MultiAgent import MultiAgent
    from KerasEADQN import DeepQLearner
    from Environment import Environment
    from AFEnvironment import AFEnvironment
    from ReplayMemory import ReplayMemory
    from gensim.models import KeyedVectors
    from keras.backend.tensorflow_backend import set_session
    args = preset_args()
    args.gui_mode = True
    args.fold_id = 0
    args.actionDB = 'cooking'
    args.agent_mode = 'multi'
    args.replay_size = 1000
    args.gpu_rate = 0.1
    args.load_weights = 'weights'
    args = args_init(args)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    # args.optimizer = 'rmsprop'
    # args.positive_rate = 0.6
    # args.learning_rate = 0.001
    env_act = Environment(args)
    net_act = DeepQLearner(args, 'act')

    mem_act = ReplayMemory(args, 'act')
    # args.optimizer = 'adadelta'
    # args.positive_rate = 0.6
    env_obj = AFEnvironment(args)
    net_obj = DeepQLearner(args, 'arg')
    mem_obj = ReplayMemory(args, 'arg')
    agent = MultiAgent(env_act, env_obj, mem_act, mem_obj, net_act, net_obj, args)
    #ipdb.set_trace()

    if args.load_weights:
        print('Loading weights ...')
        if args.actionDB == 'all':
            filename = 'weights/online_test/%s/eas/k%d_fold%d.h5' % (args.actionDB, args.k_fold, args.fold_id)
            net_act.load_weights(filename)
            filename = 'weights/online_test/%s/af/k%d_fold%d.h5' % (args.actionDB, args.k_fold, args.fold_id)
            net_obj.load_weights(filename)
        else:
            filename = 'data/online_test/%s/eas/fold%d.h5' % (args.actionDB, args.fold_id)
            net_act.load_weights(filename)
            filename = 'data/online_test/%s/af/fold%d.h5' % (args.actionDB, args.fold_id)
            net_obj.load_weights(filename)

    #raw_text = open('data/online_test/test.txt').read()
    #raw_text = re.sub(r'\n|\r|\(|\)|,|;', ' ', raw_text)
    #text = re.split(r'\. |\? |\! ', raw_text)
    #agent.predict(text)
    return agent



class EASGUI(wx.Frame):
    """docstring for EASGUI"""
    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Action Sequence Extraction")
        self.panel = wx.Panel(self)
        self.font_size = 12
        # img = wx.Image('data/planlab_logo.png', wx.BITMAP_TYPE_ANY)
        # logo = wx.StaticBitmap(self.panel, -1, wx.BitmapFromImage(img))
        ext_button = self.create_button('Extract', self.OnExtract)
        del_button = self.create_button('Delete', self.OnDelete)
        rvs_button = self.create_button('Revise', self.OnRevise)
        ins_button = self.create_button('Insert', self.OnInsert)  
        qit_button = self.create_button('Quit', self.OnQuit)
        self.create_io_text_ctrl()   
 
        in_box = self.create_static_sizer('Input Texts', self.in_text)
        out1_box = self.create_static_sizer('Output Results', self.out1_text)
        #out2_box = self.create_static_sizer('Action Sequence', self.out2_text)
        choice_box = self.create_boxsizer([self.show_name('Act/Arg:'), self.act_obj_choice])
        type_box = self.create_boxsizer([self.show_name('ActType/ArgType:'), self.item_type])
        #act_box = self.create_boxsizer([self.show_name('ActSeqNum:'), self.act_idx_in])
        sent_box = self.create_boxsizer([self.show_name('SentId:'), self.sent_idx_in])
        word_box = self.create_boxsizer([self.show_name('ActId/ArgId:'), self.word_idx_in])
        related_si_box = self.create_boxsizer([self.show_name('ExSentId:'), self.related_sent_idx])
        related_it_box = self.create_boxsizer([self.show_name('ExActId/ExArgId:'), self.related_item])
        #revise_boxes = self.create_boxsizer(
        #    [choice_box, type_box, act_box, sent_box, word_box, related_si_box, related_it_box], gap=0, direction=wx.HORIZONTAL)


        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.create_boxsizer([in_box, out1_box]), 0, wx.EXPAND, 1) 
        #main_sizer.Add(self.create_boxsizer([revise_boxes, out2_box]), 0, wx.EXPAND, 1)
        main_sizer.Add(self.create_boxsizer([choice_box, type_box]), 0, wx.EXPAND, 1)
        main_sizer.Add(self.create_boxsizer([sent_box, word_box]), 0, wx.EXPAND, 1)
        main_sizer.Add(self.create_boxsizer([related_si_box, related_it_box]), 0, wx.EXPAND, 1)
        main_sizer.Add(self.create_boxsizer(
            [ext_button, del_button, rvs_button, ins_button, qit_button]), 0, wx.EXPAND, 1)
        self.panel.SetSizer(main_sizer)
        main_sizer.Fit(self)
        self.agent = EASDRL_init()
        self.data = []
        self.current_sents = []
        self.act2sent = {}


    def create_io_text_ctrl(self):
        self.in_text = self.create_textbox((400, 400))
        self.out1_text = self.create_textbox((400, 400), style=wx.TE_MULTILINE)# | wx.TE_READONLY)
        #self.out2_text = self.create_textbox((400, 200), style=wx.TE_MULTILINE)# | wx.TE_READONLY)
        self.act_obj_choice = self.create_textbox(style=wx.TE_LEFT)
        self.item_type = self.create_textbox(style=wx.TE_LEFT)
        self.item_type.SetValue('1')
        #self.act_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.sent_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.word_idx_in = self.create_textbox(style=wx.TE_LEFT)
        self.related_sent_idx = self.create_textbox(style=wx.TE_LEFT)
        self.related_item = self.create_textbox(style=wx.TE_LEFT)


    def create_static_sizer(self, label, items, expend=1, direction=wx.HORIZONTAL):
        box = wx.StaticBox(self.panel, -1, label)
        sizer = wx.StaticBoxSizer(box, direction)
        if type(items) == list:
            for item in items:
                sizer.Add(item, expend, wx.EXPAND|wx.ALL)
        else:
            sizer.Add(items, expend, wx.EXPAND|wx.ALL)
        return sizer


    def create_boxsizer(self, items, expend=1, gap=5, direction=wx.HORIZONTAL):
        sizer = wx.BoxSizer(direction)
        for item in items:
            sizer.Add(item, expend, wx.EXPAND|wx.ALL, gap)
        return sizer


    def show_name(self, label, pos=wx.DefaultPosition, size=wx.DefaultSize, font_style=wx.DEFAULT):
        name = wx.StaticText(self.panel, -1, label, pos, size, style=wx.ALIGN_CENTER)
        font = wx.Font(self.font_size, font_style, wx.NORMAL, wx.NORMAL)
        name.SetFont(font)
        return name


    def create_button(self, label, func, pos=wx.DefaultPosition, size=wx.DefaultSize):
        button = buttons.GenButton(self.panel, -1, label, pos, size)  
        self.Bind(wx.EVT_BUTTON, func, button)
        return button


    def create_textbox(self, size=wx.DefaultSize, pos=wx.DefaultPosition, style=wx.TE_MULTILINE | wx.TE_RICH2):
        textctrl = wx.TextCtrl(self.panel, -1, "", pos=pos, size=size, style=style)
        font = wx.Font(self.font_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        textctrl.SetFont(font)
        textctrl.SetInsertionPoint(0)
        return textctrl


    def show_results(self):
        #ipdb.set_trace()
        self.out1_text.Clear()
        #self.out2_text.Clear()
        out1_start = self.out1_text.GetInsertionPoint()
        #out2_start = self.out2_text.GetInsertionPoint()
        #self.out2_text.AppendText('\n')
        count_act = 0
        act2sent = {}
        sents = self.current_sents
        for i in range(len(sents)):
            words = sents[i]['last_sent'] + sents[i]['this_sent']
            self.out1_text.AppendText('\nNO%d: ' % (i + 1))
            for j, w in enumerate(sents[i]['this_sent']):
                self.out1_text.AppendText('%s(%d) '%(w, j + 1))
            self.out1_text.AppendText('\n')
            #self.out1_text.AppendText('NO%d: %s\n'%(i, ' '.join(sents[i]['this_sent'])))
            for k, act in enumerate(sents[i]['acts']):
                objs = []
                for oi in act['obj_idxs'][0]+act['obj_idxs'][1]:
                    if oi >= 0:
                        objs.append(words[oi])
                    else:
                        objs.append('UNKNOWN_TOKEN')
                act2sent[count_act] = [i, k]
                self.out1_text.AppendText(
                    '<%d>  %s (%s)    '%(count_act + 1, words[act['act_idx']], ', '.join(objs)))
                #self.out2_text.AppendText(
                #    '<%d>  %s (%s)\n'%(count_act + 1, words[act['act_idx']], ', '.join(objs)))
                count_act += 1
            if len(sents[i]['acts']) > 0:
                self.out1_text.AppendText('\n')
        self.act2sent = act2sent
        self.out1_text.ShowPosition(out1_start)
        #self.out2_text.ShowPosition(out2_start)


    def OnExtract(self, event):
        #ipdb.set_trace()
        if len(self.current_sents) > 0:
            self.data.append(self.current_sents)
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
        self.current_sents = self.agent.predict(text)
        self.show_results()


    def OnDelete(self, event):
        act_idx = self.act_idx_in.GetValue()
        if not act_idx:
            return
        act_idx = int(act_idx.split()[0])
        if act_idx >= len(self.act2sent):
            return
        #ipdb.set_trace()
        si, ai = self.act2sent[act_idx]
        self.current_sents[si]['acts'].pop(ai)
        self.show_results()


    def OnRevise(self, event):
        choice = self.act_obj_choice.GetValue().strip()
        item_type = self.item_type.GetValue().strip()
        act_idx = self.act_idx_in.GetValue().strip()
        sent_idx = self.sent_idx_in.GetValue().strip()
        word_ids = self.word_idx_in.GetValue().split()
        related_sent_idx = self.related_sent_idx.GetValue().strip()
        related_item = self.related_item.GetValue().split()
        self.clear_boxes()
        if choice in ['n', 'N', '0']: # for act
            act_idx = int(act_idx)
            sent_idx = int(sent_idx)
            item_type = int(item_type)
            assert item_type in [1, 2, 3]
            si, ai = self.act2sent[act_idx]
            bias = len(self.current_sents[si]['last_sent'])
            self.current_sents[si]['acts'][ai]['act_idx'] = int(word_ids[0]) + bias
            self.current_sents[si]['acts'][ai]['act_type'] = item_type
            if item_type == 3:
                assert len(related_item) > 0
                if len(related_sent_idx.split()) == 1:
                    related_sent_idx = int(related_sent_idx.split()[0])
                    if related_sent_idx != si:
                        bias = 0
                    self.current_sents[si]['acts'][ai]['related_acts'] = []
                    for ra in related_item:
                        ra = int(ra)
                        self.current_sents[si]['acts'][ai]['related_acts'].append(ra + bias)
                else:
                    self.current_sents[si]['acts'][ai]['related_acts'] = []
                    for i, rsi in enumerate(related_sent_idx.split()):
                        if int(rsi) != si:
                            bias = 0
                        else:
                            bias = len(self.current_sents[si]['last_sent'])
                        ra = int(related_item[i])
                        self.current_sents[si]['acts'][ai]['related_acts'].append(ra + bias)
            self.show_results()
        elif choice in ['a', 'A', '1']: # for obj
            #ipdb.set_trace()
            act_idx = int(act_idx)
            item_type = int(item_type)
            si, ai = self.act2sent[act_idx]
            if len(sent_idx.split()) == 1:
                sent_idx = int(sent_idx.split()[0])
                if sent_idx == si:
                    bias = len(self.current_sents[si]['last_sent'])
                else:
                    bias = 0
                self.current_sents[si]['acts'][ai]['obj_idxs'][0] = [int(wi)+bias for wi in word_ids]
                if item_type == 3:
                    self.current_sents[si]['acts'][ai]['obj_idxs'][1] = [int(rs)+bias for rs in related_item]
            else:
                sent_idx = [int(s) for s in sent_idx.split()]
                word_ids = [int(w) for w in word_ids]
                assert len(sent_idx) == len(word_ids)
                self.current_sents[si]['acts'][ai]['obj_idxs'] = [[], []]
                for i in range(len(sent_idx)):
                    if sent_idx[i] == si:
                        bias = len(self.current_sents[si]['last_sent'])
                    else:
                        bias = 0
                    self.current_sents[si]['acts'][ai]['obj_idxs'][0].append(word_ids[i] + bias)
                if item_type == 3:
                    assert len(related_item) > 0
                    if len(related_sent_idx.split()) == 1:
                        related_sent_idx = int(related_sent_idx.split()[0])
                        if related_sent_idx == si:
                            bias = len(self.current_sents[si]['last_sent'])
                        else:
                            bias = 0
                        self.current_sents[si]['acts'][ai]['obj_idxs'][1] = [int(ra)+bias for ra in related_item]
                    else:
                        for i, rsi in enumerate(related_sent_idx.split()):
                            if int(rsi) != si:
                                bias = 0
                            else:
                                bias = len(self.current_sents[si]['last_sent'])
                            ra = int(related_item[i])
                            self.current_sents[si]['acts'][ai]['obj_idxs'][1].append(ra + bias)
            self.show_results()


    def clear_boxes(self):
        self.act_obj_choice.Clear()
        self.act_idx_in.Clear()
        self.sent_idx_in.Clear()
        self.word_idx_in.Clear()
        self.related_sent_idx.Clear()
        self.related_item.Clear()


    def OnInsert(self, event):
        choice = self.act_obj_choice.GetValue().strip()
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
            self.current_sents[sent_idx]['acts'].append({'act_idx': word_ids, 'act_type': item_type,
                                                        'obj_idxs': [[-1], []], 'related_acts': []})
            self.current_sents[sent_idx]['acts'].sort(key=lambda x:x['act_idx'])
            if item_type == '3':
                ai = -1
                assert len(related_item) > 0
                if len(related_sent_idx.split()) == 1:
                    related_sent_idx = int(related_sent_idx.split()[0])
                    if related_sent_idx != si:
                        bias = 0
                    for ra in related_item:
                        ra = int(ra)
                        self.current_sents[si]['acts'][ai]['related_acts'].append(ra + bias)
                else:
                    self.current_sents[si]['acts'][ai]['related_acts'] = []
                    for i, rsi in enumerate(related_sent_idx.split()):
                        if int(rsi) != si:
                            bias = 0
                        else:
                            bias = len(self.current_sents[si]['last_sent'])
                        ra = int(related_item[i])
                        self.current_sents[si]['acts'][ai]['related_acts'].append(ra + bias)
        elif choice in ['o', '1']:
            act_idx = int(act_idx)
            item_type = int(item_type)
            si, ai = self.act2sent[act_idx]
            if len(sent_idx.split()) == 1:
                sent_idx = int(sent_idx.split()[0])
                if sent_idx == si:
                    bias = len(self.current_sents[si]['last_sent'])
                else:
                    bias = 0
                self.current_sents[si]['acts'][ai]['obj_idxs'][0] = [int(wi)+bias for wi in word_ids]
                if item_type == 3:
                    self.current_sents[si]['acts'][ai]['obj_idxs'][1] = [int(rs)+bias for rs in related_item]
            else:
                sent_idx = [int(s) for s in sent_idx.split()]
                word_ids = [int(w) for w in word_ids]
                assert len(sent_idx) == len(word_ids)
                self.current_sents[si]['acts'][ai]['obj_idxs'] = [[], []]
                for i in range(len(sent_idx)):
                    if sent_idx[i] == si:
                        bias = len(self.current_sents[si]['last_sent'])
                    else:
                        bias = 0
                    self.current_sents[si]['acts'][ai]['obj_idxs'][0].append(word_ids[i] + bias)
                if item_type == 3:
                    assert len(related_item) > 0
                    if len(related_sent_idx.split()) == 1:
                        related_sent_idx = int(related_sent_idx.split()[0])
                        if related_sent_idx == si:
                            bias = len(self.current_sents[si]['last_sent'])
                        else:
                            bias = 0
                        self.current_sents[si]['acts'][ai]['obj_idxs'][1] = [int(ra)+bias for ra in related_item]
                    else:
                        for i, rsi in enumerate(related_sent_idx.split()):
                            if int(rsi) != si:
                                bias = 0
                            else:
                                bias = len(self.current_sents[si]['last_sent'])
                            ra = int(related_item[i])
                            self.current_sents[si]['acts'][ai]['obj_idxs'][1].append(ra + bias)
        self.show_results()


    def OnQuit(self, event):
        dlg = wx.TextEntryDialog(None, 'Input file name to save or cancel(do not save)!', 
            'Message Window', 'data/online_test/online_labeled_text.pkl', wx.OK | wx.CANCEL)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetValue()
            if filename:
                with open(filename, 'wb') as f:
                    pickle.dump(self.data, f)
        wx.Exit()



class MyApp(wx.App):
    def __init__(self, redirect=True):
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wx.App.__init__(self, redirect)
        
    def OnInit(self):
        self.frame = EASGUI()
        self.frame.Show()
        return True
    
    def OnExit(self):
        pass



        
if __name__ == '__main__':
    #EASDRL_init()
    app = MyApp(redirect=False)
    app.MainLoop()    