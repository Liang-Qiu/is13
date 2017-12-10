"""
Copyright 2017 Liang Qiu, Zihan Li, Yuanyi Ding

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Task-specific callbacks for the fit() function.
"""

from keras.callbacks import Callback
import numpy as np

import pysts.eval as ev
import pysts.loader as loader


class AnsSelCB(Callback):
    """ A callback that monitors answer selection validation MRR after each epoch """
    def __init__(self, task, val_gr):
        self.task = task
        self.val_s0 = np.array(val_gr['si0']) + np.array(val_gr['sj0'])
        self.val_gr = val_gr

    def on_epoch_end(self, epoch, logs={}):
        ypred = self.task.predict(self.model, self.val_gr)
        mrr = ev.mrr(self.val_s0, self.val_gr['score'], ypred)
        print('                                                       val mrr %f' % (mrr,))
        logs['mrr'] = mrr


class ParaCB(Callback):
    """ A callback that monitors paraphrasing validation accuracy after each epoch """
    def __init__(self, task, val_gr):
        self.task = task
        self.val_gr = val_gr  # graph_input()

    def on_epoch_end(self, epoch, logs={}):
        ypred = self.task.predict(self.model, self.val_gr)
        tmp = []
        for yp in ypred:
            tmp.append(yp[0]) 
        ypred = np.array(tmp)
        acc, y0acc, y1acc, balacc, f_score = ev.binclass_accuracy(self.val_gr['score'], ypred)
        print('           val acc %f    val f1 %f' % (acc, f_score))
        logs['acc'] = acc


class HypEvCB(Callback):
    """ A callback that monitors hypothesis evaluation validation accuracy after each epoch """
    def __init__(self, task, val_gr):
        self.task = task
        self.val_gr = val_gr  # graph_input()

    def on_epoch_end(self, epoch, logs={}):
        ypred = self.task.predict(self.model, self.val_gr)
        if 'qids' not in self.val_gr or self.val_gr['qids'] is None:
            acc = ev.binclass_accuracy(self.val_gr['score'], ypred)[0]
            print('                                                       val acc %f' % (acc,))
        else:
            acc = ev.recall_at(self.val_gr['qids'], self.val_gr['score'], ypred, N=1)
            print('                                                       val abcdacc %f' % (acc,))
        logs['acc'] = acc


class STSPearsonCB(Callback):
    def __init__(self, task, train_gr, val_gr):
        self.task = task
        self.train_gr = train_gr
        self.val_gr = val_gr
    def on_epoch_end(self, epoch, logs={}):
        prtr = ev.eval_sts(self.task.predict(self.model, self.train_gr),
                           loader.sts_categorical2labels(self.train_gr['classes']), 'Train', quiet=True).Pearson
        prval = ev.eval_sts(self.task.predict(self.model, self.val_gr),
                            loader.sts_categorical2labels(self.val_gr['classes']), 'Val', quiet=True).Pearson
        print('                  train Pearson %f    val Pearson %f' % (prtr, prval))
        logs['pearson'] = prval


class RTECB(Callback):
    """ A callback that monitors RTE validation accuracy after each epoch """
    def __init__(self, task):
        self.task = task

    def on_epoch_end(self, epoch, logs={}):
        ypred=[]
        for ogr in self.task.sample_pairs(self.task.grv, batch_size=len(self.task.grv['score']), shuffle=False, once=True):
            ypred += list(self.model.predict(ogr)['score'])
        ypred = np.array(ypred)
        acc, cls_acc = ev.multiclass_accuracy(self.task.grv['score'], ypred)
        print('                                                       val acc %f' % (acc,))
        logs['acc'] = acc
