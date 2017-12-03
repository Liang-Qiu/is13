import os
import random
import subprocess
import sys
import time

import numpy as np
from keras.layers import Embedding, SimpleRNN, Dense, Activation, TimeDistributed
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

sys.path.append('../')
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle

if __name__ == '__main__':

    s = {'fold': 3,  # 5 folds 0,1,2,3,4
         'lr': 0.1,
         'verbose': 0,
         'nhidden': 100,  # number of hidden units
         'seed': 345,
         'emb_dimension': 100,  # dimension of word embedding
         'nepochs': 50}

    folder = os.path.join('out/', os.path.basename(__file__).split('.')[0])  # folder = 'out/elman-keras'
    os.makedirs(folder, exist_ok=True)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(s['fold'])
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex, test_ne, test_y = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_lex)

    # instantiate the model
    np.random.seed(s['seed'])
    random.seed(s['seed'])

    model = Sequential()
    model.add(Embedding(vocsize, s['emb_dimension']))
    model.add(SimpleRNN(s['nhidden'], activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(units=nclasses)))
    model.add(Activation("softmax"))

    sgd = SGD(lr=s['lr'], momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # train with early stopping on validation set
    best_f1 = -np.inf
    for e in range(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_ne, train_y], s['seed'])
        s['ce'] = e
        tic = time.time()
        for i in range(nsentences):
            X = np.asarray([train_lex[i]])
            Y = to_categorical(np.asarray(train_y[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]
            if X.shape[1] == 1:
                continue  # bug with X, Y of len 1
            model.train_on_batch(X, Y)

            if s['verbose']:
                print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / nsentences),
                      'completed in %.2f (sec) <<\r' % (time.time() - tic))
                sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        predictions_test = [map(lambda x: idx2label[x],
                                model.predict_on_batch(np.asarray([x])).argmax(2)[0])
                            for x in test_lex]
        print(len(predictions_test))
        groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
        words_test = [map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [map(lambda x: idx2label[x],
                                 model.predict_on_batch(np.asarray([x])).argmax(2)[0])
                             for x in valid_lex]
        groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
        words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]

        # print('?', len(list(predictions_test[0])))
        # print('?', len(list(groundtruth_test[0])))
        # print('?', list(words_test[0]))
        # evaluation // compute the accuracy using conlleval.pl
        res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:  # TODO best valid-f1?
            os.makedirs('weights/', exist_ok=True)
            model.save_weights('weights/best_model.h5', overwrite=True)
            best_f1 = res_valid['f1']
            # if s['verbose']:
            print('NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20)
            s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test['f1'], res_test['p'], res_test['r']
            s['be'] = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print('')

    print('BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder)
