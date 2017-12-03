import gzip
import os
from os.path import isfile

import _pickle as cPickle
import urllib.request


def download(origin):
    '''
    download the corresponding atis file
    from http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/
    '''
    print('Downloading data from %s' % origin)
    filepath = os.path.join('data/', origin.split('/')[-1])
    urllib.request.urlretrieve(origin, filepath)


# def download_dropbox():
#     '''
#     download from drop box in the meantime
#     '''
#     print('Downloading data from https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')
#     os.system('wget -O atis.pkl https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0')
#
#
# def load_dropbox(filename):
#     if not isfile(filename):
#         # download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/'+filename)
#         download_dropbox()
#     # f = gzip.open(filename,'rb')
#     f = open(filename, 'rb')
#     return f


def load_udem(filename):
    filepath = os.path.join('data/', filename)
    if not isfile(filepath):
        download('http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/' + filename)
    f = gzip.open(filepath, 'rb')
    return f


# def atisfull():
#     f = load_dropbox(PREFIX + 'atis.pkl')
#     train_set, test_set, dicts = cPickle.load(f)
#     return train_set, test_set, dicts


def atisfold(fold):
    assert fold in range(5)
    f = load_udem('atis.fold' + str(fold) + '.pkl.gz')
    train_set, valid_set, test_set, dicts = cPickle.load(f, encoding='latin1')
    return train_set, valid_set, test_set, dicts


if __name__ == '__main__':

    ''' visualize a few sentences '''

    import pdb

    # w2ne, w2la = {}, {}  # word2named_entities, word2labels
    # train, test, dic = atisfull()
    train, _, test, dic = atisfold(0)

    w2idx, ne2idx, labels2idx = dic['words2idx'], dic['tables2idx'], dic['labels2idx']

    idx2w = dict((v, k) for k, v in w2idx.items())
    idx2ne = dict((v, k) for k, v in ne2idx.items())
    idx2la = dict((v, k) for k, v in labels2idx.items())

    test_x, test_ne, test_label = test
    train_x, train_ne, train_label = train
    wlength = 35

    for e in ['train', 'test']:  # Train: 3983, Test: 893
        nsentence = 0
        for sw, se, sl in zip(eval(e + '_x'), eval(e + '_ne'), eval(e + '_label')):  # one sentence
            nsentence += 1
            print('Number of sentences:', nsentence)
            print('WORD'.rjust(wlength), 'LABEL'.rjust(wlength))  # text is aligned along the right margin
            for wx, la in zip(sw, sl):  # one word
                print(idx2w[wx].rjust(wlength), idx2la[la].rjust(wlength))
            print('\n' + '**' * 30 + '\n')
            # pdb.set_trace()  # python debugger
