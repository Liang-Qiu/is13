import os
import random
import subprocess
import sys
import time
from os.path import isfile

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

sys.path.append('../')
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle
from is13.lm_1b.lm_1b_eval import SentenceEmbedding
import is13.pysts.embedding as emb

# --------------------------------------------model hyperparameters-----------------------------------------------------
flags = tf.app.flags

tf.flags.DEFINE_boolean(
    'with_lm', True,
    'With pre-trained language model or not.')

flags.DEFINE_boolean(
    'with_glove', True,
    'Use Glove Embedding or not.')

flags.DEFINE_boolean(
    'bi_lstm', True,
    'Use bidirectional LSTM or forward LSTM.')

flags.DEFINE_integer(
    'fold', 3,  # TODO
    'ATIS dataset fold.')

flags.DEFINE_integer(
    'nsentences', 1000,
    'Partition used for training.')

flags.DEFINE_integer(
    'lr', 0.001,  # TODO
    'Training learning rate.')

flags.DEFINE_integer(
    'verbose', False,
    'Verbose when training.')

flags.DEFINE_integer(
    'batch_size', 1,
    'Training batch size.')

flags.DEFINE_integer(
    'emb_size', 300,
    'Word embedding size which is the same as the number of hidden units.')

flags.DEFINE_integer(
    'nlayers', 2,
    'RNN hidden layer number.')

flags.DEFINE_float(
    'nepochs', 40,
    'Training epochs.')

flags.DEFINE_float(
    'keep_prob', 0.8,
    'Drop out keep probability.')

flags.DEFINE_integer(
    'seed', 345,
    'Random number generation seed.')

FLAGS = flags.FLAGS
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    folder = os.path.join('out/', os.path.basename(__file__).split('.')[0])  # folder = 'out/bilstm-lm'
    os.makedirs(folder, exist_ok=True)

    print('ATIS fold:', FLAGS.fold)
    print('with language model:', FLAGS.with_lm)
    print('with GloVe:', FLAGS.with_glove)
    print('with bi-LSTM:', FLAGS.bi_lstm)
    print('Training epochs:', FLAGS.nepochs)
    print('Training data size:', FLAGS.nsentences)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(FLAGS.fold)  # size: 3983, 893, 893
    idx2word = dict((k, v) for v, k in dic['words2idx'].items())
    idx2label = dict((k, v) for v, k in dic['labels2idx'].items())

    # id, named entity, label
    train_x, train_ne, train_y = train_set
    valid_x, valid_ne, valid_y = valid_set
    test_x, test_ne, test_y = test_set

    vocsize = len(dic['words2idx'])
    nclasses = len(dic['labels2idx'])
    nsentences = len(train_x)
    assert FLAGS.nsentences <= nsentences, 'Training data size needs to be less than 3983.'

    sentences_train = [' '.join(list(map(lambda x: idx2word[x], s))) for s in train_x]
    sentences_valid = [' '.join(list(map(lambda x: idx2word[x], s))) for s in valid_x]
    sentences_test = [' '.join(list(map(lambda x: idx2word[x], s))) for s in test_x]

    # run and save the embedding matrix if it doesn't exist
    train_lm_file = 'data/train_language_embedding_' + str(FLAGS.fold) + '.npy'
    if not isfile(train_lm_file):
        SentenceEmbedding(sentences_train, train_lm_file)

    valid_lm_file = 'data/valid_language_embedding_' + str(FLAGS.fold) + '.npy'
    if not isfile(valid_lm_file):
        SentenceEmbedding(sentences_valid, valid_lm_file)

    test_lm_file = 'data/test_language_embedding_' + str(FLAGS.fold) + '.npy'
    if not isfile(test_lm_file):
        SentenceEmbedding(sentences_test, test_lm_file)

    # load pre-trained language model embedding matrix
    train_lm = np.load(train_lm_file)
    valid_lm = np.load(valid_lm_file)
    test_lm = np.load(test_lm_file)

    # GloVe embedding
    glove = emb.GloVe(N=FLAGS.emb_size)
    sd = 1 / np.sqrt(FLAGS.emb_size)
    glove_embedding = np.random.normal(0, scale=sd, size=[vocsize, FLAGS.emb_size])  # words not in GloVe dict
    glove_embedding = glove_embedding.astype(np.float32)
    for k, v in idx2word.items():
        if v in glove.w.keys():
            glove_embedding[k] = glove.g[glove.w[v]]

    # instantiate the model
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # record training result
    record = {}

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None])
                labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None, nclasses])

                if FLAGS.with_glove:
                    embeddings = tf.nn.embedding_lookup(glove_embedding, inputs, name='embeddings')
                else:
                    with tf.device("/cpu:0"):
                        word_embedding = tf.get_variable("word_embedding", [vocsize, FLAGS.emb_size],
                                                         dtype=tf.float32)
                    embeddings = tf.nn.embedding_lookup(word_embedding, inputs, name='embeddings')

                if FLAGS.with_lm:
                    lm_embedding = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, 1024])
                    embeddings = tf.concat([embeddings, lm_embedding], axis=2)

                with tf.variable_scope('RNN'):
                    # Add a gru_cell
                    if FLAGS.with_lm:
                        hidden_size = FLAGS.emb_size + 1024
                    else:
                        hidden_size = FLAGS.emb_size
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)

                    # TODO if is_training and FLAGS.keep_prob < 1:
                    #    gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=FLAGS.keep_prob)

                    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.nlayers, state_is_tuple=True)
                    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

                    # TODO if is_training and FLAGS.keep_prob < 1:
                    embeddings = tf.nn.dropout(embeddings, FLAGS.keep_prob)

                    if FLAGS.bi_lstm:
                        (outputs, final_state) = tf.nn.bidirectional_dynamic_rnn(cell, cell, embeddings,
                                                                                 initial_state_fw=initial_state,
                                                                                 initial_state_bw=initial_state)
                        output_f = outputs[0]
                        output_b = outputs[1]
                        output_f = tf.reshape(output_f, [-1, hidden_size])
                        output_b = tf.reshape(output_b, [-1, hidden_size])
                        weights_f = tf.get_variable("weights_f", [hidden_size, nclasses], dtype=tf.float32)
                        weights_b = tf.get_variable("weights_b", [hidden_size, nclasses], dtype=tf.float32)
                        biases_f = tf.get_variable("biases_f", [nclasses], dtype=tf.float32)
                        biases_b = tf.get_variable("biases_b", [nclasses], dtype=tf.float32)

                        logits = tf.add(tf.add(tf.matmul(output_f, weights_f), biases_f),
                                        tf.add(tf.matmul(output_b, weights_b), biases_b), name="logits")
                        logits = tf.reshape(logits, [FLAGS.batch_size, -1, nclasses])

                    else:
                        (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
                        output = tf.reshape(outputs, [-1, hidden_size])
                        weights = tf.get_variable("weights", [hidden_size, nclasses], dtype=tf.float32)
                        biases = tf.get_variable("biases", [nclasses], dtype=tf.float32)
                        logits = tf.add(tf.matmul(output, weights), biases, name="logits")
                        logits = tf.reshape(logits, [FLAGS.batch_size, -1, nclasses])

                    prediction = tf.nn.softmax(logits, name='prediction')

                    # if is_training:
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cost = tf.reduce_mean(cross_entropy, name='cost')
                    tf.summary.scalar('cost', cost)
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.lr).minimize(cost, name='train_op')

        # for n in tf.get_default_graph().get_operations():
        #     try:
        #         tmp_tensor = sess.graph.get_tensor_by_name(n.name + ':0')
        #         print (tmp_tensor.get_shape)
        #     except KeyError:
        #         continue
        # word_embedding_tensor = sess.graph.get_tensor_by_name('Train/Model/embedding:0')
        # print(sess.run(tf.shape(word_embedding_tensor)))
        train_op = sess.graph.get_operation_by_name('Train/Model/RNN/train_op')
        prediction_tensor = sess.graph.get_tensor_by_name('Train/Model/RNN/prediction:0')
        # print(prediction_tensor.get_shape())

        # language model tensors
        # lm_init_op = sess.graph.get_operation_by_name('Train/Model/states_init')
        # char_inputs_in_tensor = sess.graph.get_tensor_by_name('Train/Model/char_inputs_in')
        # inputs_in_tensor = sess.graph.get_tensor_by_name('Train/Model/inputs_in')
        # targets_in_tensor = sess.graph.get_tensor_by_name('Train/Model/targets_in')
        # targets_weights_in_tensor = sess.graph.get_tensor_by_name('Train/Model/target_weights_in')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # train with early stopping on training set
        best_f1 = -np.inf
        early_stop_count = 0
        for e in range(FLAGS.nepochs):
            # shuffle
            shuffle([train_x, train_ne, train_y, train_lm], FLAGS.seed)
            record['current epoch'] = e
            tic = time.time()

            # training
            for i in range(FLAGS.nsentences):
                X = np.asarray([train_x[i]])
                Y = to_categorical(np.asarray(train_y[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]

                if FLAGS.with_lm:
                    # don't use the lm_embedding for the start token <S>
                    [_] = sess.run([train_op], feed_dict={inputs: X, labels: Y, lm_embedding: [train_lm[i][1:]]})
                else:
                    [_] = sess.run([train_op], feed_dict={inputs: X, labels: Y})

                if FLAGS.verbose:
                    print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / FLAGS.nsentences),
                          'completed in %.2f (sec) <<\r' % (time.time() - tic))
                    sys.stdout.flush()

            # evaluation
            words_valid = [map(lambda x: idx2word[x], w) for w in valid_x]
            groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
            predictions_valid = []
            for i in range(len(valid_x)):
                X = np.asarray([valid_x[i]])
                zero_labels = np.zeros([1, X.shape[1], nclasses], dtype=np.int32)

                if FLAGS.with_lm:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels,
                                                                           lm_embedding: [valid_lm[i][1:]]})
                else:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels})
                predict_labels = map(lambda x: idx2label[x], predict_y.argmax(2)[0])
                predictions_valid.append(predict_labels)

            # test
            words_test = [map(lambda x: idx2word[x], w) for w in test_x]
            groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
            predictions_test = []
            for i in range(len(test_x)):
                X = np.asarray([test_x[i]])
                zero_labels = np.zeros([1, X.shape[1], nclasses], dtype=np.int32)

                if FLAGS.with_lm:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels,
                                                                           lm_embedding: [test_lm[i][1:]]})
                else:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels})
                predict_labels = map(lambda x: idx2label[x], predict_y.argmax(2)[0])
                predictions_test.append(predict_labels)

            # evaluation // compute the accuracy using conlleval.pl
            res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
            res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')
            if FLAGS.verbose:
                print('epoch', e, 'valid F1', res_valid['f1'], 'test F1', res_test['f1'], ' ' * 20)

            if res_valid['f1'] > best_f1:
                # os.makedirs('weights/', exist_ok=True)
                # model.save_weights('weights/best_model.h5', overwrite=True)
                best_f1 = res_valid['f1']
                if FLAGS.verbose:
                    print('NEW BEST: epoch', e, 'best valid F1', res_valid['f1'], 'test F1', res_test['f1'], ' ' * 20)
                record['vf1'], record['vp'], record['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
                record['tf1'], record['tp'], record['tr'] = res_test['f1'], res_test['p'], res_test['r']
                record['best epoch'] = e
                subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
                subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
                early_stop_count = 0
            else:
                early_stop_count += 1
                print('')

            if early_stop_count >= 3:
                break
    print('BEST RESULT: epoch', record['best epoch'], 'best valid F1', record['vf1'], 'test F1', record['tf1'],
          'with the model', folder)
