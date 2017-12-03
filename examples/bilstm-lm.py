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

flags = tf.app.flags

flags.DEFINE_integer(
    'batch_size', 1,  # TODO
    'Training batch size.')

flags.DEFINE_integer(
    'embedding_size', 200,  # TODO
    'Word embedding size.')

# flags.DEFINE_integer(
#     'hidden_size', 200,  # TODO
#     'RNN hidden size.')

flags.DEFINE_integer(
    'num_layers', 2,  # TODO
    'RNN layers.')

flags.DEFINE_float(
    'keep_prob', 0.8,  # TODO
    'Drop out keep probability.')

tf.flags.DEFINE_boolean(
    'with_lm', True,
    'with pre-trained language model or not.')

# tf.flags.DEFINE_string(
#     'pbtxt', 'data/graph-2016-09-10.pbtxt',
#     'GraphDef proto text file used to construct model structure.')
#
# tf.flags.DEFINE_string(
#     'ckpt', 'data/ckpt-*',
#     'Checkpoint directory used to fill model values.')
#
# tf.flags.DEFINE_string(
#     'vocab_file', 'data/vocab-2016-09-10.txt',
#     'Vocabulary file.')

FLAGS = flags.FLAGS
# def _LoadLM(gd_file, ckpt_file):
#     """Load the model from GraphDef and Checkpoint.
#
#     Args:
#       gd_file: GraphDef proto text file.
#       ckpt_file: TensorFlow Checkpoint file.
#
#     Returns:
#       TensorFlow session and tensors dict.
#     """
#     with tf.Graph().as_default():
#
#
#         # sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
#         # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#         # sess.run('save/restore_all', {'save/Const:0': ckpt_file})
#         # sess.run(t['states_init'])
#
#     return t

# def _rnn_model(is_training, lm_inputs, st_inputs, labels, lengths, batch_size=FLAGS.batch_size):
#     # TODO
#
#
#             #		learning_rate = tf.Variable(0.0, trainable=False)
#             #		tvars = tf.trainable_variables()
#             #		grads, _ = tf.clip_by_global_norm(tf.gradient(cost, tvars),
#             #										max_grad_norm)
#             #		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#             #		train_op = optimizaer.apply_gradients(
#             #			zip(grads, tvars),
#             #			global_step=tf.contrib.framework.get_or_create_global_step())
#             ##TODO new learning rate

if __name__ == '__main__':

    s = {'fold': 0,  # 5 folds 0,1,2,3,4
         'lr': 0.1,
         'verbose': 0,
         'nhidden': 100,  # number of hidden units
         'seed': 345,
         'emb_dimension': 100,  # dimension of word embedding
         'nepochs': 5}

    folder = os.path.join('out/', os.path.basename(__file__).split('.')[0])  # folder = 'out/bilstm-lm'
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

    sentences_train = [' '.join(list(map(lambda x: idx2word[x], w))) for w in train_lex]
    sentences_valid = [' '.join(list(map(lambda x: idx2word[x], w))) for w in valid_lex]
    sentences_test = [' '.join(list(map(lambda x: idx2word[x], w))) for w in test_lex]

    # sentences = (['I wwant to fly to Boston.', 'I want to fly to USA bla bla bla bla.'])
    # SentenceEmbedding(sentences, 'data/test_embed')

    train_lm_file = 'data/train_language_embedding.npy'
    if not isfile(train_lm_file):
        SentenceEmbedding(sentences_train, train_lm_file)

    valid_lm_file = 'data/valid_language_embedding.npy'
    if not isfile(valid_lm_file):
        SentenceEmbedding(sentences_valid, valid_lm_file)

    test_lm_file = 'data/test_language_embedding.npy'
    if not isfile(test_lm_file):
        SentenceEmbedding(sentences_test, test_lm_file)

    train_lm = np.load(train_lm_file)
    valid_lm = np.load(valid_lm_file)
    test_lm = np.load(test_lm_file)

    # vocab of LM
    # vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)

    # instantiate the model
    np.random.seed(s['seed'])
    random.seed(s['seed'])

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None])
                labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None, nclasses])
                with tf.device("/cpu:0"):
                    word_embedding = tf.get_variable("word_embedding", [vocsize, FLAGS.embedding_size],
                                                     dtype=tf.float32)
                embeddings = tf.nn.embedding_lookup(word_embedding, inputs, name='embeddings')

                # TODO st_embedding_char = CNN(one-hot(sentences))
                # TODO st_embedding_word = GloVe(one-hot(sentences))
                # lm_embedding = LM['lstm/lstm_1/control_dependency']
                if FLAGS.with_lm:
                    lm_embedding = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, None, 1024])
                    embeddings = tf.concat([embeddings, lm_embedding], axis=2)

                with tf.variable_scope('RNN'):
                    # Add a gru_cell
                    if FLAGS.with_lm:
                        hidden_size = FLAGS.embedding_size + 1024
                    else:
                        hidden_size = FLAGS.embedding_size
                    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
                    # if is_training and FLAGS.keep_prob < 1:
                    #    gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=FLAGS.keep_prob)
                    cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * FLAGS.num_layers, state_is_tuple=True)
                    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

                    # if is_training and FLAGS.keep_prob < 1:
                    embeddings = tf.nn.dropout(embeddings, FLAGS.keep_prob)
                    # sequence_length = tf.reshape(lengths, [-1])
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
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost, name='train_op')

        # with tf.name_scope('Test'):
        #     with tf.variable_scope('Model', reuse=None):
        #         _rnn_model(is_training=True, embeddings=train_embeddings, labels=train_labels, lengths=train_lengths)

        # Initialize all variables in the model
        # print(z.get_shape)
        # print(final_state.get_shape)
        # for n in tf.get_default_graph().get_operations():
        #     try:
        #         tmp_tensor = sess.graph.get_tensor_by_name(n.name + ':0')
        #         #print (tmp_tensor.get_shape)
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

        # init
        # sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        # sess.run(lm_init_op)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # train with early stopping on validation set
        best_f1 = -np.inf
        for e in range(s['nepochs']):
            # shuffle
            shuffle([train_lex, train_ne, train_y, train_lm], s['seed'])
            # print(nsentences, '@')
            # print(len(words_train), '@')
            # LM_embedding = SentenceEmbedding(words_train)
            s['ce'] = e
            tic = time.time()
            for i in range(nsentences):
                X = np.asarray([train_lex[i]])
                # print(train_lex[i])
                # print(X.shape)
                # print(words_train[i])
                Y = to_categorical(np.asarray(train_y[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]

                if FLAGS.with_lm:
                    # LM_embedding = SentenceEmbedding(words_train[i])
                    # print(LM_embedding.shape)
                    [_] = sess.run([train_op], feed_dict={inputs: X, labels: Y, lm_embedding: [train_lm[i][:-1]]})
                else:
                    [_] = sess.run([train_op], feed_dict={inputs: X, labels: Y})  # TODO print loss

                if s['verbose']:
                    print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / nsentences),
                          'completed in %.2f (sec) <<\r' % (time.time() - tic))
                    sys.stdout.flush()

            # evaluation
            words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
            groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
            predictions_valid = []
            for i in range(len(valid_lex)):
                X = np.asarray([valid_lex[i]])
                zero_labels = np.zeros([1, X.shape[1], nclasses], dtype=np.int32)

                if FLAGS.with_lm:
                    # LM_embedding = SentenceEmbedding(words_valid[i])
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels,
                                                                           lm_embedding: [valid_lm[i][:-1]]})
                else:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels})
                predict_labels = map(lambda x: idx2label[x], predict_y.argmax(2)[0])
                predictions_valid.append(predict_labels)

            # test
            words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
            groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
            predictions_test = []
            for i in range(len(test_lex)):
                X = np.asarray([test_lex[i]])
                zero_labels = np.zeros([1, X.shape[1], nclasses], dtype=np.int32)

                if FLAGS.with_lm:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels,
                                                                           lm_embedding: [test_lm[i][:-1]]})
                else:
                    [predict_y] = sess.run([prediction_tensor], feed_dict={inputs: X,
                                                                           labels: zero_labels})
                predict_labels = map(lambda x: idx2label[x], predict_y.argmax(2)[0])
                predictions_test.append(predict_labels)

            # print('?', list(predictions_test[0]))
            # print('?', list(groundtruth_test[0]))
            # print('?', list(words_test[0]))

            # evaluation // compute the accuracy using conlleval.pl
            res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
            res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')
            print(res_test)
            print('epoch', e, 'valid F1', res_valid['f1'], 'test F1', res_test['f1'], ' ' * 20)

            if res_valid['f1'] > best_f1:  # TODO best valid-f1?
                # os.makedirs('weights/', exist_ok=True)
                # model.save_weights('weights/best_model.h5', overwrite=True) TODO
                best_f1 = res_valid['f1']
                # if s['verbose']:
                print('NEW BEST: epoch', e, 'best valid F1', res_valid['f1'], 'test F1', res_test['f1'], ' ' * 20)
                s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
                s['tf1'], s['tp'], s['tr'] = res_test['f1'], res_test['p'], res_test['r']
                s['be'] = e
                subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
                subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
            else:
                print('')

    print('BEST RESULT: epoch', s['be'], 'best valid F1', s['vf1'], 'test F1', s['tf1'], 'with the model', folder)
