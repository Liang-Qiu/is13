import os
import random
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from keras.utils.np_utils import to_categorical

sys.path.append('../')
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle

flags = tf.app.flags

flags.DEFINE_integer(
    'batch_size', 1,  # TODO
    'Training batch size.')

flags.DEFINE_integer(
    'embedding_size', 200,  # TODO
    'Word embedding size.')

flags.DEFINE_integer(
    'hidden_size', 200,  # TODO
    'RNN hidden size.')

flags.DEFINE_integer(
    'num_layers', 2,  # TODO
    'RNN layers.')

flags.DEFINE_float(
    'keep_prob', 0.8,  # TODO
    'Drop out keep probability.')

tf.flags.DEFINE_string(
    'pbtxt', 'data/graph-2016-09-10.pbtxt',
    'GraphDef proto text file used to construct model structure.')

tf.flags.DEFINE_string(
    'ckpt', 'data/ckpt-*',
    'Checkpoint directory used to fill model values.')

tf.flags.DEFINE_string(
    'vocab_file', 'data/vocab-2016-09-10.txt',
    'Vocabulary file.')

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

    s = {'fold': 3,  # 5 folds 0,1,2,3,4
         'lr': 0.1,
         'verbose': 0,
         'nhidden': 100,  # number of hidden units
         'seed': 345,
         'emb_dimension': 100,  # dimension of word embedding
         'nepochs': 50}

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

    # instantiate the model
    np.random.seed(s['seed'])
    random.seed(s['seed'])

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # TODO
        inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None])  # TODO batch_size
        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, None, nclasses])

        with tf.name_scope('Train'):
            # with tf.name_scope('LM'):
            #     # vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)
            #
            #     sys.stderr.write('Recovering graph.\n')
            #     with tf.gfile.FastGFile(FLAGS.pbtxt, 'r') as f:
            #         s = f.read()  # .decode()
            #         gd = tf.GraphDef()
            #         text_format.Merge(s, gd)
            #     tf.logging.info('Recovering Graph %s', FLAGS.pbtxt)
            #     LM = {}
            #     [LM['states_init'], LM['lstm/lstm_0/control_dependency'],
            #      LM['lstm/lstm_1/control_dependency'], LM['softmax_out'], LM['class_ids_out'],
            #      LM['class_weights_out'], LM['log_perplexity_out'], LM['inputs_in'],
            #      LM['targets_in'], LM['target_weights_in'], LM['char_inputs_in'],
            #      LM['all_embs'], LM['softmax_weights'], LM['global_step']
            #      ] = tf.import_graph_def(gd, {}, ['states_init',
            #                                       'lstm/lstm_0/control_dependency:0',
            #                                       'lstm/lstm_1/control_dependency:0',
            #                                       'softmax_out:0',
            #                                       'class_ids_out:0',
            #                                       'class_weights_out:0',
            #                                       'log_perplexity_out:0',
            #                                       'inputs_in:0',
            #                                       'targets_in:0',
            #                                       'target_weights_in:0',
            #                                       'char_inputs_in:0',
            #                                       'all_embs_out:0',
            #                                       'Reshape_3:0',
            #                                       'global_step:0'], name='')

            with tf.variable_scope('Model', reuse=None):
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable("word_embedding", [vocsize, FLAGS.embedding_size], dtype=tf.float32)
                word_embedding = tf.nn.embedding_lookup(embedding, inputs, name='word_embedding')

                # TODO st_embedding_char = CNN(one-hot(sentences))
                # TODO st_embedding_word = GloVe(one-hot(sentences))
                # lm_embedding = LM['lstm/lstm_1/control_dependency']
                # embeddings = tf.concat([word_embedding, lm_embedding], 2)  # TODO

                with tf.variable_scope('RNN'):
                    # Add a gru_cell
                    gru_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
                    #if is_training and FLAGS.keep_prob < 1:
                    #    gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=FLAGS.keep_prob)
                    cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * FLAGS.num_layers, state_is_tuple=True)
                    initial_state = cell.zero_state(FLAGS.batch_size, tf.float32)

                    # if is_training and FLAGS.keep_prob < 1:
                    embeddings = tf.nn.dropout(word_embedding, FLAGS.keep_prob)
                    # sequence_length = tf.reshape(lengths, [-1])
                    (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
                    output = tf.reshape(outputs, [-1, FLAGS.hidden_size])
                    weights = tf.get_variable("weights", [FLAGS.hidden_size, nclasses], dtype=tf.float32)
                    biases = tf.get_variable("biases", [nclasses], dtype=tf.float32)
                    logits = tf.add(tf.matmul(output, weights), biases, name="logits")
                    logits = tf.reshape(logits, [FLAGS.batch_size, -1, nclasses])

                    # prediction = tf.nn.softmax(logits, name='prediction')
                    #		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1), name='result')

                    #if is_training:
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cost = tf.reduce_mean(cross_entropy, name='cost')
                    tf.summary.scalar('cost', cost)
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost, name='train_op')


        # with tf.name_scope('Test'):
        #     with tf.variable_scope('Model', reuse=None):
        #         _rnn_model(is_training=True, embeddings=train_embeddings, labels=train_labels, lengths=train_lengths)

        # Initialize all variables in the model

        for n in tf.get_default_graph().as_graph_def().node:
            print (n.name)
        # word_embedding_tensor = sess.graph.get_tensor_by_name('Train/Model/word_embedding:0')
        # print(word_embedding_tensor.get_shape())
        # print(sess.run(tf.shape(word_embedding_tensor)))
        train_op = sess.graph.get_operation_by_name('Train/Model/RNN/train_op')

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
            shuffle([train_lex, train_ne, train_y], s['seed'])
            s['ce'] = e
            tic = time.time()
            step = 0
            for i in range(nsentences):
                X = np.asarray([train_lex[i]])
                Y = to_categorical(np.asarray(train_y[i])[:, np.newaxis], nclasses)[np.newaxis, :, :]
                if X.shape[1] == 1:
                    continue  # bug with X, Y of len 1
                # print(X.shape)
                # print(Y.shape)

                [_] = sess.run([train_op], feed_dict={inputs: X, labels: Y})
                step += 1
        #
        # if s['verbose']:
        #     print('[learning] epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / nsentences),
        #         'completed in %.2f (sec) <<\r' % (time.time() - tic))
        #     sys.stdout.flush()

    #     # evaluation // back into the real world : idx -> words
    #         try:
    #             step = 0
    #             total_case_num = 0
    #             right_num = 0
    #             while not coord.should_stop():
    #                 [predict_y, test_y] = sess.run([test_pred_tensor, test_label_tensor])
    #                 #					total_case_num += FLAGS.batch_size
    #                 #					right_num += np.sum(result)
    #                 if step == 0:
    #                     predict_Y = np.argmax(predict_y, axis=1)
    #                     test_Y = np.argmax(test_y, axis=1)
    #                 else:
    #                     predict_Y = np.concatenate((predict_Y, np.argmax(predict_y, axis=1)), axis=0)
    #                     test_Y = np.concatenate((test_Y, np.argmax(test_y, axis=1)), axis=0)
    #                 # print('step:', step)
    #                 #					print('predict_y', predict_y)
    #                 #					print('test_y', test_y)
    #                 #					for i in test_y:
    #                 #						if np.sum(i) != 1:
    #                 #							print('test_y', i)
    #                 #					print('result:', result)
    #                 step += 1
    #
    #
    #
    #     predictions_test = [map(lambda x: idx2label[x],
    #                             model.predict_on_batch(np.asarray([x])).argmax(2)[0])
    #                         for x in test_lex]
    #     groundtruth_test = [map(lambda x: idx2label[x], y) for y in test_y]
    #     words_test = [map(lambda x: idx2word[x], w) for w in test_lex]
    #
    #     predictions_valid = [map(lambda x: idx2label[x],
    #                              model.predict_on_batch(np.asarray([x])).argmax(2)[0])
    #                          for x in valid_lex]
    #     groundtruth_valid = [map(lambda x: idx2label[x], y) for y in valid_y]
    #     words_valid = [map(lambda x: idx2word[x], w) for w in valid_lex]
    #
    #     # evaluation // compute the accuracy using conlleval.pl
    #     res_test = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
    #     res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')
    #
    #     if res_valid['f1'] > best_f1:  # TODO best valid-f1?
    #         os.makedirs('weights/', exist_ok=True)
    #         model.save_weights('weights/best_model.h5', overwrite=True)
    #         best_f1 = res_valid['f1']
    #         # if s['verbose']:
    #         print('NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' ' * 20)
    #         s['vf1'], s['vp'], s['vr'] = res_valid['f1'], res_valid['p'], res_valid['r']
    #         s['tf1'], s['tp'], s['tr'] = res_test['f1'], res_test['p'], res_test['r']
    #         s['be'] = e
    #         subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
    #         subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
    #     else:
    #         print('')
    #
    # print('BEST RESULT: epoch', e, 'valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder)