import json

import numpy as np
import tensorflow as tf


class Network:
    def __init__(self):
        self.saver = None
        self.stype = None

    def _build_network(self):
        self.saver = tf.train.Saver()
        self._merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train')

    def _get_model_info(self):
        raise NotImplemented()

    def save(self, path, minset):
        # save model info
        with open(path + '.json', 'w') as f:
            info = self._get_model_info()
            info['model'] = self.stype
            info['minset'] = minset
            f.write(json.dumps(info))

        # save model
        save_path = self.saver.save(self.session, path)
        print("{} Model saved in file: {}".format(self.stype, save_path))
        return save_path

    def load(self, path):
        self.saver.restore(self.session, path)
        print("{} Model loaded from file: {}".format(self.stype, path))

    def update(self, state, y):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run([self._merged, self._loss, self._Y,
                                 self._logits, self._train],
                                feed_dict={self._X: x, self._Y: y})


class CNN(Network):

    def __init__(self, session, input_size, o_depth, k_size, o_width, o_height,
                 c1_fcnt, c2_fcnt, l_rate, multi_shot, output_size,
                 name="main"):
        super(CNN, self).__init__()
        self.stype = 'CNN'

        self.session = session
        self.input_size = input_size
        self.o_depth = o_depth
        self.k_size = k_size
        self.o_width = o_width
        self.o_height = o_height
        self.oq_width = int(o_width / 4.0)
        self.oq_height = int(o_height / 4.0)
        self.c1_fcnt = c1_fcnt
        self.c2_fcnt = c2_fcnt
        self.multi_shot = multi_shot
        self.output_size = output_size
        self.l_rate = l_rate
        self.net_name = name

        self._build_network(l_rate)

    def _get_model_info(self):
        data = dict(input_size=self.input_size, o_depth=self.o_depth,
                    k_size=self.k_size, o_width=self.o_width,
                    o_height=self.o_height, c1_fcnt=self.c1_fcnt,
                    c2_fcnt=self.c2_fcnt, l_rate=self.l_rate,
                    multi_shot=self.multi_shot, output_size=self.output_size)
        return data

    def _build_network(self, l_rate):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size])
            self._X_img = tf.reshape(self._X, [-1, self.o_width, self.o_height,
                                               self.o_depth])
            self._Y = tf.placeholder(tf.float32, [None, self.output_size])

            # Conv 1
            shape = [self.k_size, self.k_size, self.o_depth, self.c1_fcnt]
            init = tf.random_normal(shape=shape, stddev=0.01)
            self._W1 = tf.get_variable("W1", initializer=init)
            L1 = tf.nn.conv2d(self._X_img, self._W1, strides=[1, 1, 1, 1],
                              padding='SAME')
            L1 = tf.nn.relu(L1)
            self._L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')

            # Conv 2
            shape = [self.k_size, self.k_size, self.c1_fcnt, self.c2_fcnt]
            init = tf.random_normal(shape=shape, stddev=0.01)
            self._W2 = tf.get_variable("W2", initializer=init)
            L2 = tf.nn.conv2d(self._L1, self._W2, strides=[1, 1, 1, 1],
                              padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME')
            self._L2_flat = tf.reshape(L2, [-1, self.oq_width * self.oq_height
                                            * self.c2_fcnt])

            # Fully Connected
            shape = [self.oq_width * self.oq_height * self.c2_fcnt, self.output_size]
            init = tf.contrib.layers.xavier_initializer()
            self._W3 = tf.get_variable("W3", shape=shape, initializer=init)
            self._bias = tf.get_variable("b", initializer=
                                         tf.random_normal([self.output_size]))
            self._logits = tf.matmul(self._L2_flat, self._W3) + self._bias
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self._logits,
                                                        labels=self._Y))
            tf.summary.scalar('loss', self._loss)
            self._train = tf.train.AdamOptimizer(learning_rate=self.l_rate)\
                .minimize(self._loss)

        super(CNN, self)._build_network()

    def update(self, state, y):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run([self._merged, self._loss, self._L1,
                                 self._L2_flat, self._Y, self._logits,
                                 self._train],
                                feed_dict={self._X: x, self._Y: y})

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._logits, feed_dict={self._X: x})


class FullyConnected(Network):

    def __init__(self, session, input_size, h_size, l_rate, multi_shot,
                 output_size, name="main"):
        super(FullyConnected, self).__init__()
        self.stype = 'FC'

        self.session = session
        self.input_size = input_size
        self.h_size = h_size
        self.output_size = output_size
        self.l_rate = l_rate
        self.multi_shot = multi_shot
        self.net_name = name

        self._build_network(l_rate)

    def _get_model_info(self):
        data = dict(input_size=self.input_size, h_size=self.h_size,
                    l_rate=self.l_rate, multi_shot=self.multi_shot,
                    output_size=self.output_size)
        return data

    def _build_network(self, l_rate):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [1, self.input_size],
                                     name="input_x")
            shape = [self.input_size, self.h_size]
            # initial = tf.truncated_normal(shape, stddev=0.1)
            initial = tf.contrib.layers.xavier_initializer()
            self.W1 = tf.get_variable("W1", shape=shape, initializer=initial)
            initial = tf.constant(0.1, shape=[self.h_size])
            self.bias1 = tf.get_variable("b1", initializer=initial)
            self.layer1 = tf.nn.relu(tf.matmul(self._X, self.W1) + self.bias1)

            shape = [self.h_size, self.h_size]
            initial = tf.contrib.layers.xavier_initializer()
            self.W2 = tf.get_variable("W2", shape=shape, initializer=initial)
            initial = tf.constant(0.1, shape=[self.h_size])
            self.bias2 = tf.get_variable("b2", initializer=initial)
            self.layer2 = tf.nn.relu(tf.matmul(self._X, self.W1) + self.bias2)

            shape = [self.h_size, self.output_size]
            # initial = tf.truncated_normal(shape, stddev=0.1)
            initial = tf.contrib.layers.xavier_initializer()
            self.W3 = tf.get_variable("W3", shape=shape, initializer=initial)
            initial = tf.constant(0.1, shape=[self.output_size])
            bias3 = tf.get_variable("b3", initializer=initial)
            self._logits = tf.matmul(self.layer2, self.W3) + bias3
            if self.multi_shot:
                self._logits = tf.nn.sigmoid(self._logits)

        self._Y = tf.placeholder(shape=[1, self.output_size], dtype=tf.float32)
        if not self.multi_shot:
            self._loss = tf.reduce_mean(tf.square(self._Y - self._logits))
        else:
            self._loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._Y,
                                                        logits=self._logits))
        tf.summary.scalar('loss', self._loss)
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).\
            minimize(self._loss)

        correct_prediction = tf.equal(tf.argmax(self._Y, 1),
                                      tf.argmax(self._logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        super(FullyConnected, self)._build_network()

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._logits, feed_dict={self._X: x})


class DQN:

    def __init__(self, session, input_size, h_size, l_rate, output_size,
                 name="main"):
        super(DQN, self).__init__()

        self.session = session
        self.input_size = input_size
        self.h_size = h_size
        self.l_rate = l_rate
        self.output_size = output_size
        self.net_name = name

        self._build_network(l_rate)

    def _get_model_info(self):
        data = dict(input_size=self.input_size, h_size=self.h_size,
                    l_rate=self.l_rate, output_size=self.output_size)
        return data

    def _build_network(self, l_rate):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size],
                                     name="input_x")
            W1 = tf.get_variable("W1",
                                 shape=[self.input_size, self.h_size],
                                 initializer=tf.contrib.layers.
                                 xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2",
                                 shape=[self.h_size, self.output_size],
                                 initializer=tf.contrib.layers.
                                 xavier_initializer())

            self._Qpred = tf.matmul(layer1, W2)

        self._Y = tf.placeholder(shape=[None, self.output_size],
                                 dtype=tf.float32)
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).\
            minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train],
                                feed_dict={self._X: x_stack, self._Y: y_stack})


def simple_replay_train(DQN, train_batch, dis):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        # check Q dimension
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return DQN.update(x_stack, y_stack)
