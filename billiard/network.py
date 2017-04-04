import numpy as np
import tensorflow as tf


class Network:
    def __init__(self):
        self.saver = None

    def _build_network(self):
        self.saver = tf.train.Saver()

    def save(self, path):
        assert self.saver is not None

    def save(self, path):
        save_path = self.saver.save(self.session, path)
        print("NN Model saved in file: {}".format(save_path))
        return save_path

    def load(self, path):
        self.saver.restore(self.session, path)
        print("NN Model loaded from file: {}".format(path))


class NN(Network):
    def __init__(self, session, input_size, h1_size, output_size, name="main"):
        super(NN, self).__init__()

        self.session = session
        self.input_size = input_size
        self.h1_size = h1_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, l_rate=0.1):
        # Hidden 1
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [1, self.input_size],
                                     name="input_x")
            shape = [self.input_size, self.h1_size]
            initial = tf.truncated_normal(shape, stddev=0.1)
            self.W1 = tf.get_variable("W1", initializer=initial)
            initial = tf.constant(0.1, shape=[self.h1_size])
            self.bias1 = tf.get_variable("b1", initializer=initial)
            self.layer1 = tf.nn.relu(tf.matmul(self._X, self.W1) + self.bias1)

            shape = [self.h1_size, self.output_size]
            initial = tf.truncated_normal(shape, stddev=0.1)
            self.W2 = tf.get_variable("W2", initializer=initial)
            initial = tf.constant(0.1, shape=[self.output_size])
            bias2 = tf.get_variable("b2", initializer=initial)
            # self._logits = tf.nn.softmax(tf.matmul(self.layer1, self.W2) + bias2)
            self._logits = tf.matmul(self.layer1, self.W2) + bias2

        self._Y = tf.placeholder(shape=[1, self.output_size], dtype=tf.float32)
        self._cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self._Y,
                                                    logits=self._logits))
        tf.summary.scalar('cross_entropy', self._cross_entropy)
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).\
            minimize(self._cross_entropy)

        correct_prediction = tf.equal(tf.argmax(self._Y, 1), tf.argmax(self._logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        self._merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('train')

        super(NN, self)._build_network()


    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._logits, feed_dict={self._X: x})

    def update(self, state, y):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run([self._merged, self._cross_entropy, self._Y, self._logits, self._train],
                                feed_dict={self._X: x, self._Y: y})


class DQN:
    def __init__(self, session, input_size, h_size, output_size, name="main"):
        super(DQN, self).__init__()

        self.session = session
        self.input_size = input_size
        self.h_size = h_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, l_rate=1e-1):
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
