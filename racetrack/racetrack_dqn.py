"""Racetracke with DQN."""
import os
import math
import random

import numpy as np
import tensorflow as tf

from racetrack_env import RacetrackEnv, Map, REWARD_SUCCESS

MAX_STEP = 10

vel_info = (
    0, 3,  # vx min/max
    -2, 2   # vy min/max
)


class ReplayMemory(object):
    """Play memory for trainning."""

    def __init__(self, n_state, max_mem, discount):
        """initialize."""
        self.max_mem = max_mem
        self.n_state = n_state
        self.discount = discount

        self.input_states = np.empty((max_mem, n_state), dtype=np.float32)
        self.actions = np.zeros(max_mem, dtype=np.uint8)
        self.next_states = np.empty((max_mem, n_state), dtype=np.float32)
        self.gameovers = np.empty(max_mem, dtype=np.bool)
        self.rewards = np.empty(self.max_mem, dtype=np.int8)

        self.count = 0
        self.cur = 0

    def remember(self, cur_state, action, reward, next_state, gameover):
        """Remember this state."""
        self.actions[self.cur] = action
        self.rewards[self.cur] = reward
        self.input_states[self.cur] = cur_state
        self.next_states[self.cur] = next_state
        self.gameovers[self.cur] = gameover
        self.count = max(self.count, self.cur + 1)
        self.cur += 1
        self.cur %= self.max_mem

    def get_batch(self, model, n_batch, n_action, sess, X):  # noqa
        mem_len = self.count
        batch_size = min(n_batch, mem_len)

        inputs = np.zeros((batch_size, self.n_state))
        targets = np.zeros((batch_size, n_action))

        for i in range(batch_size):
            max_idx = 2 if mem_len == 1 else mem_len
            idx = random.randrange(1, max_idx)
            cur_state = np.reshape(self.input_states[idx], (1, self.n_state))
            target = sess.run(model, feed_dict={X: cur_state})

            next_state = np.reshape(self.next_states[idx], (1, self.n_state))
            cur_output = sess.run(model, feed_dict={X: next_state})
            next_stateQ = np.amax(cur_output)  # noqa

            if self.gameovers[idx]:
                target[0, [self.actions[idx]]] = self.rewards[idx]
            else:
                target[0, [self.actions[idx]]] = self.rewards[idx] + \
                    self.discount * next_stateQ

            inputs[i] = cur_state
            targets[i] = target

        return inputs, targets


def randf(s, e):
    """Generate random float between tow boundaries."""
    return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s


def main(_):
    """Training new model."""
    print("Training new model")

    # Init map, environment and consts
    with open('racetrack_map_1.txt', 'r') as f:
        amap = Map(f.read(), v_mgn=2, h_mgn=2)
    env = RacetrackEnv(amap, vel_info, MAX_STEP)

    n_action = env.action_space.n
    epoch = 100000
    n_hidden = 100
    max_mem = 500
    n_batch = 50
    n_state = env.total_states()
    discount = 0.9
    lr = 0.2
    eps = 1.0
    min_eps = 0.001

    memory = ReplayMemory(n_state, max_mem, discount)

    # Create base model
    X = tf.placeholder(tf.float32, [None, n_state])  # NOQA

    stddev = 1.0 / math.sqrt(n_state)
    W1 = tf.Variable(tf.truncated_normal([n_state, n_hidden], stddev=stddev))  # NOQA
    b1 = tf.Variable(tf.truncated_normal([n_hidden], stddev=stddev))
    input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=stddev)) # NOQA
    b2 = tf.Variable(tf.truncated_normal([n_hidden], stddev=stddev))
    hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)

    W3 = tf.Variable(tf.truncated_normal([n_hidden, n_action], stddev=stddev)) # NOQA
    b3 = tf.Variable(tf.truncated_normal([n_action], stddev=stddev))
    output_layer = tf.nn.relu(tf.matmul(hidden_layer, W3) + b3)

    # True labels
    Y = tf.placeholder(tf.float32, [None, n_action])  # NOQA

    # Mean squared error cost function
    cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * n_batch)

    # Stochastic Gradient Decent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    saver = tf.train.Saver()

    # Train
    n_win = 0
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(epoch):
            # show = True if i % 100 == 0 else False
            show = True
            err = 0
            cur_state = env.reset()
            gameover = False

            while not gameover:
                if randf(0, 1) <= eps:
                    action = random.randrange(n_action)
                    # print("Random action: {}".format(action))
                else:
                    state = np.reshape(np.array(cur_state), (1, n_state))
                    q = sess.run(output_layer, feed_dict={X: state})
                    action = q.argmax()

                if eps > min_eps:
                    eps *= 0.999

                next_state, reward, gameover, _ = env.step(action)
                if show:
                    print(cur_state, next_state, gameover, eps)
                    env._draw(next_state)
                if reward == REWARD_SUCCESS:
                    n_win += 1

                memory.remember(cur_state, action, reward, next_state,
                                gameover)
                cur_state = next_state
                inputs, targets = memory.get_batch(output_layer, n_batch,
                                                   n_action, sess, X)
                _, loss = sess.run([optimizer, cost], feed_dict={X: inputs,
                                   Y: targets})
                err += loss

            wratio = float(n_win) / (i + 1)
            if show:
                print("Epoch {}: error {}, win {}, win ratio {}".
                      format(i, err, n_win, wratio))

        save_path = saver.save(sess, os.path.join(os.getcwd(), 'racemodel.tf'))
        print("Model saved in '{}'".format(save_path))


tf.app.run()
