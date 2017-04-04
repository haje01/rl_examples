import random
from collections import deque

import numpy as np
import tensorflow as tf
import scipy

from network import DQN, simple_replay_train, NN
import rendering as rnd

from environment import BilliardEnv, INPUT_SIZE, OUTPUT_SIZE


NUM_BALL = 5
BALL_NAME = [
    "Cue",
    "Red"
]
BALL_COLOR = [
    (1, 1, 1),  # Cue
    (1, 0, 0),  # Red
]
BALL_POS = [
    (150, 200),  # Cue
    (550, 200),  # Red
]

MAX_EPISODE = 20000
EGREEDY_EPS = 0.1
DIS = 0.9
REPLAY_MEMORY = 20000
MINIBATCH_SIZE = 10
HIDDEN_SIZE = 50
LEARNING_STEP = 20


class TwoBallEnv(BilliardEnv):
    def __init__(self):
        super(TwoBallEnv, self).__init__(BALL_NAME, BALL_COLOR, BALL_POS)

    def _step(self, action):
        # print("  _step")
        hit_list, obs = self.viewer.shot_and_get_result(action)
        reward = 1 if len(hit_list) > 0 else 0
        if reward == 1:
            print("action {}, hit".format(action))
        done = True
        return obs, reward, done, {}

    def good_random_shot(self):
        self.viewer.store_balls()
        # print("==== good_random_shot")
        while True:
            action = np.random.randint(rnd.DIV_OF_CIRCLE)
            obs, reward, done, _ = self._step((action, 5))
            self.viewer.restore_balls()
            if reward == 1:
                # print("---- good_random_shot")
                return action

    def all_good_shots(self):
        all_shots = [0] * rnd.DIV_OF_CIRCLE
        self.viewer.store_balls()
        for a in range(rnd.DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, 5))
            self.viewer.restore_balls()
            if reward == 1:
                all_shots[a] = 1
        s = sum(all_shots)
        return np.array(all_shots) / s

    def first_good_shot(self):
        all_shots = [0] * rnd.DIV_OF_CIRCLE
        self.viewer.store_balls()
        for a in range(rnd.DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, 5))
            self.viewer.restore_balls()
            if reward == 1:
                all_shots[a] = 1
                return all_shots
        return all_shots


env = TwoBallEnv()
env.query_viewer()


def dqn_bot_play(dqn):
    env.reset()
    while True:
        env._render()
        if env.viewer.move_end():
            s = env._get_obs()
            a = np.argmax(dqn.predict(s))
            env.viewer.shot((a, 5))


def nn_test(nn, viewer):
    print("NN Test start")
    hit_cnt = 0
    s = env.reset()
    test_cnt = 1000
    for i in range(test_cnt):
        a = np.argmax(nn.predict(s))
        hit_list, s = viewer.shot_and_get_result((a, 5))
        reward = 1 if len(hit_list) > 0 else 0
        if i % 100 == 0:
            print("episode {} action {} reward {}".format(i, a, reward))
        if reward == 1:
            hit_cnt += 1
    print("Accuracy: {}".format(float(hit_cnt) / test_cnt))


def nn_bot_play(nn):
    env.reset()
    while True:
        env._render()
        if env.viewer.move_end():
            s = env._get_obs()
            s = np.random.randint(0, 255, (1, 33500))
            probs = nn.predict(s)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            a = np.argmax(probs)
            env.viewer.shot((a, 5))


def train_dqn():
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(MAX_EPISODE):
            eps = 1. / ((eidx / (MAX_EPISODE * 0.1)) + 1)
            done = False

            action = env.good_random_shot()
            #if np.random.rand(1) < eps:
                ## action = env.action_space.sample()
                #action = env.good_random_shot()
                #print("  Random shot")
            #else:
                #action = np.argmax(mainDQN.predict(state))
                ##scipy.misc.imsave('shot.png', state.reshape(rnd.OBS_HEIGHT,
                                                            ##rnd.OBS_WIDTH,
                                                            ##rnd.OBS_DEPTH))

            next_state, reward, done, _ = env.step((action, 5))
            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state

            if eidx % LEARNING_STEP == 0 and len(replay_buffer) > MINIBATCH_SIZE:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, MINIBATCH_SIZE)
                    loss, _ = simple_replay_train(mainDQN, minibatch, DIS)
                print("Episode {} EPS {} ReplayBuffer {} Loss: {}".format(eidx,
                                                                          eps,
                                                                          len(replay_buffer),
                                                                          loss))

        while True:
            res = input("Learning finished. Enter 'y' to bot play: ")
            if res == 'y':
                dqn_bot_play(mainDQN)


def train_nn():
    buf = deque()
    save_path = None

    with tf.Session() as sess:
        nn = NN(sess, INPUT_SIZE, 200, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(1):
            # y = env.all_good_shots()
            y = [env.all_good_shots()]
            summary, cross_entropy, Y, logits, train = nn.update(state, y)
            a = np.argmax(y)
            state, reward, done, _ = env.step((a, 5))
            nn.train_writer.add_summary(summary, eidx)
            if eidx % 10 == 0:
                print("Cross entropy {}, Y {}, logits {}".format(cross_entropy, Y, logits))
            if reward == 1:
                print("Episode {} Hit".format(eidx))

        save_path = nn.save('saved/2ball_nn_model.ckpt')


    tf.reset_default_graph()
    with tf.Session() as sess:
        nn = NN(sess, INPUT_SIZE, 200, OUTPUT_SIZE)
        nn.load(save_path)
        nn_test(nn, env.viewer)
        #while True:
            #res = input("Learning finished. Enter 'y' to bot play: ")
            #if res == 'y':
                #nn_bot_play(nn)


def save_minibatch(minibatch):
    for i in range(MINIBATCH_SIZE):
        state = minibatch[i][0]
        fname = "minibatch_{:03d}.png".format(i)
        scipy.misc.imsave(fname, state.reshape(rnd.OBS_HEIGHT, rnd.OBS_WIDTH,
                                               rnd.OBS_DEPTH))
        if i > 10:
            break


def test_shot():
    shot = False
    while True:
        env._render()
        if not shot:
            # env.viewer.random_shot()
            env.viewer.shot(15, 10)
            shot = True
        if env.viewer.move_end():
            print(env.viewer.hit_list)
            break


if __name__ == '__main__':
    # train_dqn()
    train_nn()
    env.viewer.save_image('result.png')
