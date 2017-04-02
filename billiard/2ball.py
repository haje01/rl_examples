import random
from collections import deque

import numpy as np
import tensorflow as tf
import scipy

from dqn import DQN, simple_replay_train
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

MAX_EPISODE = 5000
EGREEDY_EPS = 0.1
DIS = 0.9
REPLAY_MEMORY = 5000
MINIBATCH_SIZE = 10
HIDDEN_SIZE = 100


class TwoBallEnv(BilliardEnv):
    def __init__(self):
        super(TwoBallEnv, self).__init__(BALL_NAME, BALL_COLOR, BALL_POS)

    def _step(self, action):
        hit_list, obs = self.viewer.shot_and_get_result(action)
        reward = 1 if len(hit_list) > 0 else 0
        if reward == 1:
            print("  Hit")
        done = True
        return obs, reward, done, {}


env = TwoBallEnv()
env.query_viewer()


def bot_play(mainDQN):
    env.reset()
    while True:
        env._render()
        if env.viewer.move_end():
            s = env._get_obs()
            a = np.argmax(mainDQN.predict(s))
            env.viewer.shot((a, 5))


def train():
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(MAX_EPISODE):
            eps = 1. / ((eidx / (MAX_EPISODE * 0.2)) + 1)
            done = False

            if np.random.rand(1) < eps:
                action = env.action_space.sample()
                print("  Random shot")
            else:
                action = np.argmax(mainDQN.predict(state))
                #scipy.misc.imsave('shot.png', state.reshape(rnd.OBS_HEIGHT,
                                                            #rnd.OBS_WIDTH,
                                                            #rnd.OBS_DEPTH))

            next_state, reward, done, _ = env.step((action, 5))
            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state

            if eidx % 10 == 0 and len(replay_buffer) > MINIBATCH_SIZE:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, MINIBATCH_SIZE)
                    loss, _ = simple_replay_train(mainDQN, minibatch, DIS)
                print("Episode {} EPS {} ReplayBuffer {} Loss: {}".format(eidx,
                                                                          eps,
                                                                          len(replay_buffer),
                                                                          loss))

        bot_play(mainDQN)


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
    train()
    env.viewer.save_image('result.png')
