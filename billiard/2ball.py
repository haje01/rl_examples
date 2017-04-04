import time
import random
from collections import deque

import click
import numpy as np
import tensorflow as tf

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


@click.group()
def cli():
    pass


@cli.group(help="Train model and save it.")
def train():
    pass


@cli.group(help="Test accuracy of saved model.")
def test():
    pass


@cli.group(help="Bot play from saved model.")
def botplay():
    pass


@cli.group(help="Shot to test dynamics.")
def shottest():
    pass


@cli.group(help="Human play with bot.")
def play(playcnt):
    pass


@botplay.command('dqn')
@click.argument('model_path')
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--playcnt', 'play_cnt', show_default=True, default=100,
              help="Play episode count.")
def dqn_bot_play(model_path, hidden_size, play_cnt):
    env = TwoBallEnv()
    env.query_viewer()
    env.reset()

    with tf.Session() as sess:
        dqn = DQN(sess, INPUT_SIZE, hidden_size, OUTPUT_SIZE)
        dqn.load(model_path)

        for i in range(play_cnt):
            env._render()
            if env.viewer.move_end():
                s = env._get_obs()
                a = np.argmax(dqn.predict(s))
                env.viewer.shot((a, 5))


@test.command('nn')
@click.argument('model_path')
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--test', 'test_cnt', default=1000, show_default=True,
              help="Test episode count.")
def test_nn(model_path, hidden_size, test_cnt):
    env = TwoBallEnv()
    env.query_viewer()

    with tf.Session() as sess:
        nn = NN(sess, INPUT_SIZE, hidden_size, OUTPUT_SIZE)
        nn.load(model_path)
        tf.global_variables_initializer().run()

        hit_cnt = 0
        s = env.reset()
        for i in range(test_cnt):
            a = np.argmax(nn.predict(s))
            hit_list, s = env.viewer.shot_and_get_result((a, 5))
            reward = 1 if len(hit_list) > 0 else 0
            if i % 100 == 0:
                print("episode {} action {} reward {}".format(i, a, reward))
            if reward == 1:
                hit_cnt += 1
        print("Accuracy: {}".format(float(hit_cnt) / test_cnt))


@botplay.command('nn')
@click.argument('model_path')
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--playcnt', 'play_cnt', default=100, show_default=True,
              help="Play episode count.")
def nn_bot_play(model_path, hidden_size, play_cnt):
    env = TwoBallEnv()
    env.query_viewer()
    env.reset()

    with tf.Session() as sess:
        nn = NN(sess, INPUT_SIZE, hidden_size, OUTPUT_SIZE)
        nn.load(model_path)
        for i in range(play_cnt):
            # shot
            s = env._get_obs()
            s = np.random.randint(0, 255, (1, 33500))
            probs = nn.predict(s)
            a = np.argmax(probs)
            env.viewer.shot((a, 5))

            # wait
            time.sleep(1)
            while True:
                env._render()
                if env.viewer.move_end():
                    break


@train.command('dqn')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--gamma', default=0.9, show_default=True, help='Gamma for TD.')
@click.option('--replay', 'replay_size', default=10000, show_default=True,
              help="Replay buffer size.")
@click.option('--minibatch', 'minibatch_size', default=10, show_default=True,
              help="Minibatch size.")
@click.option('--modelpath', 'model_path', default="saved/2ball_dqn_model",
              show_default=True, help="Model save path.")
def train_dqn(episode_size, hidden_size, show_step, gamma, replay_size,
              minibatch_size, model_path):
    env = TwoBallEnv()
    env.query_viewer()
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, hidden_size, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(episode_size):
            eps = 1. / ((eidx / (episode_size * 0.1)) + 1)
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
            if len(replay_buffer) > replay_size:
                replay_buffer.popleft()

            state = next_state

            if eidx % show_step == 0 and len(replay_buffer) > minibatch_size:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, minibatch_size)
                    loss, _ = simple_replay_train(mainDQN, minibatch, gamma)
                print("Episode {} EPS {} ReplayBuffer {} Loss: {}".
                      format(eidx, eps, len(replay_buffer), loss))
        mainDQN.save(model_path)


@train.command('nn')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--modelpath', 'model_path', default="saved/2ball_nn_model",
              show_default=True, help="Model save path.")
def train_nn(episode_size, hidden_size, show_step, model_path):
    env = TwoBallEnv()
    env.query_viewer()

    with tf.Session() as sess:
        nn = NN(sess, INPUT_SIZE, hidden_size, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(episode_size):
            # y = env.all_good_shots()
            y = [env.all_good_shots()]
            summary, cross_entropy, Y, logits, train = nn.update(state, y)
            a = np.argmax(y)
            state, reward, done, _ = env.step((a, 5))
            nn.train_writer.add_summary(summary, eidx)
            if eidx % 10 == 0:
                print("Cross entropy {}, Y {}, logits {}".format(cross_entropy,
                                                                 Y, logits))
            if reward == 1:
                print("Episode {} Hit".format(eidx))

        nn.save(model_path)


@shottest.command('1')
def test_shot_1():
    env = TwoBallEnv()
    env.query_viewer()
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
    # train_nn()
    # env.viewer.save_image('result.png')
    cli(obj={})
