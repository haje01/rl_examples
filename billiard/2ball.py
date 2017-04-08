import json
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
    (150, 202),  # Cue
    (550, 200),  # Red
]
MINSET_BALL_POS = [
    (200, 300),  # 0
    (500, 300),  # 1
    (500, 100),  # 2
    (200, 100),  # 3
]

DEFAULT_FORCE = 2.5


class TwoBallEnv(BilliardEnv):
    def __init__(self):
        super(TwoBallEnv, self).__init__(BALL_NAME, BALL_COLOR, BALL_POS)

    def _step(self, action):
        # print("  _step")
        # st = time.time()
        hit_list, obs, fcnt = self.viewer.shot_and_get_result(action)
        # print("shot_and_get_result elapsed {:.2f}".format(time.time() - st))
        reward = 100.0 / fcnt if len(hit_list) > 0 else 0
        # if reward > 0:
        #    print("action {}, fcnt {}".format(action, fcnt))
        done = True
        return obs, reward, done, {}

    def good_random_shot(self):
        self.viewer.store_balls()
        # print("==== good_random_shot")
        while True:
            action = np.random.randint(rnd.DIV_OF_CIRCLE)
            obs, reward, done, _ = self._step((action, DEFAULT_FORCE))
            self.viewer.restore_balls()
            if reward == 1:
                # print("---- good_random_shot")
                return action

    def all_good_shots(self):
        shots = [0] * rnd.DIV_OF_CIRCLE
        self.viewer.store_balls()
        for a in range(rnd.DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, DEFAULT_FORCE))
            self.viewer.restore_balls()
            if reward > 0:
                shots[a] = 1
        return np.array(shots)

    def best_shot(self):
        action = None
        best_reward = 0
        self.viewer.store_balls()
        for a in range(rnd.DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, DEFAULT_FORCE))
            self.viewer.restore_balls()
            # if reward > 0:
            #    print("action {} reward {}".format(reward, a))
            if reward > best_reward:
                action = a
                best_reward = reward
        return action


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


@cli.command("genshot", help="Generate shot data.")
@click.argument('data_path')
@click.option('--shotcnt', 'shot_cnt', show_default=True, default=10000,
              help="Generate shot data of this count.")
@click.option('--minset', 'minset', is_flag=True, default=False,
              show_default=True, help="Use minimal learning set for"
              "verification.")
@click.option('--visible', 'visible', is_flag=True, default=False,
              show_default=True, help="Visible shot process.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display shot status.")
@click.option('--imgstate', 'img_state', is_flag=True, default=False,
              show_default=True, help="Save state image before shot.")
def gen_shot_data(data_path, shot_cnt, minset, visible, show_step, img_state):
    env = TwoBallEnv()
    env.query_viewer()
    state = env.reset()
    st = time.time()

    with open(data_path, 'wb') as f:
        states = []
        actions = []
        for pi in range(shot_cnt):
            if minset:
                state = minset_rotate(env, pi)

            if img_state:
                rnd.save_encoded_image("genshot_{}.png".format(pi), state)

            a = env.best_shot()
            if pi % show_step == 0:
                print("Episode {} action {}".format(pi, a))

            if visible:
                env.viewer.shot((a, DEFAULT_FORCE))
                while True:
                    env._render()
                    if env.viewer.move_end():
                        break
                state = env._get_obs()
                time.sleep(0.5)
            else:
                state, reward, done, _ = env.step((a, DEFAULT_FORCE))

            states.append(state)
            actions.append(a)

        np.savez(f, shot_cnt=shot_cnt, reset_cnt=reset_cnt, minset=minset,
                 states=states, actions=actions)
        elapsed = time.time() - st
        print("Saved {} shot data '{}' in {:.2f} sec.".format(shot_cnt,
                                                              data_path,
                                                              elapsed))


@cli.command("descshot", help="Describe shot data.")
@click.argument('data_path')
@click.option('--action', 'show_action', is_flag=True, default=False,
              show_default=True, help="Show actions")
def play_shot_data(data_path, show_action):
    with open(data_path, 'rb') as f:
        data = np.load(f)
        print("shot_cnt {}, reset_cnt {}, minset "
              "{}".format(data['shot_cnt'], data['reset_cnt'],
                          data['minset']))

        if show_action:
            actions = data['actions']
            for i in range(data['shot_cnt']):
                a = actions[i]
                print("  episode {} action {}".format(i, a))


def minset_rotate(env, pi):
    if pi % 4 == 0:
        state = env.reset_balls(MINSET_BALL_POS[:2])
    elif pi % 4 == 1:
        state = env.reset_balls(MINSET_BALL_POS[1:3])
    elif pi % 4 == 2:
        state = env.reset_balls(MINSET_BALL_POS[2:])
    else:
        state = env.reset_balls([MINSET_BALL_POS[3], MINSET_BALL_POS[0]])
    return state


@botplay.command('dqn')
@click.argument('model_path')
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--playcnt', 'play_cnt', show_default=True, default=100,
              help="Play episode count.")
def dqn_bot_play(model_path, hidden_size, l_rate, play_cnt):
    env = TwoBallEnv()
    env.query_viewer()
    env.reset()

    with tf.Session() as sess:
        dqn = DQN.create_from_model_info(sess, model_path + '.json')

        for i in range(play_cnt):
            env._render()
            if env.viewer.move_end():
                s = env._get_obs()
                a = np.argmax(dqn.predict(s))
                env.viewer.shot((a, DEFAULT_FORCE))


@test.command('nn')
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--test', 'test_cnt', default=1000, show_default=True,
              help="Test episode count.")
def test_nn(model_info_path, hidden_size, test_cnt):
    env = TwoBallEnv()
    env.query_viewer()

    with tf.Session() as sess:
        nn = NN(sess, env.input_size, hidden_size, OUTPUT_SIZE)
        tf.global_variables_initializer().run()

        hit_cnt = 0
        s = env.reset()
        for i in range(test_cnt):
            prob = nn.predict(s)
            a = np.random.choice(np.nonzero(prob == prob.max())[0])
            hit_list, s, fcnt = env.viewer.shot_and_get_result((a,
                                                                DEFAULT_FORCE))
            reward = 1 if len(hit_list) > 0 else 0
            if i % 100 == 0:
                print("episode {} action {} reward {}".format(i, a, reward))
            if reward == 1:
                hit_cnt += 1
        print("Accuracy: {}".format(float(hit_cnt) / test_cnt))


@botplay.command('nn')
@click.argument('model_path')
@click.option('--playcnt', 'play_cnt', default=100, show_default=True,
              help="Play episode count.")
@click.option('--imgstate', 'img_state', is_flag=True, default=False,
              show_default=True, help="Save state image before shot.")
def nn_bot_play(model_path, play_cnt, img_state):
    env = TwoBallEnv()
    env.query_viewer()
    s = env.reset()

    with tf.Session() as sess:
        nn = NN.create_from_model_info(sess, model_path + '.json')
        for i in range(play_cnt):
            if nn.minset:
                s = minset_rotate(env, i)
            # shot
            s = env._get_obs()
            if img_state:
                rnd.save_encoded_image("botplay_{}.png".format(i), s)

            prob = nn.predict(s)
            if not nn.multi_shot:
                fa = prob[0][0]
                a = np.rint(fa)
                print("prob {}, fa {} a {}".format(prob, fa, a))
            else:
                pc = np.percentile(prob, 95)
                a = np.random.choice(np.nonzero(prob[0] >= pc)[0])
                a %= rnd.DIV_OF_CIRCLE
                print("prob {}, a {}".format(prob, a))

            env.viewer.shot((a, DEFAULT_FORCE))

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
@click.option('--lrate', 'l_rate', default=0.1, show_default=True,
              help="Learning rate.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--gamma', default=0.9, show_default=True, help='Gamma for TD.')
@click.option('--repbuf', 'replay_size', default=10000, show_default=True,
              help="Replay buffer size.")
@click.option('--repcnt', 'replay_cnt', default=50, show_default=True,
              help="Model save path.")
@click.option('--minibatch', 'minibatch_size', default=10, show_default=True,
              help="Minibatch size.")
@click.option('--modelpath', 'model_path', default="saved/2ball_dqn_model",
              show_default=True, help="Model save path.")
def train_dqn(episode_size, hidden_size, l_rate, show_step, gamma, replay_size,
              replay_cnt, minibatch_size, model_path):
    st = time.time()
    env = TwoBallEnv()
    env.query_viewer()
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, env.input_size, hidden_size, l_rate, OUTPUT_SIZE)
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
                # action = np.random.choice(np.nonzero(prob == prob.max())[0])
                ##scipy.misc.imsave('shot.png', state.reshape(rnd.OBS_HEIGHT,
                                                            ##rnd.OBS_WIDTH,
                                                            ##rnd.OBS_DEPTH))

            next_state, reward, done, _ = env.step((action, DEFAULT_FORCE))
            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > replay_size:
                replay_buffer.popleft()

            state = next_state

            if eidx % show_step == 0 and len(replay_buffer) > minibatch_size:
                for _ in range(replay_cnt):
                    minibatch = random.sample(replay_buffer, minibatch_size)
                    loss, _ = simple_replay_train(mainDQN, minibatch, gamma)
                print("Episode {} EPS {} ReplayBuffer {} Loss: {}".
                      format(eidx, eps, len(replay_buffer), loss))
        mainDQN.save(model_path)

    print("Train finished in {:.2f} sec".format(time.time() - st))


@train.command('nn')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--minset', 'minset', is_flag=True, default=False,
              show_default=True, help="Use minimal learning set for"
              "verification.")
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--lrate', 'l_rate', default=0.1, show_default=True,
              help="Learning rate.")
@click.option('--multishot', 'multi_shot', is_flag=True,
              help="Enable multi shot learning.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--imgstate', 'img_state', is_flag=True, default=False,
              show_default=True, help="Save state as image before shot.")
@click.option('--modelpath', 'model_path', default="saved/2ball_nn_model",
              show_default=True, help="Model save path.")
def train_nn(episode_size, minset, hidden_size, l_rate, multi_shot, show_step,
             img_state, model_path):
    env = TwoBallEnv()
    env.query_viewer()
    st = time.time()

    # fw = open('progress.txt', 'w')
    with tf.Session() as sess:
        output_size = rnd.DIV_OF_CIRCLE if multi_shot else 1
        nn = NN(sess, INPUT_SIZE, minset, hidden_size, l_rate, multi_shot,
                output_size)
        tf.global_variables_initializer().run()

        state = env.reset()
        for eidx in range(episode_size):
            if minset:
                state = minset_rotate(env, eidx)
            if img_state:
                rnd.save_encoded_image('train_{}.png'.format(eidx), state)

            if not multi_shot:
                y = [[env.best_shot()]]
                a = y[0][0]
            else:
                y = env.all_good_shots()
                a = np.random.choice(np.nonzero(y == y.max())[0])
                y = [env.all_good_shots()]

            summary, loss, Y, logits, train = nn.update(state, y)
            state, reward, done, _ = env.step((a, DEFAULT_FORCE))
            nn.train_writer.add_summary(summary, eidx)
            if eidx % show_step == 0:
                if nn.multi_shot:
                    print("Episode {}, loss {:.2f}, \nY {}, \nlogits {}, "
                          "action {}".format(eidx, loss, Y.reshape(12, 5),
                                             logits.reshape(12, 5), a))
                else:
                    print("Episode {}, loss {:.2f}, \nY {}, \nlogits {}, "
                          "action {}".format(eidx, loss, Y, logits, a))
                # fw.write("{:2.f}\n".format(loss))

            # if reward == 1:
            #    print("Episode {} Hit".format(eidx))
            if eidx % 100 == 0:
                nn.save(model_path)

        nn.save(model_path)

    # fw.close()

    print("Train finished in {:.2f} sec".format(time.time() - st))


@shottest.command('1')
def test_shot_1():
    env = TwoBallEnv()
    env.query_viewer()
    shot = False

    while True:
        env._render()
        if not shot:
            # env.viewer.random_shot()
            env.viewer.shot((11, DEFAULT_FORCE))
            shot = True
        if env.viewer.move_end():
            print(env.viewer.hit_list)
            break


if __name__ == '__main__':
    st = time.time()
    cli(obj={})
