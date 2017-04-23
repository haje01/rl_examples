import json
import time
import random
from collections import deque

import click
import numpy as np
import scipy
import tensorflow as tf

from network import DQN, simple_replay_train, FullyConnected, CNN
import rendering as rnd

from environment import BilliardEnv, DIV_OF_CIRCLE

WIN_WIDTH = 740
WIN_HEIGHT = 400

DIV_OF_FORCE = 10
MAX_VEL = 1400

BALL_NAME = [
    "Cue",
    "Red"
]
BALL_COLOR = [
    (1, 1, 1),  # Cue
    (1, 0, 0),  # Red ]
]
BALL_POS = [
    (120, 202),  # Cue
    # (300, 200),  # Red
    (550, 200),  # Red
]
MINSET_BALL_POS = [
    (200, 300),  # 0
    (500, 300),  # 1
    (500, 100),  # 2
    (200, 100),  # 3
]

DEFAULT_FORCE = 2.7
SPLIT_SHOT_VAR = 360


def find_nearest(arr, value):
    idx = (np.abs(arr - value)).argmin()
    return arr[idx]


class TwoBallEnv(BilliardEnv):
    def __init__(self, enc_output):
        ball_info = list(zip(BALL_NAME, BALL_COLOR, BALL_POS))
        super(TwoBallEnv, self).__init__(ball_info, None, MAX_VEL,
                                         DIV_OF_FORCE, enc_output)

    def _step(self, action):
        # print("  _step")
        # st = time.time()
        hit_list, obs, fcnt = self.shot_and_get_result(action)
        # print("shot_and_get_result elapsed {:.2f}".format(time.time() - st))
        reward = (100.0 / fcnt) if len(hit_list) > 0 else 0
        # if reward > 0:
        #  print("action {}, fcnt {}, reward {}".format(action, fcnt, reward))
        done = True
        return obs, reward, done, {}

    def shot_and_get_result(self, action):
        self.shot(action)
        fcnt = 1
        hitted = False
        while True:
            hit, _ = self.view.frame_move()
            if hit and not hitted:
                hitted = True
            if not hitted:
                fcnt += 1
            if self.view.move_end():
                break
        self.view.render(True)
        return self.view.hit_list[:], self._get_obs(), fcnt

    def good_random_shot(self):
        self.view.store_balls()
        # print("==== good_random_shot")
        while True:
            action = np.random.randint(DIV_OF_CIRCLE)
            obs, reward, done, _ = self._step((action, DEFAULT_FORCE))
            self.view.restore_balls()
            if reward == 1:
                # print("---- good_random_shot")
                return action

    def all_good_shots(self):
        shots = [0] * DIV_OF_CIRCLE
        self.view.store_balls()
        for a in range(DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, DEFAULT_FORCE))
            self.view.restore_balls()
            if reward > 0:
                shots[a] = 1
        return np.array(shots)

    def best_shot(self):
        max_a = 0
        max_reward = 0
        self.view.store_balls()
        for a in range(DIV_OF_CIRCLE):
            obs, reward, done, _ = self._step((a, DEFAULT_FORCE))
            self.view.restore_balls()
            if reward > max_reward:
                max_a = a
                max_reward = reward
        return max_a


@click.group()
def cli():
    pass


@cli.group(help="Train model and save it.")
def train():
    pass


@cli.group(help="Test accuracy of saved model.")
def test():
    pass


@cli.group(help="Shot to test dynamics.")
def shottest():
    pass


@cli.group(help="Human play with bot.")
def play(playcnt):
    pass


@cli.command("gendata", help="Generate shot data.")
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
@click.option('--encout', 'enc_output', is_flag=True, default=False,
              show_default=True, help="Encode rgb output to number.")
def gen_data(data_path, shot_cnt, minset, visible, show_step, img_state,
             enc_output):
    env = TwoBallEnv(enc_output)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    state = env.reset()
    st = time.time()

    states = []
    ball_poss = []
    actions = []
    for pi in range(shot_cnt):
        if minset:
            state = minset_rotate(env, pi)

        states.append(state)
        if img_state:
            rnd.save_image("shots/gendata_{}.png".format(pi), state,
                           enc_output, env.obs_depth)

        ball_pos = [ball.pos for ball in env.view.balls]
        ball_poss.append(ball_pos)

        a = env.best_shot()
        if pi % show_step == 0:
            print("Episode {} action {}".format(pi, a))

        if visible:
            env.shot((a, DEFAULT_FORCE))
            while True:
                env._render()
                if env.view.move_end():
                    break
            time.sleep(0.5)
        else:
            state, reward, done, _ = env.step((a, DEFAULT_FORCE))

        actions.append(a)

    save_data(data_path, states, actions, ball_poss, env.obs_size, shot_cnt,
              minset, env.obs_depth)

    elapsed = time.time() - st
    print("Saved {} shot data '{}' in {:.2f} sec.".format(shot_cnt, data_path,
                                                          elapsed))


#@cli.command("playshot", help="Play shot data.")
#@click.argument('data_path')
#@click.option('--visible', 'visible', is_flag=True, default=False,
              #show_default=True, help="Visible shot process.")
#def play_shot_data(data_path, visible):
    #with open(data_path, 'rb') as f:
        #data = np.load(f)
        #print("shot_cnt {}, minset {}".format(data['shot_cnt'],
                                              #data['minset']))

        #shot_cnt = data['shot_cnt']
        #ball_poss = data['ball_poss']
        #actions = data['actions']
        #if visible:
            #env = TwoBallEnv()
            #env.query_viewer()
            #env.reset()
            #for i in range(shot_cnt):
                #ball_pos = ball_poss[i]
                #a = actions[i]
                #env.reset_balls(ball_pos)
                #env.shot((a, DEFAULT_FORCE))

                #while not env.view.move_end():
                    #env._render()
                #time.sleep(0.5)


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


#@test.command('nn')
#@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              #help="Hidden layer size.")
#@click.option('--test', 'test_cnt', default=1000, show_default=True,
              #help="Test episode count.")
#def test_fc(model_info_path, hidden_size, test_cnt):
    #env = TwoBallEnv()
    #env.query_viewer()

    #with tf.Session() as sess:
        #nn = FullyConnected(sess, env.obs_size, hidden_size, 1)
        #tf.global_variables_initializer().run()

        #hit_cnt = 0
        #s = env.reset()
        #for i in range(test_cnt):
            #prob = nn.predict(s)
            #a = np.random.choice(np.nonzero(prob == prob.max())[0])
            #hit_list, s, fcnt = env.shot_and_get_result((a, DEFAULT_FORCE))
            #reward = 1 if len(hit_list) > 0 else 0
            #if i % 100 == 0:
                #print("episode {} action {} reward {}".format(i, a, reward))
            #if reward == 1:
                #hit_cnt += 1
        #print("Accuracy: {}".format(float(hit_cnt) / test_cnt))


def load_model(sess, model_path):
    info_path = '{}.json'.format(model_path)
    with open(info_path, 'rt') as f:
        info = json.loads(f.read())

    model = info['model']
    if model == 'FC':
        net = FullyConnected(sess, info['input_size'], info['h_size'],
                             info['l_rate'], info['multi_shot'],
                             info['output_size'])
    elif model == 'CNN':
        net = CNN(sess, info['input_size'], info['o_depth'], info['k_size'],
                  info['o_width'], info['o_height'], info['c1_fcnt'],
                  info['c2_fcnt'], info['l_rate'], info['multi_shot'],
                  info['output_size'])

    net.load(model_path)
    return info, net


@cli.command("botplay", help="Bot play from saved model.")
@click.argument('model_path')
@click.option('--playcnt', 'play_cnt', default=100, show_default=True,
              help="Play episode count.")
@click.option('--imgstate', 'img_state', is_flag=True, default=False,
              show_default=True, help="Save state image before shot.")
@click.option('--encout', 'enc_output', is_flag=True, default=False,
              show_default=True, help="Encode rgb output to number.")
def bot_play(model_path, play_cnt, img_state, enc_output):
    env = TwoBallEnv(enc_output)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    s = env.reset()

    with tf.Session() as sess:
        info, net = load_model(sess, model_path)
        for i in range(play_cnt):
            if info['minset']:
                s = minset_rotate(env, i)
            # shot
            s = env._get_obs()
            if img_state:
                rnd.save_image("botplay_{}.png".format(i), s, enc_output,
                               env.obs_depth)

            prob = net.predict(s)
            if not net.multi_shot:
                if info['output_size'] > 1:
                    a = np.argmax(prob)
                    print("shot {}, a {}".format(i+1, prob, a))
                else:
                    fa = prob[0][0]
                    a = np.rint(fa) % DIV_OF_CIRCLE
                    print("shot {} prob {}, fa {} a {}".format(i+1, prob, fa,
                                                               a))
            else:
                pc = np.percentile(prob, 95)
                a = np.random.choice(np.nonzero(prob[0] >= pc)[0])
                a %= DIV_OF_CIRCLE
                print("shot {} prob {}, a {}".format(i+1, prob, a))

            env.shot((a, DEFAULT_FORCE))

            # wait
            time.sleep(1)
            while True:
                env._render()
                if env.view.move_end():
                    break


@train.command('dqn')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--lrate', 'l_rate', default=0.001, show_default=True,
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
@click.option('--modelpath', 'model_path', default="models/2ball_dqn",
              show_default=True, help="Model save path.")
def train_dqn(episode_size, hidden_size, l_rate, show_step, gamma, replay_size,
              replay_cnt, minibatch_size, model_path):
    st = time.time()
    env = TwoBallEnv()
    env.query_viewer()
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, env.input_size, hidden_size, l_rate, 1)
        tf.global_variables_initializer().run()

        state = env.reset()

        for eidx in range(episode_size):
            eps = 1. / ((eidx / (episode_size * 0.1)) + 1)
            done = False

            action = env.good_random_shot()

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


class Data:
    def __init__(self, data_path, samp_rate):
        info_path = '{}.json'.format(data_path)
        npy_path = '{}.npy'.format(data_path)
        with open(info_path, 'rt') as f:
            info = json.loads(f.read())
            self.shot_cnt = int(info['shot_cnt'] * samp_rate)
            self.minset = info['minset']
            self.input_size = info['input_size']
            self.o_depth = info['o_depth']

        with open(npy_path, 'rb') as f:
            data = np.load(f)
            self.states = data['states']
            self.actions = data['actions']


def save_data(data_path, states, actions, ball_poss, input_size, shot_cnt,
              minset, o_depth):
    info_path = "{}.json".format(data_path)
    with open(info_path, 'wt') as w:
        info = dict(shot_cnt=shot_cnt, minset=minset, input_size=input_size,
                    o_depth=o_depth)
        w.write(json.dumps(info))

    npy_path = '{}.npy'.format(data_path)
    with open(npy_path, 'wb') as f:
        np.savez(f, shot_cnt=shot_cnt, minset=minset, ball_poss=ball_poss,
                 states=states, actions=actions)


class Train:

    def __init__(self, env, net, data, minibatch_size, samp_rate):
        self.env = env
        self.net = net
        self.data = data
        self.minibatch_size = minibatch_size

    def _update(self, state, y, eidx, midx):
        summary, loss, Y, logits, train = self.net.update(state, y)
        return summary, loss, Y, logits, train

    def _train(self, state, y):
        summary, loss, Y, logits, train = self.net.update(state, y)
        return  summary, loss, Y, logits, train

    def run(self, sess, episode_size, stop_loss, show_step, model_path):
        st = time.time()
        tf.global_variables_initializer().run()

        shot_losses = np.empty(self.data.shot_cnt)
        shot_losses.fill(1000)
        for eidx in range(episode_size):
            selected = np.random.randint(0, self.data.shot_cnt,
                                         self.minibatch_size)
            minibatch = self.data.states[selected]
            # rnd.save_encoded_image("train_{}.png".format(eidx), states[0])

            for i, state in enumerate(minibatch):
                si = selected[i]
                y = [[self.data.actions[si]]]
                summary, loss, Y, logits, train = self._update(state, y, eidx,
                                                               i)
                # print("ep {} batch {} y {} logit {}".format(eidx, i, np.argmax(y), np.argmax(logits)))
                shot_losses[si] = loss
                # state, reward, done, _ = env.step((a, DEFAULT_FORCE))

            all_shot_pass = True
            for i in range(self.data.shot_cnt):
                if shot_losses[i] > stop_loss:
                    all_shot_pass = False
                    break

            if all_shot_pass:
                break

            if eidx % show_step == 0:
                self.net.train_writer.add_summary(summary, eidx)
                if self.net.multi_shot:
                    print("== Episode {}, loss {:.2f}, \nY {}, \nlogits {}, "
                          "action {}".format(eidx, loss, Y.reshape(12, 5),
                                             logits.reshape(12, 5), y))
                else:
                    print("== Episode {}, state {}, loss {:.2f}, \nY {}, \n"
                          "logits {}, action {}".format(eidx, si, loss, Y,
                                                        logits, y))
                # fw.write("{:2.f}\n".format(loss))

        self.net.save(model_path, minset=self.data.minset)

        # fw.close()

        print("Train finished with {} episode in {:.2f} sec".\
              format(eidx, time.time() - st))


class TrainFC(Train):
    def __init__(self, env, net, data, minibatch_size, samp_rate):
        super(TrainFC, self).__init__(env, net, data, minibatch_size, samp_rate)


class TrainCNN(Train):
    def __init__(self, env, net, data, minibatch_size, samp_rate, img_conv):
        super(TrainCNN, self).__init__(env, net, data, minibatch_size, samp_rate)
        self.img_conv = img_conv

    def _update(self, state, y, eidx, midx):
        ay = np.zeros(DIV_OF_CIRCLE)
        ay[y[0][0]] = 1
        scipy.misc.imsave('state.png', state)
        summary, loss, L1, L2, Y, logits, train = self.net.update(state, [ay])

        if self.img_conv and eidx % 100 == 0:
            for i in range(self.net.c1_fcnt):
                img = L1[0][:, :, i]
                m = img.max()
                if m > 0:
                    img = img / m
                fname = "shots/cnn-ep{}-c1_{}.png".format(eidx, i)
                scipy.misc.imsave(fname, img)

            L2 = L2[0].reshape(self.net.oq_width, self.net.oq_height,
                               self.net.c2_fcnt)
            for i in range(self.net.c2_fcnt):
                img = L2[:, :, i]
                m = img.max()
                if m > 0:
                    img = img / m
                fname = "shots/cnn-ep{}-c2_{}.png".format(eidx, i)
                shape = (int(self.env.view.obs_height * 0.25),
                         int(self.env.view.obs_width * 0.25))
                scipy.misc.imsave(fname, img.reshape(shape))

        return  summary, loss, Y, logits, train


@train.command('cnn')
@click.argument('data_path')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--minibatch', 'minibatch_size', default=20, show_default=True,
              help="Learning minibatch size.")
@click.option('--ksize', 'k_size', default=3, show_default=True, help="Kernel "
              "size.")
@click.option('--c1fcnt', 'c1_fcnt', default=32, show_default=True, help="Conv1 "
              "filter count.")
@click.option('--c2fcnt', 'c2_fcnt', default=64, show_default=True, help="Conv2 "
              "filter count.")
@click.option('--lrate', 'l_rate', default=0.001, show_default=True,
              help="Learning rate.")
@click.option('--samprate', 'samp_rate', default=1.0, show_default=True,
              help="Sampling rate of shot data.")
@click.option('--multishot', 'multi_shot', is_flag=True,
              help="Enable multi shot learning.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--imgconv', 'img_conv', is_flag=True, default=False,
              show_default=True, help="Save convolution result as image.")
@click.option('--modelpath', 'model_path', default="models/2ball_cnn",
              show_default=True, help="Model save path.")
@click.option('--stoploss', 'stop_loss', default=0.1, show_default=True,
              help="Stop train if all shot losses are below this value.")
def train_cnn(data_path, episode_size, minibatch_size, k_size, c1_fcnt,
              c2_fcnt,l_rate, samp_rate, multi_shot, show_step, img_conv,
              model_path, stop_loss):

    env = TwoBallEnv(False)
    view = env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    env.reset()

    with tf.Session() as sess:
        data = Data(data_path, samp_rate)
        output_size = DIV_OF_CIRCLE
        cnn = CNN(sess, data.input_size, data.o_depth, k_size, view.obs_width,
                  view.obs_height, c1_fcnt, c2_fcnt, l_rate, multi_shot,
                  output_size)
        train = TrainCNN(env, cnn, data, minibatch_size, samp_rate, img_conv)
        train.run(sess, episode_size, stop_loss, show_step, model_path)


@train.command('fc')
@click.argument('data_path')
@click.option('--episode', 'episode_size', default=10000, show_default=True,
              help="Learning episode number.")
@click.option('--minibatch', 'minibatch_size', default=20, show_default=True,
              help="Learning minibatch size.")
@click.option('--hidden', 'hidden_size', default=100, show_default=True,
              help="Hidden layer size.")
@click.option('--lrate', 'l_rate', default=0.001, show_default=True,
              help="Learning rate.")
@click.option('--samprate', 'samp_rate', default=1.0, show_default=True,
              help="Sampling rate of shot data.")
@click.option('--multishot', 'multi_shot', is_flag=True,
              help="Enable multi shot learning.")
@click.option('--showstep', 'show_step', default=20, show_default=True,
              help="Step for display learning status.")
@click.option('--imgstate', 'img_state', is_flag=True, default=False,
              show_default=True, help="Save state as image before shot.")
@click.option('--modelpath', 'model_path', default="models/2ball_fc",
              show_default=True, help="Model save path.")
@click.option('--stoploss', 'stop_loss', default=0.1, show_default=True,
              help="Stop train if all shot losses are below this value.")
def train_fc(data_path, episode_size, minibatch_size, hidden_size, l_rate,
             samp_rate, multi_shot, show_step, img_state, model_path,
             stop_loss):

    env = TwoBallEnv(True)
    env.query_viewer()

    with tf.Session() as sess:
        data = Data(data_path, samp_rate)
        output_size = DIV_OF_CIRCLE if multi_shot else 1
        fc = FullyConnected(sess, env.obs_size, hidden_size, l_rate, multi_shot,
                            output_size)
        train = TrainFC(env, fc, data, minibatch_size, samp_rate)
        train.run(sess, episode_size, stop_loss, show_step, model_path)


@shottest.command('1')
def test_shot_1():
    env = TwoBallEnv()
    view = env.query_viewer()
    shot = False
    env.reset_balls([(200, 300), (500, 300)])

    while True:
        env._render()
        if not shot:
            env.shot((12, DEFAULT_FORCE))
            shot = True
        if view.move_end():
            print(view.hit_list)
            break


if __name__ == '__main__':
    st = time.time()
    cli(obj={})
