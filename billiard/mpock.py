import time
import math
from collections import deque, defaultdict
from enum import Enum

import click
import numpy as np
from tqdm import tqdm

import rendering as rnd
from environment import BilliardEnv, calc_angle2, DIV_OF_CIRCLE

WIN_WIDTH = 300
WIN_HEIGHT = 300
DIV_OF_FORCE = 5
MAX_VEL = 600
GAMEOVER_TIME = 3

NUM_BALL = 3
BALL_NAME = [
    "Cue",
    "Red",
    "Blue",
]
BALL_COLOR = [
    (1, 1, 1),  # Cue
    (1, 0, 0),  # Red
    (0, 0, 1),  # Blue
]
BALL_POS = [
    (65, 65),  # Cue
    (80, 240),  # Red
    (240, 80),  # Blue
]
HOLES = [
    [270, 270]
]


class NodeT(Enum):
    Root = 0
    Shot = 1
    Freeball = 2


DEFAULT_FORCE = 2
GOOD_ANG = 1.0
GOOD_DIST = 35
EXPLORE_RATE = 0.5

ACTION_CNT = DIV_OF_CIRCLE * DIV_OF_FORCE
ALL_ACTION = []
for d in range(DIV_OF_CIRCLE):
    for f in range(DIV_OF_FORCE):
        ALL_ACTION.append((d, f+1))

BUDGET = 700
DEPTH = 3
EXPAND_RATE = 0.5
SHOT_EXPAND = int(ACTION_CNT * EXPAND_RATE)
FB_EXPAND = 316
MIN_REWARD_RATE = -1000


@click.group()
def cli():
    pass


class Player:
    def __init__(self, tball):
        self.tball = tball


def flip_side(view, side):
    eside = side
    if side == view.balls[1]:
        side = view.balls[2]
    else:
        side = view.balls[1]
    return side, eside


class MiniPocketEnv(BilliardEnv):
    def __init__(self, enc_output):
        ball_info = list(zip(BALL_NAME, BALL_COLOR, BALL_POS))
        super(MiniPocketEnv, self).__init__(ball_info, HOLES, MAX_VEL,
                                            DIV_OF_FORCE, enc_output)
        self.start_side = None
        self._clear()

    def query_viewer(self, width, height):
        super(MiniPocketEnv, self).query_viewer(width, height)
        if self.side is None:
            self.side = self.view.balls[1]
            self.eside = self.view.balls[2]

    def _reset(self):
        super(MiniPocketEnv, self)._reset()
        self._clear(self.start_side)

    def get_other_side(self, side):
        return [b for b in self.view.balls[1:] if b != side][0]

    def _clear(self, side=None):
        if side is not None:
            self.side = side
            self.eside = self.get_other_side(side)
        else:
            self.side = self.eside = None
        self.winner = self.gover_time = None

    def is_freeball(self):
        return self.view.is_freeball()

    def set_winner(self, winner):
        self.winner = winner
        self.gover_time = time.time()

    def _render(self, mode='human', close=False):
        super(MiniPocketEnv, self)._render(mode, close)
        if self.gover_time is not None:
            if mode != 'human':
                self.reset()
            elif time.time() - self.gover_time > GAMEOVER_TIME:
                self.reset()

    def flip_side(self):
        self.side, self.eside = flip_side(self.view, self.side)

    def _step(self, action):
        obs, reward = self.shot_and_get_result(action)
        print("reward {}".format(reward))
        return obs, reward, True, {}

    def shot_and_get_result(self, action, side):
        self.shot(action, False)
        while True:
            self.view.frame_move()
            if self.view.move_end():
                break

        self.view.render(True)
        return self._get_obs(), self._decide_reward(side)

    def _in_good_pos(self, side):
        cpos = self.view.cueball.pos
        hpos = HOLES[0]
        bpos = side.pos
        bhvec = (hpos[0] - bpos[0], hpos[1] - bpos[1])
        cbvec = (bpos[0] - cpos[0], bpos[1] - cpos[1])
        cbdist = np.linalg.norm(cbvec)
        ang = calc_angle2(bhvec, cbvec)
        # print("cpos {} hpos {} bpos {}".format(cpos, hpos, bpos))
        # print("bhvec {} cbvec {} ang {}".format(bhvec, cbvec, ang))
        # print("cbdist {}".format(cbdist))
        return ang < GOOD_ANG and cbdist > GOOD_DIST

    def target_near(self, target):
        cpos = self.view.cueball.pos
        tpos = target.pos
        vec = (cpos[0] - tpos[0], cpos[1] - tpos[1])
        dist = np.linalg.norm(vec)
        return dist <= rnd.BALL_RAD * 2

    def _decide_reward(self, side, show_info=False):
        reward = 0
        # lost
        if self.eside in self.view.holein_list:
            reward = -2
        # enemy's free ball
        elif self.view.cueball in self.view.holein_list or\
                len(self.view.hit_list) == 0 or\
                side != self.view.hit_list[0]:
            if show_info:
                ch = self.view.cueball in self.view.holein_list
                nh = len(self.view.hit_list) == 0
                wh = side != self.view.hit_list[0] if not nh else None
                print("FB: cueball holein {}, no hit {}, wrong ball hit {}".format(ch, nh, wh))
            reward = -1
        # win
        elif side in self.view.holein_list:
            reward = (DIV_OF_FORCE + 5)/ float(math.sqrt(self.last_force) +
                                               len(self.view.hit_list))
        # too near target
        elif self.target_near(side):
            reward = -1.0
        # good position
        # elif self._in_good_pos(side):
            # reward = 1
        return reward

    def _decide_fb_reward(self, side):
        reward = 0
        if self._in_good_pos(side):
            reward = 1
        return reward

    def build_freeballQ(self):
        poss = []
        ox, oy = self.view.cueball.pos
        off = rnd.WALL_DEPTH + rnd.BALL_RAD
        for x in range(off, WIN_WIDTH - off, 4):
            for y in range(off, WIN_HEIGHT - off, 4):
                self.view.cueball.pos = x, y
                if self.view.is_valid_fb():
                    poss.append((x, y))
        self.view.cueball.pos = (ox, oy)
        np.random.shuffle(poss)
        return deque(poss)

    def _get_hud(self):
        if self.winner is not None:
            msg = "{}'s Win!".format(self.winner)
            return msg, 110, 150
        else:
            msg = "{}'s turn".format(self.side.name)
            if self.drag_shot_info is not None:
                _, _, force = self.drag_shot_info
                msg += " F: {:.1f}".format(force/self.div_of_force)
            elif self.is_freeball():
                msg += " [Free Ball]"
            return msg, 10, 5

    def process_result(self, show_info=False):
        reward = self._decide_reward(self.side, show_info)
        print("reward {}".format(reward))
        if reward >= 1.0:
            self.set_winner(self.side.name)
            self.start_side = self.get_other_side(self.side)
        elif reward == -2:
            self.set_winner(self.eside.name)
            self.start_side = self.side
        else:
            self.flip_side()
            if reward == -1:
                self.view.start_freeball()


@cli.command('shottest', help="Play specified shots to test.")
def shottest():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    # actionQ =  deque([(0, 2), (23, 3)])
    shot = False
    while True:
        env._render()
        if env.view.move_end():
            if shot:
                env.process_result()
            if len(actionQ) > 0:
                env.shot(actionQ.popleft())
                shot = True
            else:
                break


def shuffle_actions():
    actions = ALL_ACTION[:]
    np.random.shuffle(actions)
    return deque(actions)


class Node:
    def __init__(self, ntype, action, parent=None, side=None):
        self.visits = 0
        self.ntype = ntype
        self.reward = 0.0
        self.action = action
        self.parent = parent
        self.side = side
        self.children = []
        self.actionQ = shuffle_actions()

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    @property
    def depth(self):
        d = 0 if self.ntype == NodeT.Freeball else 1
        if self.parent is not None:
            d += self.parent.depth
        return d

    @property
    def reward_rate(self):
        if self.visits > 0:
            return self.reward / float(self.visits)
        return MIN_REWARD_RATE

    def __repr__(self):
        return "Node({},{}): children {} visits {} action {} reward_rate {:.2f}".\
            format(self.side, self.ntype, len(self.children), self.visits,
                   self.action, self.reward_rate)


def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent


def calc_dpr_rate(tp_rwd, dp_rwd):
    if tp_rwd >= 1.0:
        return 0.1
    elif dp_rwd >= 2.0:
        return 1.0
    else:
        return 0.3


class MonteCarloTree:
    def __init__(self, env):
        self.env = env

    def search_shot(self, root, side, budget=BUDGET, shot_expand=SHOT_EXPAND,
                    skip_ec=None, _taQ=None):
        if root is None:
            root = Node(NodeT.Root, None, None, side)

        for i in tqdm(range(budget)):
            stored = self.env.view.store_balls()
            front = self.tree_policy(root, side, root.actionQ, shot_expand,
                                     None, _taQ)
            if front.reward < 1.0 and front.reward != -2.0:
                dp_reward = self.default_policy(front, side, skip_ec, _taQ)
                # dpr_rate = calc_dpr_rate(front.reward, dp_reward)
                if dp_reward != 0:
                    backup(front, dp_reward)
            self.env.view.restore_balls(stored)

        best = self.best_child(root, 0)
        for child in root.children:
            print(child)
        print("Best: {}".format(best))
        return best.action

    def search_freeball(self, root, side, freeballQ, budget=BUDGET,
                        shot_expand=SHOT_EXPAND, skip_ec=None,
                        fb_expand=FB_EXPAND):
        if root is None:
            root = Node(NodeT.Root, None, None, side)

        for i in tqdm(range(budget)):
            stored = self.env.view.store_balls()
            front = self.tree_policy(root, side, freeballQ, shot_expand,
                                     fb_expand)
            if front.reward < 1.0 and front.reward != -2.0:
                dp_reward = self.default_policy(front, side, skip_ec)
                # dpr_rate = calc_dpr_rate(front.reward, dp_reward)
                if dp_reward != 0:
                    backup(front, dp_reward)
            self.env.view.restore_balls(stored)

        fb = self.best_child(root, 0)
        # for child in root.children:
        #   print(child)
        # print("Best Freeball: {}".format(fb))
        shot = self.best_child(fb, 0)
        print("Best Shot: {}".format(shot))
        return fb.action, shot.action

    def tree_policy(self, root, root_side, Q, shot_expand, fb_expand=None,
                    _taQ=None):
        node = root
        side = root_side

        if fb_expand is not None:
            # process freeball
            if len(node.children) < fb_expand:
                node = self.expand_fb(node, side, Q)
            else:
                node = self.best_child(node, EXPLORE_RATE)
                self.env.view.set_freeball(*node.action)
            Q = node.actionQ
            fb_expand = None

        if len(node.children) < shot_expand:
            return self.expand(node, side, Q, _taQ)
        else:
            node = self.best_child(node, EXPLORE_RATE)
            _, reward = self.env.shot_and_get_result(node.action, side)
            backup(node, reward)
            Q = node.actionQ
            if _taQ is not None:
                _taQ.popleft()
            return node

    def default_policy(self, node, side, skip_ec, _taQ=None):
        stored = self.env.view.store_balls()

        def random_shot(side):
            if _taQ is None:
                aidx = np.random.randint(ACTION_CNT)
                action = ALL_ACTION[aidx]
            else:
                action = _taQ.popleft()
            _, reward = self.env.shot_and_get_result(action, side)
            return reward

        r1 = 0.2 if random_shot(side) >= 1.0 else 0
        r2 = 0
        if skip_ec != side.name:
            side, _ = flip_side(self.env.view, side)
            r2 = random_shot(side)
            r2 = -1.0 if r2 >= 1.0 else 0

        # print("  r1 {} r2 {}".format(r1, r2))
        self.env.view.restore_balls(stored)

        return r1 + r2

    def best_child(self, node, explore_rate):
        best = -100.0
        best_children = []
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(math.log(2 * node.visits) / float(c.visits))
            score = exploit + explore_rate * explore
            if score > best:
                best_children = [c]
                best = score
            elif score == best:
                best_children.append(c)

        if len(best_children) == 0:
            print("No best child found")

        return np.random.choice(best_children)

    def expand_fb(self, node, side, Q):
        action = Q.popleft()
        child = Node(NodeT.Freeball, action, node, side)
        # place cueball as freeball
        self.env.view.set_freeball(*action)
        reward = self.env._decide_fb_reward(side)
        backup(child, reward)
        node.add_child(child)
        return child

    def expand(self, node, side, Q, _taQ):
        if _taQ is not None:
            action = _taQ.popleft()
            Q.popleft()  # consume default action
        else:
            action = Q.popleft()
        child = Node(NodeT.Shot, action, node, side)
        # shot and get reward
        _, reward = self.env.shot_and_get_result(action, side)
        backup(child, reward)
        node.add_child(child)
        return child


def test_mcts():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    _, reward = env.shot_and_get_result((45, 1))
    assert reward == -1
    env.process_result()
    assert env.is_freeball()
    action = mct.search_freeball(env.build_freeballQ(), env.side)
    env.view.set_freeball(*action)
    assert not env.is_freeball()


def test_mcts2():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)

    root = Node(NodeT.Root, None)
    mct.search_shot(root, env.side, budget=1)
    assert len(root.children) == 1
    assert len(root.children[0].children) == 0
    assert root.visits == 1
    assert root.children[0].visits == 1

    pos = env.view.balls[0].pos
    mct.search_shot(root, env.side, budget=2)
    assert np.array_equal(env.view.balls[0].pos, pos)
    assert len(root.children) == 2
    assert len(root.children[0].children) == 0
    assert len(root.children[1].children) == 0
    c1, c2 = root.children
    assert root.visits == 2
    assert root.reward == c1.reward + c2.reward


def test_mcts3():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    root = Node(NodeT.Root, None)
    red = env.view.balls[1]
    test_actionQ = deque([(59, 4), (11, 2)])
    # test_actionQ = deque([(0, 2), (23, 3)])
    front = mct.tree_policy(root, red, root.actionQ, 1, _taQ=test_actionQ)
    assert front.visits == 1
    assert front.reward == 1
    assert front.action == (59, 4)
    assert side == red

    def_reward, ddepth = mct.default_policy(front, side, _taQ=test_actionQ)
    assert ddepth == 1
    backup(front, def_reward)
    assert front.reward == root.reward == 0


def test_mcts4():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    root = Node(NodeT.Root, None)
    red = env.view.balls[1]
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    shot_expand = 1
    cnt = len(root.actionQ)
    front = mct.tree_policy(root, red, root.actionQ, shot_expand, _taQ=actionQ)
    assert len(root.actionQ) + 1 == cnt
    def_reward, ddepth = mct.default_policy(front, side, _taQ=actionQ)
    assert def_reward == 0
    backup(front, def_reward)
    assert front.reward_rate == 0.5

    # retry same action. will not create tree node
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    front = mct.tree_policy(root, red, root.actionQ, shot_expand, _taQ=actionQ)
    def_reward, ddepth = mct.default_policy(front, side, _taQ=actionQ)


def test_mcts5():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    root = Node(NodeT.Root, None)
    org_acnt = len(root.actionQ)
    mct.search_shot(root, env.side, budget=1, shot_expand=1, _taQ=actionQ)

    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    mct.search_shot(root, env.side, budget=1, shot_expand=1, _taQ=actionQ)
    assert len(root.children) == 1
    assert len(root.children[0].children) == 1
    assert len(root.actionQ) == org_acnt - 1
    assert len(root.children[0].actionQ) == org_acnt - 1


def test_mcts6():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    root = Node(NodeT.Root, None)
    fbQ = env.build_freeballQ()
    mct.search_freeball(root, env.side, fbQ, budget=1, fb_expand=1)
    node2 = root.children[0]
    node3 = root.children[0].children[0]
    assert node2.side != node3.side


@cli.command('game')
def game():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)

    while True:
        _game(env, mct)


def _game(env, mct):
    fb_shot = None
    fb_wait = 1
    shot = finished = False
    while True:
        env._render()
        if env.view.move_end():
            if finished:
                if env.winner is None:
                    break
                continue

            if shot:
                env.process_result()
                shot = False
                if env.winner is not None:
                    # record_winner(env)
                    finished = True
            elif env.side.name == 'Blue':
                if env.is_freeball():
                    freeballQ = env.build_freeballQ()
                    fb_expand = int(len(freeballQ) * EXPAND_RATE)
                    fb, fb_shot = mct.search_freeball(None, env.side,
                                                      freeballQ,
                                                      fb_expand=fb_expand,
                                                      skip_ec='Blue')

                    print("Freeball {}".format(fb))
                    env.view.set_freeball(*fb)
                    fb_time = time.time()
                elif fb_shot is not None:
                    if time.time() - fb_time > fb_wait:
                        env.shot(fb_shot)
                        fb_shot = fb_time = None
                else:
                    action = mct.search_shot(None, env.side, skip_ec='Blue')
                    print("Shot {}".format(action))
                    env.shot(action)
        else:
            shot = True


@cli.command('train')
@click.option('--traincnt', 'train_cnt', show_default=True, default=100,
              help="Train pla count")
@click.option('--skipec', 'skip_ec', default=None,
              show_default=True, help="Skip default policy for the side.")
@click.option('--fbstart', 'fb_start', is_flag=True, default=False,
              show_default=True, help="Start with freeball.")
def train(train_cnt, skip_ec, fb_start):
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    win_cnts = defaultdict(int)

    for i in range(train_cnt):
        _train(env, mct, skip_ec, fb_start, win_cnts)


def _train(env, mct, skip_ec, fb_start, win_cnts):
    env.reset()
    fb_shot = fb_time = None
    fb_wait = 1
    shot = finished = False
    if fb_start:
        env.view.start_freeball()

    def record_winner(env):
        win_cnts[env.winner] += 1
        rw = win_cnts['Red']
        bw = win_cnts['Blue']
        print("Red Wins {}, Blue Wins {}".format(rw, bw))

    while True:
        env._render()
        if env.view.move_end():
            # when game is finished, wait until reset
            if finished:
                if env.winner is None:
                    break
                continue

            if shot:
                env.process_result(True)
                shot = False
                if env.winner is not None:
                    record_winner(env)
                    finished = True
            else:
                if env.is_freeball():
                    freeballQ = env.build_freeballQ()
                    fb_expand = int(len(freeballQ) * EXPAND_RATE)
                    fb, fb_shot = mct.search_freeball(None, env.side,
                                                      freeballQ,
                                                      fb_expand=fb_expand,
                                                      skip_ec=skip_ec)
                    print("Freeball {}".format(fb))
                    env.view.set_freeball(*fb)
                    fb_time = time.time()
                elif fb_shot is not None:
                    if time.time() - fb_time > fb_wait:
                        env.shot(fb_shot)
                        fb_shot = fb_time = None
                else:
                    action = mct.search_shot(None, env.side, skip_ec=skip_ec)
                    print("Shot {}".format(action))
                    env.shot(action)
        else:
            shot = True


if __name__ == '__main__':
    st = time.time()
    cli(obj={})
