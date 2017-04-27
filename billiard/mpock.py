import time
import math
from collections import deque

import click
import numpy as np
from tqdm import tqdm

from environment import BilliardEnv, calc_angle2, DIV_OF_CIRCLE
from rendering import BState

WIN_WIDTH = 200
WIN_HEIGHT = 200
DIV_OF_FORCE = 5
MAX_VEL = 350
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
    (80, 140),  # Red
    (140, 80),  # Blue
]
HOLES = [
    [170, 170]
]

DEFAULT_FORCE = 2
GOOD_ANG = 1.0
GOOD_DIST = 35
EXPLORE_RATE = 1/math.sqrt(2.0)

ACTION_CNT = DIV_OF_CIRCLE * DIV_OF_FORCE
ALL_ACTION = []
for d in range(DIV_OF_CIRCLE):
    for f in range(DIV_OF_FORCE):
        ALL_ACTION.append((d, f+1))

MCTS_BUDGET = 1000
MCTS_DEPTH = 1
MCTS_MIN_EXPAND = int(ACTION_CNT * 0.5)
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
        return self._get_obs(), self._decide_reward(side),

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

    def _decide_reward(self, side):
        reward = 0
        # lost
        if self.eside in self.view.holein_list:
            reward = -2
        # enemy's free ball
        elif self.view.cueball in self.view.holein_list or\
                len(self.view.hit_list) == 0 or\
                side != self.view.hit_list[0]:
            reward = -1
        # win
        elif side in self.view.holein_list:
            reward = 2
        # good position
        elif self._in_good_pos(side):
            reward = 1
        return reward

    def build_freeballQ(self):
        poss = []
        ox, oy = self.view.cueball.pos
        off = WALL_DEPTH + BALL_RAD
        for x in range(off, WIN_WIDTH - off, 4):
            for y in range(off, WIN_HEIGHT - off, 4):
                self.view.cueball.pos = x, y
                if self.view.is_valid_fb():
                    poss.append((x, y))
        self.view.cueball.pos = (ox, oy)
        return deque(poss)

    def _get_hud(self):
        if self.winner is not None:
            msg = "{}'s Win!".format(self.winner)
            return msg, 62, 90
        else:
            msg = "{}'s turn".format(self.side.name)
            if self.drag_shot_info is not None:
                _, _, force = self.drag_shot_info
                msg += " F: {:.1f}".format(force/self.div_of_force)
            elif self.is_freeball():
                msg += " [Free Ball]"
            return msg, 10, 5

    def process_result(self):
        reward = self._decide_reward(self.side)
        print("reward {}".format(reward))
        if reward == 2:
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
    def __init__(self, action, parent=None):
        self.visits = 0
        self.reward = 0.0
        self.action = action
        self.parent = parent
        self.children = []
        self.actionQ = shuffle_actions()

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    @property
    def reward_rate(self):
        if self.visits > 0:
            return self.reward / float(self.visits)
        return MIN_REWARD_RATE

    def expanded_enough(self):
        return len(self.children) > MCTS_MIN_EXPAND

    def __repr__(self):
        return "Node: children {} visits {} action {} reward_rate {:.1f}".\
            format(len(self.children), self.visits, self.action,
                   self.reward_rate)


def backup(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent


class MonteCarloTree:
    def __init__(self, env):
        self.env = env

    def search_shot(self, root, side, budget=MCTS_BUDGET, depth=MCTS_DEPTH,
                    min_expand=MCTS_MIN_EXPAND, _taQ=None):
        if root is None:
            root = Node(None)

        for i in tqdm(range(budget)):
            self.env.view.store_balls()
            front, tdepth, front_side = self.tree_policy(root, side,
                                                         root.actionQ, depth,
                                                         min_expand, _taQ)
            if depth - tdepth > 0:
                reward, _ = self.default_policy(front, front_side, depth -
                                                tdepth, _taQ)
                backup(front, reward)
            self.env.view.restore_balls()

        best = self.best_child(root, 0)
        # for child in root.children:
        #    print(child)
        print("Best: {}".format(best))
        return best.action

    def search_freeball(self, root, freeballQ, root_side, budget=MCTS_BUDGET,
                        depth=MCTS_DEPTH, min_expand=MCTS_MIN_EXPAND):
        if root is None:
            root = Node(None)

        for i in tqdm(range(budget)):
            front, tdepth, front_side = self.tree_policy(root, root_side,
                                                         freeballQ, depth,
                                                         min_expand)
            reward, _ = self.default_policy(front, front_side, depth - tdepth)
            backup(front, reward)

        best = self.best_child(root, 0)
        # for child in root.children:
        #    print(child)
        print("Best: {}".format(best))
        return best.action

    def tree_policy(self, node, root_side, Q, max_depth, min_expand,
                    _taQ=None):
        depth = 0
        side = root_side
        while depth < max_depth:
            depth += 1
            if depth > 1:
                side, _ = flip_side(self.env.view, side)

            if len(node.children) < min_expand:
                return self.expand(node, side, Q, _taQ), depth, side
            else:
                node = self.best_child(node, EXPLORE_RATE)
                Q = node.actionQ
                if _taQ is not None:
                    _taQ.popleft()

        return node, depth, side

    def default_policy(self, node, parent_side, max_depth, _taQ=None):
        def_reward = 0

        side = parent_side
        for i in range(max_depth):
            side, _ = flip_side(self.env.view, side)

            if _taQ is None:
                aidx = np.random.randint(ACTION_CNT)
                action = ALL_ACTION[aidx]
            else:
                action = _taQ.popleft()

            _, reward = self.env.shot_and_get_result(action, side)
            if side != parent_side:
                reward *= -1
            def_reward += reward
            # stop if game finished.
            if reward == 2 or reward == -2:
                break

        return def_reward, i+1 if max_depth > 0 else 0

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

    def expand(self, node, side, Q, _taQ):
        if _taQ is not None:
            action = _taQ.popleft()
            Q.popleft()  # consume default action
        else:
            if len(Q) == 0:
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                pass
            action = Q.popleft()
        child = Node(action, node)
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

    root = Node(None)
    action = mct.search_shot(root, env.side, budget=1, depth=1)
    assert len(root.children) == 1
    assert len(root.children[0].children) == 0
    assert root.visits == 1
    assert root.children[0].visits == 1

    pos = env.view.balls[0].pos
    action = mct.search_shot(root, env.side, budget=2, depth=1)
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
    root = Node(None)
    depth = 2
    red = env.view.balls[1]
    blue = env.view.balls[2]
    test_actionQ = deque([(59, 4), (11, 2)])
    # test_actionQ = deque([(0, 2), (23, 3)])
    front, tdepth, side = mct.tree_policy(root, red, root.actionQ, depth,
                                          1, _taQ=test_actionQ)
    assert tdepth == 1
    assert front.visits == 1
    assert front.reward == 1
    assert front.action == (59, 4)
    assert side == red

    def_reward, ddepth = mct.default_policy(front, side, depth - tdepth,
                                            _taQ=test_actionQ)
    assert ddepth == 1
    backup(front, def_reward)
    assert front.reward == root.reward == 0


def test_mcts4():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    root = Node(None)
    depth = 3
    red = env.view.balls[1]
    blue = env.view.balls[2]
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    min_expand = 1
    cnt = len(root.actionQ)
    front, tdepth, side = mct.tree_policy(root, red, root.actionQ, depth,
                                          min_expand, _taQ=actionQ)
    assert len(root.actionQ) + 1 == cnt
    def_reward, ddepth = mct.default_policy(front, side, depth - tdepth,
                                            _taQ=actionQ)
    assert def_reward == 0
    backup(front, def_reward)
    assert front.reward_rate == 0.5

    # retry same action. will not create tree node
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    front, tdepth, side = mct.tree_policy(root, red, root.actionQ, depth,
                                          min_expand, _taQ=actionQ)
    def_reward, ddepth = mct.default_policy(front, side, depth - tdepth,
                                            _taQ=actionQ)


def test_mcts5():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)
    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    root = Node(None)
    org_acnt = len(root.actionQ)
    action = mct.search_shot(root, env.side, budget=1, depth=2,
                             min_expand=1, _taQ=actionQ)

    actionQ = deque([(59, 4), (11, 2), (51, 1.5)])  # R, B, R
    action = mct.search_shot(root, env.side, budget=1, depth=2,
                             min_expand=1, _taQ=actionQ)
    assert len(root.children) == 1
    assert len(root.children[0].children) == 1
    assert len(root.actionQ) == org_acnt - 1
    assert len(root.children[0].actionQ) == org_acnt - 1


@cli.command('game')
def game():
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    env.flip_side()

    shot = False
    while True:
        env._render()
        if env.view.move_end():
            if shot:
                env.process_result()
                shot = False
        else:
            shot = True


@cli.command('train')
@click.option('--traincnt', 'train_cnt', show_default=True, default=100,
              help="Train pla count")
@click.option('--freeball', 'freeball', is_flag=True, default=False,
              show_default=True, help="Start with freeball.")
def train(train_cnt, freeball):
    env = MiniPocketEnv(False)
    env.query_viewer(WIN_WIDTH, WIN_HEIGHT)
    mct = MonteCarloTree(env)

    for i in range(train_cnt):
        _train(env, mct)


def _train(env, mct):
    env.reset()
    shot = False
    while True:
        env._render()
        if env.view.move_end():
            if shot:
                env.process_result()
                shot = False
                if env.winner is not None:
                    print("Winner is {}".format(env.winner))
                    break
            else:
                if env.is_freeball():
                    action = mct.search_freeball(None, env.build_freeballQ(),
                                                 env.side)
                    env.view.set_freeball(*action)
                    print("Freeball {}".format(action))
                else:
                    action = mct.search_shot(None, env.side)
                    print("Shot {}".format(action))
                    env.shot(action)
        else:
            shot = True


if __name__ == '__main__':
    st = time.time()
    cli(obj={})
