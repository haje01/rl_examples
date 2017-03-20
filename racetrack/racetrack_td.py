import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from racetrack_env import RacetrackEnv, Map, REWARD_SUCCESS

CONFIGS = {
    'map4.txt': {
        'max_step': 70,
        'egreedy': {
            'max_episode': 5000,
        },
        'ucb': {
            'max_episode': 300000,
        },
    },
    'map5.txt': {
        'max_step': 200,
        'egreedy': {
            'max_episode': 10000,
        },
        'ucb': {
            'max_episode': 15000,
        },
    }
}

MAP_NAME = 'map5.txt'
EGREEDY_EPS = 0.3
UCB = True  # True: UCB, False: E-Greedy
UCB_C = 0.5
GAMMA = 0.1
ALPHA = 0.5
SHOW_TERM = 1000


def make_env(map_filenm, policy):
    cfg = CONFIGS[map_filenm]
    max_step = cfg['max_step']
    max_episode = cfg[policy]['max_episode']
    with open(map_filenm, 'r') as f:
        amap = Map(f.read())

    vel_info = (
        -3, 3,  # vx min / max
        -3, 3   # vy min / max
    )

    env = RacetrackEnv(amap, vel_info, max_step)
    spolicy = "UCB" if UCB else "EGreedy"
    save_filenm = '{}_{}.sav'.format(map_filenm.split('.')[0], spolicy).lower()
    image_filenm = '{}_{}.png'.format(map_filenm.split('.')[0],
                                      spolicy).lower()
    return env, max_episode, max_step, save_filenm, image_filenm, spolicy


def max_policy(env, Q, state):
    aprobs = Q[state]
    action = np.random.choice(np.flatnonzero(aprobs == aprobs.max()))
    return action


def egreedy_policy(env, Q, state, e_no, max_episode, test_action=None):
    aprobs = Q[state]
    if test_action is not None:
        action = test_action
    else:
        action = np.random.choice(np.flatnonzero(aprobs == aprobs.max()))
    nA = env.action_space.n
    eps = EGREEDY_EPS * (1 - float(e_no) / max_episode)
    # eps = EGREEDY_EPS
    A = np.ones(nA) * eps / nA
    A[action] += (1.0 - eps)
    return A


def egreedy_action(aprobs, nA):
    return np.random.choice(range(nA), p=aprobs)


def test_egreedy_policy():
    env, max_episode, max_step, _, _, _ = make_env()
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    best_action = 1
    state = None
    aprobs = egreedy_policy(env, Q, state, 1, best_action)
    assert np.array_equal(aprobs, np.array([0.02, 0.92, 0.02, 0.02, 0.02]))
    n = 0
    acnt = defaultdict(int)
    TRY_CNT = 100
    while n < TRY_CNT:
        action = np.random.choice(range(nA), p=aprobs)
        acnt[action] += 1
        n += 1
    EPS_CNT = 100 * EGREEDY_EPS
    assert TRY_CNT - acnt[best_action] < 2 * EPS_CNT


def ucb_policy(env, Q, N, state, t, e, uqr_ratios=None, show=False,
               test_action=None):
    n = sum(N[state])
    rarity = UCB_C * np.sqrt(math.log(n) / N[state])
    rv = Q[state] + rarity
    if uqr_ratios is not None:
        q_span = np.ptp(Q[state])
        r_span = np.ptp(rarity)
        if r_span == 0.0:
            ratio = 0
        else:
            ratio = q_span / r_span
        uqr_ratios.append(ratio)
    return rv


def ucb_action(aprobs, state, N, update=True):
    action = np.random.choice(np.flatnonzero(aprobs == aprobs.max()))
    if update:
        N[state][action] += 1
    return action


def test_ucb_policy():
    env, max_episode, max_step, _, _, _ = make_env()
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.ones(nA))
    best_action = 1
    state = None
    t = 1

    acnt = defaultdict(int)
    for i in range(nA):
        aprobs = ucb_policy(env, Q, N, state, t, 1, None, False, best_action)
        action = ucb_action(aprobs, state, N)
        acnt[action] += 1
        t += 1

    assert list(acnt.values()) == [1] * nA


def make_greedy_policy(Q):
    def func(state):
        A = np.zeros_like(Q[state], dtype=float)
        high = np.max(Q[state])
        cand = np.flatnonzero(np.equal(Q[state], high))
        best_action = np.random.choice(cand)
        A[best_action] = 1.0
        return A
    return func


def _run_step(env, Q, N, state, nA, n_episode, n_step, max_episode,
              uqr_ratios, show, greedy_policy):
    if UCB:
        aprobs = ucb_policy(env, Q, N, state, n_step + 1, n_episode + 1,
                            uqr_ratios, show)
        action = ucb_action(aprobs, state, N)
        # action = max_policy(env, Q, state)
    else:
        #aprobs = egreedy_policy(env, Q, state, n_episode + 1, max_episode)
        #action = egreedy_action(aprobs, nA)
        action = max_policy(env, Q, state)

    nstate, reward, done, _ = env.step(action)

    if UCB:
        naprobs = ucb_policy(env, Q, N, state, n_step + 2, n_episode + 1)
        naction = ucb_action(naprobs, state, N, False)
    else:
        naprobs = egreedy_policy(env, Q, nstate, n_episode + 1, max_episode)
        naction = np.random.choice(range(nA), p=naprobs)

    v = Q[state][action]
    nv = Q[nstate][naction]
    td_target = reward + GAMMA * nv
    td_delta = td_target - v
    Q[state][action] += ALPHA * td_delta
    return nstate, action, reward, done


def _print_policy_progress(Q, state, N, action):
    if UCB:
        print("  ", state, Q[state], N[state], action)
    else:
        print("  ", state, Q[state], action)


def _print_done_msg(success):
    if success:
        print(" SUCCESS!!")
    else:
        print(" DONE")


def calc_progress(env, Q):
    return len(Q.keys()) / float(env.num_state_space)


def learn_Q(env, max_episode, max_step, ep_steps, ep_progs,
            ep_uqr_ratios=None):
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(lambda: np.ones(nA))
    greedy_policy = make_greedy_policy(Q)

    for n_episode in range(max_episode):
        state = env.reset()
        show = (n_episode + 1) % SHOW_TERM == 0
        if show:
            print("========== Episode: {} / {} ==========".
                  format(n_episode + 1, max_episode))

        uqr_ratios = []
        for n_step in range(max_step):
            state, action, reward, done = _run_step(env, Q, N, state, nA,
                                                    n_episode, n_step,
                                                    max_episode, uqr_ratios,
                                                    show, greedy_policy)
            success = reward == REWARD_SUCCESS
            # if show:
            #     _print_policy_progress(Q, state, N, action)
            if done:
                if show:
                    _print_done_msg(success)
                break
        if ep_uqr_ratios is not None:
            ep_uqr_ratios.append(sum(uqr_ratios) / float(len(uqr_ratios) + 1))
        ep_steps.append(n_step)

        prog = calc_progress(env, Q)
        ep_progs.append(prog)
        if show:
            print("    {0:.2f}%".format(prog * 100))

    with open('visits.txt', 'w') as fw:
        for state, visits in N.items():
            fw.write('{}: {}\n'.format(state, N[state]))

    return Q


def run():
    env, max_episode, max_step, save_filenm, image_filenm, spolicy =\
        make_env(MAP_NAME, 'ucb' if UCB else 'egreedy')
    Q = None
    if os.path.isfile(save_filenm):
        ans = input("Saved file '{}' exists. Load the file and play? (Y/N): "
                    .format(save_filenm))
        if ans.lower().startswith('y'):
            Q = env.load(save_filenm)

    if Q is None:
        print("Start new learning. Map: {}, Policy: {}".format(MAP_NAME,
                                                               spolicy))
        ep_steps = []
        ep_progs = []
        ep_uqr_ratios = [] if UCB else None
        Q = learn_Q(env, max_episode, max_step, ep_steps, ep_progs,
                    ep_uqr_ratios)
        env.save(Q, save_filenm)

        plt.figure(1)
        plt.subplot(311)
        plt.ylabel('Max step')
        plt.plot(range(max_episode), ep_steps, 'b')

        plt.subplot(312)
        plt.ylabel('Progress')
        plt.plot(range(max_episode), ep_progs, 'g')

        if UCB:
            plt.subplot(313)
            plt.ylabel('Q span / Rarity span')
            plt.plot(range(max_episode), ep_uqr_ratios, 'r')
        plt.xlabel('Episode Number')
        plt.savefig(image_filenm)

    play_policy = make_greedy_policy(Q)
    env.play(play_policy, 1)


if __name__ == "__main__":
    run()
