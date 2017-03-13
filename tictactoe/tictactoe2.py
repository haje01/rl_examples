import time
import random

import pytest
import numpy as np


EPS = 0.1
ALPHA = 0.1
MAX_EPISODE = 10000
SHOW_LEARN = False


@pytest.fixture
def g():
    return Game()


class IllegalPosition(Exception):
    pass


class Game(object):
    """Tictactoe game class.

    Game state is a numpy array. In the beggining it is filled with 0.
    State side number & symbol
        1: X (CPU)
        2: O (CPU or Player)

    Position is for tile placment, which starts from 1 and ends in 9.

        1|2|3
        -----
        4|5|6
        -----
        7|8|9

    Game state index starts from 0 and ends in 8.

    """

    def __init__(self):
        self.reset()
        self.st_values = {}

    def reset(self):
        self.state = np.zeros(9)

    def query_state_value(self, _state=None):
        state = self.state if _state is None else _state
        tstate = tuple(state)
        if tstate not in self.st_values:
            self.st_values[tstate] = calc_state_value(state)
        return self.st_values[tstate]

    def get_legal_index(self):
        return np.nonzero(np.equal(0, self.state))[0].tolist()

    def get_user_input(self, test_input=None):
        """Get user input.

        Args:
            test_input: Test input

        Returns:
            int: Index of input position.

        Raises:
            IllegalPosition
        """
        idx = None
        if test_input is not None:
            inp = test_input
        else:
            inp = input(self.user_input_prompt())
        try:
            idx = int(inp) - 1
        except ValueError:
            raise IllegalPosition()

        if idx < 0 or idx > 8:
            raise IllegalPosition()

        if idx not in self.get_legal_index():
            raise IllegalPosition()

        return idx

    def user_input_prompt(self):
        return "Enter position[1-9]: "

    def draw(self):
        return draw(self.state)


def egreedy_index(state, legal_indices, query_state_value, side, eps=EPS):
    if random.random() < eps:
        return random.choices(legal_indices)[0]
    else:
        indices = []
        max_val = -1
        for s in [side, 3 - side]:
            for li in legal_indices:
                state[li] = s
                val = query_state_value(state)
                if s == 1:
                    val = 1 - val
                if val > max_val:
                    indices = [li]
                    max_val = val
                elif val == max_val and val < 1.0:
                    indices.append(li)
                state[li] = 0
        return random.choices(indices)[0]


def draw(state):
    rv = '\n'
    for y in range(3):
        for x in range(3):
            idx = y * 3 + x
            t = state[idx]
            if t == 1:
                rv += 'X'
            elif t == 2:
                rv += 'O'
            else:
                if x < 2:
                    rv += ' '
            if x < 2:
                rv += '|'
        rv += '\n'
        if y < 2:
            rv += '-----\n'
    return rv


def judge(g):
    wside = get_win_side(g.state)
    finish = False
    if wside > 0:
        print(g.draw())
        print("Winner is: {}".format(wside))
        finish = True
    elif len(g.get_legal_index()) == 0:
        print("Draw!")
        finish = True

    if finish:
        again = input("Play again? (y/n): ")
        if again.lower() != 'y':
            return True
        else:
            g.reset()
            return False


def play_turn(g, side):
    if side == 1:
        idx = egreedy_index(g.state, g.get_legal_index(), g.query_state_value,
                            side, 0)
    else:
        while True:
            try:
                idx = g.get_user_input()
            except IllegalPosition:
                print("Illegal position!")
            else:
                break

    g.state[idx] = side
    print(g.draw())

    stop = judge(g)
    return 3 - side, stop


def play(_g=None):
    g = Game() if _g is None else _g
    side = 1
    while True:
        side, stop = play_turn(g, side)
        if stop is not None:
            if stop:
                break
            else:
                if side == 2:
                    print(g.draw())
                continue


def learn_and_play():
    g = Game()
    side = 1
    wside = 0
    for i in range(MAX_EPISODE):
        lidx = g.get_legal_index()
        if len(lidx) == 0:
            wside = -1
        else:
            state = tuple(g.state)
            idx = egreedy_index(g.state, lidx, g.query_state_value, side)
            value = g.query_state_value()
            g.state[idx] = side
            if SHOW_LEARN:
                print(g.draw())

            wside = get_win_side(g.state)
            nvalue = g.query_state_value()
            g.st_values[state] = update_values(value, nvalue)
            side = 3 - side

        if SHOW_LEARN:
            time.sleep(1)
            if wside > 0:
                print('Winner is {}'.format(wside))
            elif wside == -1:
                print('Draw')

        if wside > 0 or wside == -1:
            g.reset()
            if SHOW_LEARN:
                time.sleep(1)

    save(g.st_values)
    g.reset()
    play(g)


def calc_state_value(state):
    ws = get_win_side(state)
    if ws == 2:
        return 1
    elif ws == 1:
        return 0
    else:
        return 0.5


def update_values(this_value, next_value):
    diff = next_value - this_value
    return this_value + ALPHA * diff


def test_draw(g):
    assert g.draw() == '''
 | |
-----
 | |
-----
 | |
'''
    assert len(g.state) == 9

    # 1|2|3
    # -----
    # 4|5|6
    # -----
    # 7|8|9
    #
    # X: 1, O: 2
    pos = 1
    idx = pos - 1
    g.state[idx] = 2
    assert g.draw() == '''
O| |
-----
 | |
-----
 | |
'''
    pos = 9
    idx = pos - 1
    g.state[idx] = 1
    assert g.draw() == '''
O| |
-----
 | |
-----
 | |X
'''


def test_user_input(g):
    assert g.user_input_prompt() == "Enter position[1-9]: "
    with pytest.raises(IllegalPosition):
        g.get_user_input('eueueu')

    with pytest.raises(IllegalPosition):
        g.get_user_input('0')

    with pytest.raises(IllegalPosition):
        g.get_user_input('10')

    assert g.get_user_input('1') == 0

    g.state[0] = 1
    with pytest.raises(IllegalPosition):
        g.get_user_input('1')


def test_legal_positions(g):
    assert g.get_legal_index() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    g.state[0] = 1
    assert g.get_legal_index() == [1, 2, 3, 4, 5, 6, 7, 8]
    g.state[8] = 2
    assert g.get_legal_index() == [1, 2, 3, 4, 5, 6, 7]
    g.state[5] = 1
    assert g.get_legal_index() == [1, 2, 3, 4, 6, 7]


def test_play_update(g):
    g.state = np.array((2, 1, 2,
                        1, 2, 1,
                        0, 0, 0))
    assert 0.5 == g.query_state_value()

    g.state[8] = 2
    assert 1.0 == g.query_state_value()
    assert g.st_values[tuple(g.state)] == 1.0


def save(st_values):
    with open("result.txt", "w") as f:
        for state, value in st_values.items():
            f.write("{}: {}\n".format(state, value))


def get_win_side(_state):
    state = _state.copy().reshape((3, 3))

    for s in [1, 2]:
        for t in range(2):
            for r in range(3):
                if np.array_equal(np.unique(state[r]), [s]):
                    return s
            state = state.transpose()

        # check diagonals
        if _state[0] == s and _state[4] == s and _state[8] == s:
            return s
        if _state[2] == s and _state[4] == s and _state[6] == s:
            return s

    return 0


def test_win_state(g):
    g.state[0] = 1
    g.state[1] = 1
    g.state[2] = 1
    assert 1 == get_win_side(g.state)

    g.reset()
    g.state[0] = 1
    g.state[3] = 1
    g.state[6] = 1
    assert 1 == get_win_side(g.state)

    g.reset()
    g.state[0] = 2
    g.state[1] = 2
    g.state[2] = 2
    assert 2 == get_win_side(g.state)

    g.reset()
    g.state[0] = 2
    g.state[3] = 2
    g.state[6] = 2
    assert 2 == get_win_side(g.state)

    g.reset()
    g.state[0] = 1
    g.state[4] = 1
    g.state[8] = 1
    assert 1 == get_win_side(g.state)

    g.reset()
    g.state[0] = 2
    g.state[4] = 2
    g.state[8] = 2
    assert 2 == get_win_side(g.state)

    g.reset()
    g.state[2] = 1
    g.state[4] = 1
    g.state[6] = 1
    assert 1 == get_win_side(g.state)

    g.reset()
    g.state[2] = 2
    g.state[4] = 2
    g.state[6] = 2
    assert 2 == get_win_side(g.state)


def test_state_value(g):
    state = np.zeros(9)
    state[0] = 2
    state[1] = 2
    state[2] = 2
    assert 1 == calc_state_value(state)
    assert 1 == g.query_state_value(state)
    assert 1 == g.st_values[tuple(state)]

    state = np.zeros(9)
    state[0] = 1
    state[1] = 1
    state[2] = 1
    assert 0 == calc_state_value(state)
    assert 0 == g.query_state_value(state)

    state = np.zeros(9)
    assert 0.5 == calc_state_value(state)
    assert 0.5 == g.query_state_value(state)

    assert 1.0 == g.query_state_value(np.array((1.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                                                2.0, 0.0, 2.0)))


def test_egreedy_policy(g):
    """
O|O|   legal index: 2, 7
-----
O|X|X
-----
X| |X

O|O|O  value: 1
-----
O|X|X
-----
X| |X

O|O|   value: 0.5
-----
O|X|X
-----
X|O|X
    """
    state = np.array([2, 2, 0, 2, 1, 1, 1, 0, 1])

    results = []
    g.state = np.array(state)
    for i in range(100):
        res = egreedy_index(state, [2, 7], g.query_state_value, 2)
        results.append(res)

    results = np.array(results)

    gcnt = np.count_nonzero(results == 2)
    rcnt = np.count_nonzero(results == 7)
    assert gcnt >= rcnt * 9


def test_egreedy_policy2(g):
    """
O|O|   legal index: 2, 3, 6, 7, 8
-----
 |X|X
-----
 | |
    """
    state = np.array([2, 2, 0, 0, 1, 1, 0, 0, 0])
    results = []
    g.state = np.array(state)
    for i in range(100):
        res = egreedy_index(state, [2, 3, 6, 7, 8], g.query_state_value, 1)
        results.append(res)

    results = np.array(results)

    gcnt = np.count_nonzero(results == 3)
    rcnt = np.count_nonzero(results != 3)
    assert gcnt >= rcnt * 9


if __name__ == "__main__":
    learn_and_play()

