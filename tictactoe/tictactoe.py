import sys
import math
from random import randint
from collections import defaultdict

EMPTY = 0
O = 1
X = 2

STILE_MAP = {0: ' ', 1: 'O', 2: 'X'}

N_PRACTICE = 10000
DEBUG = True

echo = True


def stile(tile):
    return STILE_MAP[tile]


def _print(msg):
    if echo:
        print(msg)


def _write(msg):
    if echo:
        sys.stdout.write(msg)


class Player(object):
    def __init__(self, human, policy='egreedy'):
        self.game = None
        self.side = None
        self.human = human
        self.explore_rate = 0.1
        self.wincnt = 0
        self.policy = policy
        self.reset()
        _print('')

    def reset(self):
        self.prev_state = tuple([0] * 9)

    def set_prev_state(self, board):
        self.prev_state = tuple(board[:])

    def gen_next_move(self):
        if self.human and self.game.game_turn == 1:
            self.game.print_board()

        while True:
            next_moves = self.game.get_blank_tiles()
            if len(next_moves) == 0:
                return

            _write("{}'s turn. ".format(stile(self.side)))
            if self.human:
                yield from self.gen_human_move(next_moves)
            else:
                yield from self.gen_cpu_move(next_moves)

    def gen_human_move(self, next_moves):
        try:
            move = input("Enter move[1-9], q for quit: ")
            if move.lower() == 'q':
                yield None

            move = int(move) - 1
            if move in next_moves:
                yield move
            else:
                _print("Illegal move!")
        except ValueError:
            _print("Illegal move!")

    def gen_cpu_move(self, next_moves):
        yield from self.move_by_policy(next_moves)

    def move_by_policy(self, next_moves):
        if self.policy == 'egreedy':
            yield from self.policy_egreedy(next_moves)
        elif self.policy == 'UCB':
            yield from self.policy_ucb(next_moves)

    def policy_egreedy(self, next_moves):
        e = randint(1, 100)/100.0
        nmoves = len(next_moves)
        if e <= self.explore_rate:
            _write("Exploring. ")
            mi = randint(0, nmoves-1)
            mv = next_moves[mi]
        else:
            _write("Greedy")
            mv = self.greedy_move(next_moves)
            mi = next_moves.index(mv)

        self.game.update_state_value(self.side, self.prev_state, mv)
        _print('')
        yield next_moves[mi]

    def greedy_move(self, next_moves):
        maxv = -2
        maxm = []
        stvals = []
        for nm in next_moves:
            s, v = self.game.get_move_value(nm, self.side)
            stvals.append((s, v))
            if v >= maxv:
                if v > maxv:
                    maxm = []
                maxm.append(nm)
                maxv = v
        _print("(value {:.3f}). ".format(maxv))
        if DEBUG:
            for i, sv in enumerate(stvals):
                _print("  state {} value {}".format(sv[0], sv[1]))
        return maxm[0]

    def policy_ucb(self, next_moves):
        maxk = -2
        maxm = None
        c = 0.00005  # degree of exploration
        t = self.game.total_turn
        s = tuple(self.game.board[:])
        # print(s)
        for nm in next_moves:
            _, v = self.game.get_move_value(nm, self.side)
            n = self.game.ucb_cnt[s][nm]
            if n == 0:
                maxm = nm
                break
            else:
                k = v + math.sqrt(c * math.log(t) / float(n))
            # print("  nm: {} v: {} n: {} k: {}".format(nm, v, n, k))
            if maxk is None or k > maxk:
                maxm, maxk, maxn = nm, k, n

        if maxm is not None:
            self.game.ucb_cnt[s][maxm] += 1

        self.game.update_state_value(self.side, self.prev_state, maxm)
        #    if maxs == (1, 0, 0, 0, 0, 0, 0, 0, 2):
        #        print(self.game.ucb_cnt[maxs])
        # print("UCB: [{}] move: {} (n: {:.2f} t: {:.2f} k: {:.2f})".\
        #    format(s, maxm, maxn, t, maxk))
        yield maxm


def get_win_status(board):
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return

    # draw game
    return 0


def calc_state_value(board, side):
    wstatus = get_win_status(board)
    if wstatus is not None:
        if wstatus == 0:
            # draw
            return 0.5
        else:
            # win / lose
            return 1.0 if wstatus == side else 0.0
    else:
        # playing
        return 0.5


class Game(object):
    def __init__(self):
        self.players = None
        self.st_values = [None, {}, {}]
        self._init_board()
        self.alpha = 0.02
        self.player_moves = None
        self.ucb_cnt = defaultdict(lambda: defaultdict(int))
        self.total_turn = 1

    def set_players(self, players):
        self.vs_human = False
        for i, p in enumerate(players):
            p.game = self
            p.side = i + 1
            if p.human:
                self.vs_human = True
        self.players = players
        self.reset()
        self.cur_player = 0

    def _init_board(self):
        self.board = [0] * 9

    def reset(self, change_turn=False):
        moves = []
        for p in self.players:
            p.reset()
            moves.append(p.gen_next_move())
        self.player_moves = moves
        self.game_turn = 1

    def query_state_value(self, state, side):
        if state not in self.st_values[side]:
            self.st_values[side][state] = calc_state_value(state, side)
        return self.st_values[side][state]

    def get_move_value(self, move, side):
        board = self.board[:]
        board[move] = side
        state = tuple(board)
        return state, self.query_state_value(state, side)

    def update_state_value(self, side, prev_state, move=None):
        state = self.board[:]
        if move is not None:
            state[move] = side
        state = tuple(state)
        pval = self.query_state_value(prev_state, side)
        val = self.query_state_value(state, side)
        diff = val - pval
        npval = pval + self.alpha * diff
        if pval != npval:
            self.st_values[side][prev_state] = npval
            _print("{}'s state {} value from {:.3f} to {:.3f}".
                   format(stile(side), prev_state, pval, npval))

    def switch_player(self):
        self.cur_player = 1 - self.cur_player
        return self.players[self.cur_player]

    def print_board(self):
        _print('')
        for j in range(0, 9, 3):
            _print('|'.join([stile(self.board[i]) for i in range(j, j+3)]))
            if j < 6:
                _print('-----')
        _print('')

    def get_blank_tiles(self):
        return [i for i, t in enumerate(self.board) if t == 0]

    def play(self, order=0):
        _print('')
        _print("==================== Start Game {} ====================".
               format(order+1))
        _print('')
        self._init_board()
        self.cur_player = order % 2
        player = self.players[self.cur_player]
        while player:
            move = next(self.player_moves[self.cur_player])
            if move is None:
                return True
            else:
                player = self.process_move(player, move)
            self.game_turn += 1
            self.total_turn += 1
        return False

    def process_move(self, player, move):
        self.board[move] = player.side
        player.set_prev_state(self.board)
        self.print_board()
        wstatus = get_win_status(self.board)
        if wstatus is not None:
            # Game finished, loser's last learn
            if wstatus != 0:
                self.players[self.cur_player].wincnt += 1
                pl = self.switch_player()
                self.update_state_value(pl.side, pl.prev_state)
            self.print_result(wstatus)
            return None
        else:
            # Keep going
            player = self.switch_player()
            return player

    def print_result(self, wstatus):
        if wstatus != 0:
            _print("'{}' is winner! Game over.".format(stile(wstatus)))
        else:
            _print("It's a draw!")


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def learn_and_play():
    global echo
    print('Learning CPU vs CPU ({})...'.format(N_PRACTICE))
    echo = False
    p1 = Player(False, 'egreedy')
    p2 = Player(False, 'UCB')
    game = Game()
    game.set_players([p1, p2])
    for i in range(N_PRACTICE):
        game.play(i)
        game.reset(True)
        # progress(i, N_PRACTICE)
    print("------------------------")
    print("Win Count - P1: {}, P2: {}".format(p1.wincnt, p2.wincnt))
    print("------------------------")

    echo = True
    print('\nPlay with Human. ({} states learned)'.
          format(len(game.st_values[2])))
    p3 = Player(True)
    p2.explore_rate = 0.0
    game.set_players([p3, p2])
    order = 0
    while True:
        quit = game.play(order)
        if quit:
            break
        game.reset(True)
        order += 1


if __name__ == "__main__":
    learn_and_play()
