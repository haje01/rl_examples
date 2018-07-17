# -*- coding: utf8 -*-
"""Pong을 DQN으로."""
import sys
import random
import time
from collections import deque

import psutil
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F  # NOQA
import torchvision.transforms as T  # NOQA
from torch import optim
from torch.nn.init import xavier_uniform_
from tensorboardX import SummaryWriter
from PIL import Image

from wrappers import make_env

TRAIN = True
LOAD_MODEL = "pong_700.pth"
RENDER = True
RENDER_SX = 160
RENDER_SY = 210
ACTION_SIZE = 3
NUM_EPISODE = 50000
TRAIN_IMAGE_SIZE = 84
UPDATE_TARGET_FREQ = 1000
BATCH_SIZE = 32
STATE_SIZE = (4, 84, 84)
GAMMA = 0.99
OPTIM_LR = 0.0001
EGREEDY_END_EPS = 0.02  # 0.1 -> 0.02로 시도
SAVE_FREQ = 300
MAX_REPLAY = 10000
TRAIN_START = 10000
EXPLORE_STEPS = 100000
STOP_REWARD = 18

GIGA = pow(2, 30)

random.seed(0)

writer = SummaryWriter()

resize = T.Compose([
                   T.ToPILImage(),
                   T.Grayscale(),
                   T.Resize((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
                            interpolation=Image.CUBIC),
                   T.ToTensor()
                   ])


class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, action_size):
        """init."""
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, ACTION_SIZE)

    def forward(self, state):
        """전방 연쇄."""
        x = Variable(torch.FloatTensor(state))
        # PyTorch 입력은 Batch, Channel, Height, Width 를 가정
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 첫 번째 배치 사이즈는 그대로 이용하고, 나머지는 as is로 flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 최종 출력은 Q값이기에 soft-max 쓰지 않음.
        return x


def save_batch_images(histories):
    """배치 히스토리 이미지 저장."""
    from scipy import misc
    for i in range(BATCH_SIZE):
        history = histories[i]
        misc.imsave('history_{}.png'.format(i), history.reshape(4 * 84, 84))


class DQNAgent:
    """DQN 에이전트."""

    def __init__(self, device):
        """초기화."""
        self.net = DQN(action_size=ACTION_SIZE).to(device)
        self.net.apply(self.weights_init)
        self.target_net = DQN(action_size=ACTION_SIZE).to(device)
        self.update_target_net()
        self.eps = 1.0
        self.eps_start, self.eps_end = 1.0, EGREEDY_END_EPS
        self.eps_decay = (self.eps_start - self.eps_end) / EXPLORE_STEPS
        self.memory = deque(maxlen=MAX_REPLAY)
        self.avg_q_max, self.avg_loss, self.avg_reward = 0, 0, 0
        self.optimizer = optim.Adam(params=self.net.parameters(),
                                    lr=OPTIM_LR)
        self.replay_buf_size = 0

    def weights_init(self, m):
        """가중치 xavier 초기화."""
        if isinstance(m, nn.Conv2d):
            xavier_uniform_(m.weight.data)

    def get_action(self, state, device):
        """이력을 입력으로 모델에서 동작을 예측하거나 eps로 탐험.

        Args:
            state: byte 형 상태

        Returns:
            0: 정지, 1: 오른쪽, 2: 왼쪽
        """
        if np.random.randn() <= self.eps:
            return self.get_random_action()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_val = self.net(state_v)
            action = int(np.argmax(q_val.data.cpu().numpy()))
            return action

    def get_random_action(self):
        """임의 행동."""
        return random.randrange(ACTION_SIZE)

    def update_target_net(self):
        """타겟 네트웍 갱신."""
        self.target_net.load_state_dict(self.net.state_dict())

    def get_real_action(self, action):
        """이력을 입력으로 환경에 건낼 실제 동작 구함.

        0: 정지, 1: 오른쪽, 2: 왼쪽 -> 1: FIRE , 2: RIGHT, 3: LEFT 으로 맵핑
        """
        if action == 0:
            return 1
        elif action == 1:
            return 2
        else:
            return 3

    def append_sample(self, history, action, reward, next_history, dead):
        """플레이 이력을 추가.

        Args:
            history: byte 형 이력
        """
        self.memory.append((history, action, reward, next_history, dead))

    def train_model(self, device):
        """리플레이 메모리에서 무작위로 추출한 배치로 모델 학습."""
        if self.eps > self.eps_end:
            self.eps -= self.eps_decay

        # 배치 데이터 샘플링
        batch = list(zip(*random.sample(self.memory, BATCH_SIZE)))
        states, actions, rewards, next_states, dones = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = self.net(states_v).\
            gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        # 타겟에서 얻은 다음 가치는 역전파 않음
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        loss.backward()
        self.optimizer.step()

        # # 이력 버퍼 초기화
        # histories = np.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1],
        #                      STATE_SIZE[2]))
        # next_histories = np.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1],
        #                            STATE_SIZE[2]))
        # targets = np.zeros(BATCH_SIZE)
        # actions, rewards, deads = [], [], []

        # # 모든 샘플에 대해
        # for i in range(BATCH_SIZE):
        #     sample = mini_batch[i]
        #     histories[i] = sample[0]
        #     next_histories[i] = sample[3]
        #     actions.append(sample[1])
        #     rewards.append(sample[2])
        #     deads.append(sample[4])

        # # nump -> torch FloatTensor로 변환
        # histories = torch.from_numpy(histories).float()
        # next_histories = torch.from_numpy(next_histories).float()

        # # 모델에서 이번 이력의 동작 가치를 예측
        # actions = torch.LongTensor(actions).unsqueeze(1)
        # # 이력으로 동작 가치(Q) 배열 예측 후, 동작을 인덱스로 동작 가치를 얻음
        # q_values = self.net(histories).gather(1, variable(actions))

        # # 타겟 모델에서 다음 이력에 대한 동작 가치를 예측한 후, 최대값을 타겟 밸류로
        # # (Q-Learning update)
        # target_values = self.target_net(next_histories).data.cpu().\
        #     numpy().max(1)

        # # 모든 버퍼 요소에 대해
        # for i in range(BATCH_SIZE):
        #     # 타겟 가치 갱신
        #     if deads[i]:
        #         targets[i] = rewards[i]
        #     else:
        #         targets[i] = rewards[i] + GAMMA * target_values[i]
        # targets = variable(torch.from_numpy(targets)).float()

        # # 모델과 타겟 모델에서 예측한 Q밸류의 차이가 손실
        # # (Hubber 손실)
        # loss = F.smooth_l1_loss(q_values, targets.unsqueeze(1))
        # self.optimizer.zero_grad()
        # loss.backward()
        # # Gradient Exploding에는 BN보다 Gradient Clip이 유효
        # for param in self.net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        # loss = loss.item()  # loss[0]으로 하면 Variable을 리턴!
        # self.avg_loss += loss
        # return loss


def init_env():
    """환경 초기화."""
    # Deterministic: 프레임을 고정 값(3 or 4)로 스킵
    # v0: 이전 동작을 20% 확률로 반복
    # v4: 이전 동작을 반복하지 않음
    env = make_env('PongNoFrameskip-v4')
    env.reset()
    if RENDER:
        env.render()
        env.unwrapped.viewer.window.width = RENDER_SX
        env.unwrapped.viewer.window.height = RENDER_SY
    return env


def train(device):
    """학습."""
    env = init_env()
    agent = DQNAgent(device)
    global_step = 0
    epstart = None
    elapsed = 0

    for e in range(1, NUM_EPISODE + 1):
        env.reset()
        dead = False
        step, start_life = 0, 5

        observe, _, _, _ = env.step(0)
        state = observe

        done = False
        while not done:
            if RENDER:
                env.render()

            global_step += 1
            step += 1

            if len(agent.memory) < TRAIN_START:
                if global_step % 500 == 0:
                    print("filling replay buffer: {:.2f}".
                          format(len(agent.memory) / float(TRAIN_START)))
                action = agent.get_random_action()
            else:
                action = agent.get_action(state, device)

            raction = agent.get_real_action(action)
            observe, reward, done, info = env.step(raction)
            next_state = observe

            # Q값을 예측
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            r = agent.net(state_v)[0]
            agent.avg_q_max += float(r[0].max())
            agent.avg_reward += reward

            # 죽은 경우 처리
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            agent.append_sample(state, action, reward, next_state, dead)
            if len(agent.memory) >= TRAIN_START:
                agent.train_model(device)

            # 일정 시간마다 타겟 모델을 모델의 가중치로 업데이트
            if global_step % UPDATE_TARGET_FREQ == 0:
                print("update target")
                agent.update_target_net()

            if dead:
                dead = False
            else:
                state = next_state

            if done:
                if epstart is not None:
                    elapsed = time.time() - epstart
                epstart = time.time()
                if len(agent.memory) >= TRAIN_START:
                    # 학습 패러미터
                    writer.add_scalar('data/reward',
                                      agent.avg_reward, global_step)
                    writer.add_scalar('data/loss', agent.avg_loss /
                                      float(step), global_step)
                    writer.add_scalar('data/qmax', agent.avg_q_max /
                                      float(step), global_step)
                    writer.add_scalar('data/step', step, global_step)
                    writer.add_scalar('data/eps', agent.eps, global_step)
                    writer.add_scalar('data/elapse', elapsed, global_step)

                # 메모리 관련 패러미터
                vmem = psutil.virtual_memory()
                total = vmem.total / GIGA
                avail = vmem.available / GIGA
                perc = vmem.percent
                free = vmem.free / GIGA
                print("Episode: {} - replay: {}, memory available: {:.1f}GB, "
                      "percent: {:.1f}%, free: {:.1f}GB".
                      format(e, len(agent.memory), avail, perc, free))
                writer.add_scalars('data/memory',
                                   {'total': total, 'avail': avail,
                                    'free': free,
                                    'replay': agent.replay_buf_size},
                                   global_step)
                size = sum([sys.getsizeof(i) for i in agent.memory[-1]])
                agent.replay_buf_size = size * len(agent.memory) / GIGA
                # 모델 저장
                if e % SAVE_FREQ == 0:
                    path = "pong_{}.pth".format(e)
                    torch.save(agent.net.state_dict(), path)
                    print("saved: {}".format(path))
                    if agent.avg_reward >= STOP_REWARD:
                        sys.exit(0)

                agent.avg_reward, agent.avg_q_max, agent.avg_loss, step = \
                    0, 0, 0, 0


def play(device):
    """플레이."""
    env = init_env()
    agent = DQNAgent(device)
    agent.net.load_state_dict(torch.load(LOAD_MODEL))
    global_step = 0

    while True:
        observe = env.reset()
        dead = False
        step, start_life = 0, 0, 5

        state = np.array([observe], copy=False)

        done = False
        while not done:
            if RENDER:
                env.render()
                time.sleep(0.0333)
            global_step += 1
            step += 1

            action = agent.get_action(state)
            raction = agent.get_real_action(action)
            observe, reward, done, info = env.step(raction)
            next_state = np.array([observe], copy=False)

            # 죽은 경우 처리
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            if dead:
                dead = False
            else:
                state = next_state


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using '{}' device".format(device))
    if TRAIN:
        train(device)
    else:
        play(device)
