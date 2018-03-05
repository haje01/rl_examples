"""Breakout을 DQN으로."""
import random
from collections import deque

import numpy as np
import gym
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F  # NOQA
from torch import optim
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorboardX import SummaryWriter


RENDER = True
ACTION_SIZE = 3
RENDER_SX = 160
RENDER_SY = 210
ACTION_SIZE = 3
NUM_EPISODE = 50000
NO_OP_STEPS = 30
CLIP_TOP = 32
CLIP_BOTTOM = 18
TRAIN_IMAGE_SIZE = 84
MAX_REPLAY = 400000
UPDATE_TARGET_FREQ = 10000
BATCH_SIZE = 32
STATE_SIZE = (4, 84, 84)
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 5e-5
TRAIN_START = 50000
SAVE_FREQ = 100

writer = SummaryWriter()


class DQN(nn.Module):
    """Depp Q-Network."""

    def __init__(self, action_size):
        """init."""
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, ACTION_SIZE)

    def forward(self, state):
        """전방 연쇄."""
        x = Variable(torch.FloatTensor(state))
        # PyTorch 입력은 Batch, Channel, Height, Width 를 가정
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 첫 번째 배치 사이즈는 그대로 이용하고, 나머지는 as is로 flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 최종 출력은 Q값이기에 soft-max 쓰지 않음.
        return x


class DQNAgent:
    """에이전트."""

    def __init__(self):
        """초기화."""
        self.net = DQN(action_size=ACTION_SIZE)
        self.target_net = DQN(action_size=ACTION_SIZE)
        self.update_target_net()
        self.eps = 1.0
        self.eps_start, self.eps_end = 1.0, 0.1
        self.explore_steps = 100000  # 초당 7회(스킵포함)로 보면 40시간 동안 탐험 하는 것
        self.eps_decay = (self.eps_start - self.eps_end) / self.explore_steps
        self.memory = deque(maxlen=MAX_REPLAY)
        self.avg_q_max, self.avg_loss, self.avg_reward = 0, 0, 0
        self.optimizer = optim.RMSprop(params=self.net.parameters(),
                                       lr=LEARNING_RATE)

    def get_action(self, state):
        """이력을 입력으로 모델에서 동작을 예측하거나 eps로 탐험."""
        if np.random.randn() <= self.eps:
            return random.randrange(ACTION_SIZE)
        else:
            q_val = self.net(state)
            action = int(q_val[0].max(0)[1])
            return action

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

    def append_sample(self, state, action, reward, next_state, dead):
        """플레이 이력을 추가."""
        self.memory.append((state, action, reward, next_state, dead))

    def train_model(self):
        """리플레이 메모리에서 무작위로 추출한 배치로 모델 학습."""
        if self.eps > self.eps_end:
            self.eps -= self.eps_decay

        # 배치 데이터 샘플링
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        # 이력 버퍼 초기화
        states = np.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1],
                           STATE_SIZE[2]))
        next_states = np.zeros((BATCH_SIZE, STATE_SIZE[0], STATE_SIZE[1],
                                STATE_SIZE[2]))
        targets = np.zeros(BATCH_SIZE)
        actions, rewards, deads = [], [], []

        # 모든 샘플에 대해
        for i in range(BATCH_SIZE):
            sample = mini_batch[i]
            states[i] = sample[0]
            next_states[i] = sample[3]
            actions.append(sample[1])
            rewards.append(sample[2])
            deads.append(sample[4])

        # nump -> torch FloatTensor로 변환
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()

        # 모델에서 이번 이력에 대한 가치를 예측
        actions = torch.LongTensor(actions).unsqueeze(1)
        q_values = self.net(states).gather(1, Variable(actions))

        # 타겟 모델에서 다음 이력에 대한 타겟 가치를 예측
        target_values = self.target_net(next_states).data.numpy().max(1)

        # 모든 버퍼 요소에 대해
        for i in range(BATCH_SIZE):
            # 타겟 가치 갱신
            if deads[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + DISCOUNT_FACTOR * target_values[i]
        targets = Variable(torch.from_numpy(targets)).float()

        # # 이력과 타겟 이력의 차이가 로스
        loss = F.smooth_l1_loss(q_values, targets, size_average=False)
        loss.backward()
        self.optimizer.step()
        self.avg_loss += loss[0]
        return loss[0]


def init_env():
    """환경 초기화."""
    # Deterministic: 프레임을 고정 값(3 or 4)로 스킵
    # v0: 이전 동작을 20% 확률로 반복
    # v4: 이전 동작을 반복하지 않음
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    if RENDER:
        env.render()
        env.env.viewer.window.width = RENDER_SX
        env.env.viewer.window.height = RENDER_SY
    return env


def pre_processing(observe):
    """관측 이미지 전처리."""
    state = observe[CLIP_TOP:, :, :]
    state = state[:-CLIP_BOTTOM, :, :]
    state = resize(rgb2gray(state), (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
                   mode='constant')
    # from scipy import misc
    # misc.imsave('clipped.png', state)
    return state


def train():
    """학습."""
    env = init_env()
    agent = DQNAgent()
    global_step = 0

    for e in range(NUM_EPISODE):
        print("Episode: {}".format(e))
        env.reset()
        dead = False
        step, score, start_life = 0, 0, 5

        # 처음 30 스텝 스킵
        for i in range(NO_OP_STEPS):
            observe, _, _, _ = env.step(1)

        state = pre_processing(observe)
        # 최초는 동일한 4 프레임을 쌓음
        history = np.stack((state, state, state, state), axis=0)
        # batch, width, height, frames
        history = np.reshape([history], (1, 4, 84, 84))

        done = False
        while not done:
            if RENDER:
                env.render()
            if len(agent.memory) < TRAIN_START and global_step % 500 == 0:
                print("filling replay buffer: {:.2f}".
                      format(global_step / float(TRAIN_START)))
            global_step += 1
            step += 1

            state = pre_processing(observe)
            action = agent.get_action(history)
            raction = agent.get_real_action(action)
            observe, reward, done, info = env.step(raction)
            next_state = pre_processing(observe)
            # 배치가 포함된 형태로 변형
            next_state = np.reshape([next_state], (1, 1, 84, 84))
            # 픽셀 단위로 최신 + 최근 3개 이력을 설정.
            next_history = np.append(next_state, history[:, :3, :, :], axis=1)

            # Q값을 예측
            r = agent.net(history)[0]
            agent.avg_q_max += float(r[0].max())
            agent.avg_reward += reward

            # 죽은 경우 처리
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            agent.append_sample(history, action, reward, next_history, dead)
            if len(agent.memory) >= TRAIN_START:
                agent.train_model()

            # 일정 시간마다 타겟 모델을 모델의 가중치로 업데이트
            if global_step % UPDATE_TARGET_FREQ == 0:
                agent.update_target_net()

            if dead:
                dead = False
            else:
                history = next_history

            if done and len(agent.memory) > TRAIN_START:
                # 텐서 보드에 알려줌
                writer.add_scalar('data/reward',
                                  agent.avg_reward, e)
                writer.add_scalar('data/loss', agent.avg_loss / float(step), e)
                writer.add_scalar('data/qmax', agent.avg_q_max / float(step),
                                  e)
                writer.add_scalar('data/step', step, e)
                writer.add_scalar('data/eps', agent.eps, e)
                agent.avg_reward, agent.avg_q_max, agent.avg_loss, step = \
                    0, 0, 0, 0

                # 모델 저장
                if e % SAVE_FREQ == 0:
                    path = f"breakout_{e}.pth"  # NOQA
                    torch.save(agent.net.state_dict(), path)
                    print("saved: {}".format(path))

            score += reward


if __name__ == '__main__':
    train()
