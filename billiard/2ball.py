import gym
from gym import spaces

import rendering
from rendering import pair


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


class TwoBallEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):
        self.action_space = spaces.Discrete(rendering.MAX_ACTION)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(rendering.IMAGE_HEIGHT,
                                                   rendering.IMAGE_WIDTH, 4))
        self.viewer = None

    def _get_obs(self):
        img = self._get_image()
        return img

    def _get_image(self):
        self.viewer.get_image()

    def _reset(self):
        return self._get_obs()

    def _step(self, action):
        hit_list, obs = self.viewer.shot_and_get_result(action)
        reward = 1 if len(hit_list) > 0 else 0
        done = True
        return obs, reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        self.query_viewer()

        self.viewer.frame_move()
        return_rgb = mode == 'rgb_array'
        return self.viewer.render(return_rgb, True)

    def query_viewer(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(BALL_NAME, BALL_COLOR, BALL_POS)
        return self.viewer


env = TwoBallEnv()
env.query_viewer()


def train():
    obs, reward, done, _ = env.step(pair(25, 10))


def test_shot():
    shot = False
    while True:
        env._render()
        if not shot:
            # env.viewer.random_shot()
            env.viewer.shot(pair(15, 10))
            shot = True
        if env.viewer.move_end():
            print(env.viewer.hit_list)
            break


if __name__ == '__main__':
    train()
    env.viewer.save_image('result.png')
