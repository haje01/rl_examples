import gym
from gym import spaces

import rendering as rnd

INPUT_SIZE = rnd.OBS_WIDTH * rnd.OBS_HEIGHT * rnd.OBS_DEPTH
OUTPUT_SIZE = rnd.DIV_OF_CIRCLE


class BilliardEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, ball_names, ball_colors, ball_poss):
        degrees = spaces.Discrete(rnd.DIV_OF_CIRCLE)
        forces = spaces.Discrete(rnd.DIV_OF_FORCE)
        self.ball_names = ball_names
        self.ball_colors = ball_colors
        self.ball_poss = ball_poss
        # self.action_space = spaces.Tuple((degrees, forces))
        self.action_space = degrees
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(rnd.OBS_HEIGHT,
                                                   rnd.OBS_WIDTH, 4))
        self.viewer = None

    def _get_obs(self):
        return self.viewer._get_obs()

    def _get_image(self):
        return self.viewer._get_image()

    def _reset(self):
        return self._get_obs()

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
            self.viewer = rnd.Viewer(self.ball_names, self.ball_colors,
                                     self.ball_poss)
        return self.viewer
