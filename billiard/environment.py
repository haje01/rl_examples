import math

import numpy as np
import gym
from gym import spaces

import rendering as rnd


CIRCLE_DEGREE = 360
DIV_OF_CIRCLE = 60
DIV_OF_FORCE = 10
ACTION_DEGREE = CIRCLE_DEGREE / DIV_OF_CIRCLE
MAX_VEL = 1400


class BilliardEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, ball_names, ball_colors, ball_poss, enc_output):
        degrees = spaces.Discrete(DIV_OF_CIRCLE)
        forces = spaces.Discrete(DIV_OF_FORCE)
        self.ball_names = ball_names
        self.ball_colors = ball_colors
        self.ball_poss = ball_poss
        self.enc_output = enc_output
        self.obs_depth = 1 if enc_output else 3
        self.action_space = degrees
        self.view = None

    def _get_obs(self):
        return self.view._get_obs()

    def _get_image(self):
        return self.view._get_image()

    def _reset(self):
        self.view.reset_balls(self.ball_poss)
        return self._get_obs()

    def reset_balls(self, poss):
        self.view.reset_balls(poss)
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.view is not None:
                self.view.close()
                self.view = None
            return

        self.query_viewer(self.view.width, self.view.height)

        self.view.frame_move()
        return_rgb = mode == 'rgb_array'
        return self.view.render(return_rgb, True)

    def query_viewer(self, width, height):
        if self.view is None:
            self.view = rnd.Viewer(width, height, self.ball_names,
                                   self.ball_colors, self.ball_poss,
                                   self.enc_output)
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.view.obs_height,
                                                       self.view.obs_width, 3))
            self.obs_size = self.view.pixel_size * self.obs_depth
            return self.view

    def shot(self, action):
        deg, force = action
        self.view.hit_list = []
        deg *= 360 / DIV_OF_CIRCLE
        rad = math.radians(deg)
        force *= 1 / DIV_OF_FORCE
        vel = force * MAX_VEL
        vx = math.sin(rad) * vel
        vy = math.cos(rad) * vel
        self.view.set_ball_vel(np.array([vx, vy]), 0)

    def shot_and_get_result(self, action):
        self.shot(action)
        fcnt = 1
        hitted = False
        while True:
            hit = self.view.frame_move()
            if hit and not hitted:
                hitted = True
            if not hitted:
                fcnt += 1
            if self.view.move_end():
                break
        self.view.render(True)
        return self.view.hit_list[:], self._get_obs(), fcnt

    def random_shot(self):
        deg = random.randint(0, DIV_OF_CIRCLE)
        force = random.randint(0, DIV_OF_FORCE)
        self.shot((deg, force))

