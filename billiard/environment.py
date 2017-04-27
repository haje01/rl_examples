import math
import random

import numpy as np
import numpy.linalg as la
import gym
from gym import spaces

import rendering as rnd


CIRCLE_DEGREE = 360
DIV_OF_CIRCLE = 60
ACTION_DEGREE = CIRCLE_DEGREE / DIV_OF_CIRCLE
MAX_FORCE_DIST = 150


def calc_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angle = np.pi - math.atan2(det, dot)
    return angle


def calc_angle2(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


class BilliardEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self, ball_info, hole_info, max_vel, div_of_force,
                 enc_output):
        degrees = spaces.Discrete(DIV_OF_CIRCLE)
        # forces = spaces.Discrete(self.div_of_force)
        self.ball_info = ball_info
        self.hole_info = hole_info
        self.max_vel = max_vel
        self.div_of_force = div_of_force
        self.enc_output = enc_output
        self.obs_depth = 1 if enc_output else 3
        self.action_space = degrees
        self.view = None
        self.drag_shot_info = None

    def _get_obs(self, hud=False):
        return self.view._get_obs(hud)

    def _get_image(self):
        return self.view._get_image()

    def _reset(self):
        self.view.reset_balls()
        return self._get_obs(False)

    def reset_balls(self, poss):
        self.view.reset_balls(poss)
        return self._get_obs(False)

    def _render(self, mode='human', close=False):
        if close:
            if self.view is not None:
                self.view.close()
                # self.view = None
            return

        self.query_viewer(self.view.width, self.view.height)

        self.view.frame_move()
        return_rgb = mode == 'rgb_array'
        return self.view.render(return_rgb, True)

    def _get_hud(self):
        return

    def query_viewer(self, width, height):
        if self.view is None:
            self.view = rnd.Viewer(self, width, height, self.ball_info,
                                   self.hole_info, self.enc_output)
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.view.obs_height,
                                                       self.view.obs_width, 3))
            self.obs_size = self.view.pixel_size * self.obs_depth
            return self.view

    def shot(self, action, show_info=True):
        deg, force = action
        # force = int(force)
        if show_info:
            print(deg, force)
        self.view.clear_shot_result()
        deg *= 360 / DIV_OF_CIRCLE
        rad = math.radians(deg)
        force *= 1 / self.div_of_force
        vel = force * self.max_vel
        vx = math.sin(rad) * vel
        vy = math.cos(rad) * vel
        self.view.set_ball_vel(np.array([vx, vy]), 0)

    def shot_and_get_result(self, action):
        raise NotImplemented()

    def prepare_drag_shot(self, x, y):
        pos = x, y
        dist = min(self.view.cueball._dist(pos), MAX_FORCE_DIST)
        force = dist / MAX_FORCE_DIST * self.div_of_force
        bx, by = self.view.cueball.pos
        sdir = (x - bx, y - by)
        deg = calc_angle((0, 1), sdir)
        deg = calc_angle((0, 1), sdir) * 180 / np.pi
        deg *= DIV_OF_CIRCLE / 360
        deg = int(deg)
        # print("dist {} force {} sdir {} deg {}".format(dist, force, sdir, deg))
        self.drag_shot_info = (sdir, deg, force)

    def drag_shot(self):
        if self.drag_shot_info is not None:
            _, deg, force = self.drag_shot_info
            self.shot((deg, force))
            self.drag_shot_info = None

    def random_shot(self):
        deg = random.randint(0, DIV_OF_CIRCLE-1)
        force = random.randint(1, self.div_of_force)
        print("deg {}, force {}".format(deg, force))
        self.shot((deg, force))
