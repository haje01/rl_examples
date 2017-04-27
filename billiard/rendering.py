import math
from enum import Enum

import scipy.misc
import numpy as np
import pyglet
from pyglet.window import mouse
from pyglet.gl import *  # NOQA

WALL_DEPTH = 30
BALL_POINTS = 12
HOLE_POINTS = 24
SLATE_COLOR = (0, 0.4, 0, 1)
WALL_COLOR = (0.6, 0.3, 0.1)
BALL_RAD = 12
HOLE_RAD = 35
HOLEIN_DIST = HOLE_RAD * 0.9
FORCE_EPS = 30
DEFAULT_FRIC_RATE = 0.99
HIT_FRIC_RATE = 0.7
HIT_FRIC_RATE2 = 0.3
WALL_FRIC_RATE = 0.8
FIX_DELTA = 0.01667
RGB_TO_BYTE = {
    (  0, 102,   0): 0,  # Slate
    (153,  76,  25): 1,  # Wall
    (255, 255, 255): 2,  # White
    (255,   0,   0): 3,  # Red
}

BYTE_TO_RGB = {c: rgb for rgb, c in RGB_TO_BYTE.items()}

NO_COL_DIST = 2
MAX_COL_REPEL = 30
CON_HIT_LIMIT = 1
SHOT_LINE_LEN = 80


class BState(Enum):
    NORMAL = 1
    HOLEIN = 2
    FREEBALL = 3


def make_circle(rad, ptcnt):
    verts = []
    for i in range(ptcnt):
        angle = math.radians(float(i)/ptcnt * 360.0)
        x = rad * math.cos(angle)
        y = rad * math.sin(angle)
        verts += [x, y]
    return pyglet.graphics.vertex_list(ptcnt, ('v2f', verts))


def draw_polygon(v):
    vcnt = len(v)
    if vcnt == 4:
        glBegin(GL_QUADS)
    elif vcnt > 4:
        glBegin(GL_POLYGON)
    else:
        glBegin(GL_TRIANGLES)
    glColor3f(*WALL_COLOR)
    for p in v:
        glVertex2f(p[0], p[1])
    glEnd()


class Ball:
    ball_circle = make_circle(BALL_RAD, BALL_POINTS)

    def __init__(self, view, name, no, pos, color):
        self.view = view
        self.name = name
        self.no = no
        self.pos = np.array(pos)
        self.color = color
        self.reset()
        self.hit = False

    def reset(self):
        self.state = BState.NORMAL
        self.reset_vel()

    def reset_vel(self):
        self.vel = np.array([0, 0])

    def holein(self):
        self.state = BState.HOLEIN
        self.reset_vel()

    def is_normal(self):
        return self.state == BState.NORMAL

    def __repr__(self):
        return self.name

    def draw(self):
        glPushMatrix()
        if self.hit:
            glColor3f(0.5, 0.5, 0.5)
        else:
            glColor3f(*self.color)
        glTranslatef(self.pos[0], self.pos[1], 0)
        Ball.ball_circle.draw(GL_TRIANGLE_FAN)
        glPopMatrix()

    def dist(self, other):
        return self._dist(other.pos)

    def _dist(self, opos):
        x, y = self.pos
        ox, oy = opos
        return math.sqrt((x - ox) ** 2 + (y - oy) ** 2)

    def repel(self, other, dist):
        rd = (BALL_RAD * 2 - dist) * 0.5
        x, y = self.pos
        ox, oy = other.pos
        d = np.linalg.norm([x - ox, y - oy])
        if d > 0.0:
            rp = [x - ox, y - oy] / d * rd
            orp = [ox - x, oy - y] / d * rd
            self.pos += rp
            other.pos += orp
        return d

    def apply_hit_vel(self, other, dist):
        dists = dist ** 2
        x, y = self.pos
        ox, oy = other.pos
        vx, vy = self.vel
        ovx, ovy = other.vel

        f = np.linalg.norm(self.vel)
        of = np.linalg.norm(other.vel)

        nv = self.vel - np.dot(self.vel - other.vel, self.pos - other.pos) /\
            dists * (self.pos - other.pos)

        nov = self.vel - np.dot(other.vel - self.vel, other.pos - self.pos) /\
            dists * (other.pos - self.pos)

        nf = np.linalg.norm(nv)
        nof = np.linalg.norm(nov)

        fric_rate = HIT_FRIC_RATE2 if f < nf else HIT_FRIC_RATE
        ofric_rate = HIT_FRIC_RATE2 if of < nof else HIT_FRIC_RATE
        nv = decelerate(nv, fric_rate)
        nov = decelerate(nov, ofric_rate)

        self.vel = nv
        other.vel = nov

    def _check_wall_hit(self):
        if self.pos[0] < self.view.min_pos[0]:
            return True
        if self.pos[0] > self.view.max_pos[0]:
            return True
        if self.pos[1] < self.view.min_pos[1]:
            return True
        if self.pos[1] > self.view.max_pos[1]:
            return True
        return False

    def update(self):
        self.pos = self.pos + self.vel*FIX_DELTA

        hit_wall = False
        if self.pos[0] < self.view.min_pos[0]:
            self.vel[0] *= -1
            hit_wall = True
        if self.pos[0] > self.view.max_pos[0]:
            self.vel[0] *= -1
            hit_wall = True
        if self.pos[1] < self.view.min_pos[1]:
            self.vel[1] *= -1
            hit_wall = True
        if self.pos[1] > self.view.max_pos[1]:
            self.vel[1] *= -1
            hit_wall = True

        self.pos = np.clip(self.pos, self.view.min_pos, self.view.max_pos)
        if hit_wall:
            self.vel = decelerate(self.vel, WALL_FRIC_RATE)
        else:
            self.vel = decelerate(self.vel)


def decelerate(vel, fric_rate=DEFAULT_FRIC_RATE):
    f = np.linalg.norm(vel)
    if f > 0.0:
        nvel = vel / f
        vel = nvel * f * fric_rate
        if f < FORCE_EPS:
            vel = np.array([0, 0])
    return vel


def get_display(spec):
    if spec is None:
        return None
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error("Invalid display spec {}".format(spec))


class Viewer:
    hole_circle = make_circle(HOLE_RAD, HOLE_POINTS)

    def __init__(self, env, width, height, ball_info, hole_info, enc_output,
                 display=None, obs_image_skip=5, obs_woff=0, obs_hoff=0):
        display = get_display(display)
        self.env = env
        self.width = width
        self.height = height
        self.hwidth = int(width * 0.5)
        self.hheight = int(height * 0.5)
        self.obs_image_skip = obs_image_skip
        self.obs_hwoff = int(obs_woff * 0.5)
        self.obs_width = math.ceil(width / obs_image_skip) - obs_woff
        self.obs_hhoff = int(obs_hoff * 0.5)
        self.obs_height = math.ceil(height / obs_image_skip) - obs_hoff
        self.pixel_size = (self.obs_width - obs_woff) * (self.obs_height -
                                                         obs_hoff)
        self.window = pyglet.window.Window(width=width, height=height,
                                           display=display)
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_motion = self.on_mouse_motion
        self.window.on_draw = self.on_draw
        self.ball_info = ball_info
        self.num_ball = len(ball_info)
        self.enc_output = enc_output
        self.balls = []
        self.hit_list = []
        self.holein_list = []
        self.min_pos = np.array([WALL_DEPTH + BALL_RAD, WALL_DEPTH + BALL_RAD])
        self.max_pos = np.array([self.width - WALL_DEPTH - BALL_RAD,
                                 self.height - WALL_DEPTH - BALL_RAD])
        for i, bi in enumerate(ball_info):
            name = bi[0]
            color = bi[1]
            pos = bi[2]
            ball = Ball(self, name, i, pos, color)
            self.balls.append(ball)

        self.hole_info = hole_info
        self.drag_start = False

    def clear_shot_result(self):
        self.hit_list = []
        self.holein_list = []

    def reset_balls(self, ball_poss=None):
        for i, bi in enumerate(self.ball_info):
            pos = bi[2] if ball_poss is None else ball_poss[i]
            ball = self.balls[i]
            ball.pos = pos
            ball.reset()

    @property
    def cueball(self):
        return self.balls[0]

    def set_ball_vel(self, vel, ball_idx=0):
        self.balls[ball_idx].vel = vel

    def is_freeball(self):
        return self.cueball.state == BState.FREEBALL

    def is_holein(self):
        return self.cueball.state == BState.HOLEIN

    def start_freeball(self):
            self.cueball.state = BState.FREEBALL
            self.cueball.pos = self.hwidth, self.hheight

    def on_mouse_press(self, x, y, button, modifiers):
        if self.is_freeball():
            if self.is_valid_fb() and button == mouse.LEFT:
                self.set_freeball(x, y)
        else:
            self.drag_start = self.move_end()

    def on_mouse_release(self, x, y, button, modifiers):
        if not self.is_freeball():
            self.env.drag_shot()
            self.drag_start = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.drag_start:
            self.env.prepare_drag_shot(x, y)

    def on_mouse_motion(self, x, y, dx, dy):
        # print("Mouse Motion x {}, y {}, dx {}, dy {}".format(x, y, dx, dy))
        if self.is_freeball():
            self.cueball.pos = x, y

    def on_draw(self):
        self.render()

    def set_freeball(self, x, y):
        self.cueball.pos = x, y
        self.cueball.state = BState.NORMAL

    def draw_ball(self):
        for ball in self.balls:
            if ball.state == BState.HOLEIN:
                continue
            ball.draw()

    def draw_hole(self):
        if self.hole_info is None:
            return

        glColor3f(0, 0, 0)
        for hi in self.hole_info:
            glPushMatrix()
            glTranslatef(hi[0], hi[1], 0)
            Viewer.hole_circle.draw(GL_TRIANGLE_FAN)
            glPopMatrix()

    def is_valid_fb(self):
        cball = self.cueball

        if cball._check_wall_hit():
            return False

        for hi in self.hole_info:
            dist = cball._dist(hi)
            if dist < HOLEIN_DIST:
                return False

        for ball in self.balls:
            if ball is cball:
                continue
            dist = ball.dist(cball)
            if dist < BALL_RAD * 2:
                return False

        return True

    def draw_line(self):
        if self.is_freeball():
            if not self.is_valid_fb():
                self._draw_xline()
        elif self.env.drag_shot_info is not None:
            self._draw_shotline()

    def _draw_xline(self):
        glLineWidth(5)
        pyglet.gl.glColor4f(0, 0, 0, 1.0)
        start = self.cueball.pos + np.array([15, 15])
        end = self.cueball.pos + np.array([-15, -15])
        line = [int(i) for i in (*start, *end)]
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', line))
        start = self.cueball.pos + np.array([-15, 15])
        end = self.cueball.pos + np.array([15, -15])
        line = [int(i) for i in (*start, *end)]
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', line))

    def _draw_shotline(self):
        sdir, _, force = self.env.drag_shot_info
        sdir = np.array(sdir)
        nsdir = sdir / (np.linalg.norm(sdir) + 1)
        frate = force / self.env.div_of_force
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

        glLineWidth(5)
        pyglet.gl.glColor4f(frate, frate * 0.4, frate * 0.4, 0.8)
        start = self.cueball.pos + nsdir * BALL_RAD
        end = start + sdir
        line = [int(i) for i in (*start, *end)]
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', line))

        glDisable(GL_BLEND)
        pyglet.gl.glColor4f(0.6, 0.6, 0.6, 1.0)
        glLineWidth(1)
        start = self.cueball.pos - nsdir * BALL_RAD
        end = start - nsdir * SHOT_LINE_LEN
        line = [int(i) for i in (*start, *end)]
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', line))


    def draw_wall(self):
        # left
        l, t, r, b = [0, self.height, WALL_DEPTH, 0]
        v = ((l, b), (l, t), (r, t), (r, b))
        draw_polygon(v)

        # right
        l, t, r, b = [self.width-WALL_DEPTH, self.height,
                      self.width, 0]
        v = ((l, b), (l, t), (r, t), (r, b))
        draw_polygon(v)

        # bottom
        l, t, r, b = [0, WALL_DEPTH, self.width, 0]
        v = ((l, b), (l, t), (r, t), (r, b))
        draw_polygon(v)

        # top
        l, t, r, b = [0, self.height, self.width, self.height-WALL_DEPTH]
        v = ((l, b), (l, t), (r, t), (r, b))
        draw_polygon(v)

    def frame_move(self, dt=None):
        for ball in self.balls:
            if ball.state != BState.NORMAL:
                continue
            ball.update()

        # check collision
        hit = False
        for i in range(self.num_ball):
            cball = self.balls[i]
            if not cball.is_normal():
                continue

            for j in range(i, self.num_ball):
                ball = self.balls[j]
                if i == j or not ball.is_normal():
                    continue

                dist = cball.dist(ball)
                if dist < BALL_RAD * 2:
                    hit = True
                    if ball not in self.hit_list:
                        self.hit_list.append(ball)
                    cball.repel(ball, dist)
                    cball.apply_hit_vel(ball, dist)

        # check hole in
        holein = False
        for ball in self.balls:
            if not ball.is_normal():
                continue

            for hi in self.hole_info:
                dist = ball._dist(hi)
                if dist < HOLEIN_DIST:
                    ball.holein()
                    if ball not in self.holein_list:
                        self.holein_list.append(ball)
                    holein = True

        return hit, holein

    def move_end(self):
        zv = np.array([0, 0])
        for ball in self.balls:
            if not ball.is_normal():
                continue
            if not np.array_equal(ball.vel, zv):
                return False
        return True

    def draw_hud(self):
        hud = self.env._get_hud()
        if hud is None:
            return
        text, x, y = hud
        color = (255, 0, 0, 255) if 'Red' in text else (0, 0, 255, 255)
        label = pyglet.text.Label(text, font_name='Arial',
                                  font_size=13, x=x, y=y, anchor_x='left',
                                  anchor_y='bottom', color=color)
        label.draw()

    def render(self, return_rgb_array=False, window_flip=False):
        glClearColor(*SLATE_COLOR)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.draw_hole()
        self.draw_wall()
        self.draw_ball()
        self.draw_line()
        if not return_rgb_array:
            self.draw_hud()
        arr = None
        if return_rgb_array:
            arr = self._get_image()
        if window_flip:
            self.window.flip()
        return arr

    def close(self):
        self.window.close()

    def _get_image(self):
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')

        # preprocessing
        arr = arr.reshape(self.height, self.width, 4)
        arr = arr[::self.obs_image_skip, ::self.obs_image_skip, 0:3]
        if self.obs_hwoff > 0:
            arr = arr[self.obs_hwoff:-self.obs_hwoff, :]
        if self.obs_hhoff > 0:
            arr = arr[:, self.obs_hhoff:-self.obs_hhoff]
        arr = arr[::-1]

        if self.enc_output:
            # encode color
            return encode_rgb(arr)
        else:
            return arr

    def _get_obs(self, hud):
        return self.render(True, hud)

    def save_image(self, fname):
        arr = self._get_image()
        scipy.misc.imsave(fname, arr)

    def store_balls(self):
        self.balls_store = {}
        for ball in self.balls:
            self.balls_store[ball] = (ball.pos, ball.state)

    def restore_balls(self):
        for ball in self.balls:
            ball.pos = self.balls_store[ball][0]
            ball.state = self.balls_store[ball][1]
            ball.reset_vel()


def encode_rgb(arr):
    narr = np.empty([arr.shape[0], arr.shape[1]])
    for rgb, c in RGB_TO_BYTE.items():
        match = np.equal(arr, rgb).all(2)
        narr[match] = c
    return narr


def decode_rgb(arr):
    narr = np.empty([arr.shape[0], arr.shape[1], 3])
    for c, rgb in BYTE_TO_RGB.items():
        match = np.equal(arr, c)
        narr[match] = rgb
    return narr


def save_image(fname, arr, encoded, obs_depth):
    if encoded:
        s = decode_rgb(arr)
    else:
        s = arr
    scipy.misc.imsave(fname, s.reshape(s.shape[0], s.shape[1], obs_depth))
