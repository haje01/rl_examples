import math
import random

import scipy.misc
import numpy as np
import pyglet
from pyglet.window import mouse
from pyglet.gl import *  # NOQA

WIN_WIDTH = 750
WIN_HEIGHT = 400
WALL_DEPTH = 30
BALL_POINTS = 12
SLATE_COLOR = (0, 0.4, 0, 1)
WALL_COLOR = (0.6, 0.3, 0.1)
BALL_RAD = 12
FORCE_EPS = 25
FRICTION_RATE = 0.99
HIT_FRICTION_RATE = 0.7
HIT_FRICTION_RATE2 = 0.3
MAX_VEL = 1400
FIX_DELTA = 0.01667
GET_IMAGE_SKIP = 6
CIRCLE_DEGREE = 360
DIV_OF_CIRCLE = 60
DIV_OF_FORCE = 10
ACTION_DEGREE = CIRCLE_DEGREE / DIV_OF_CIRCLE
OBS_WIDTH = math.ceil(WIN_WIDTH / GET_IMAGE_SKIP)
OBS_HEIGHT = math.ceil(WIN_HEIGHT / GET_IMAGE_SKIP)
OBS_DEPTH = 3
NUM_BALL = 5
BALL_NAME = [
    "Cue",
    "Red",
    "Yellow",
    "Blue",
    "Green",
]
BALL_COLOR = [
    (1, 1, 1),  # Cue
    (1, 0, 0),  # Red
    (1, 1, 0),  # Yellow
    (0, 0, 1),  # Blue
    (0, 1, 0),  # Green
]
BALL_POS = [
    (150, 200),    # Cue
#    (440, 300),    # Cue
    (450, 300),  # Red
    (450, 100),  # Yello
    (600, 300),  # Blue
    (600, 100),  # Green
]
NO_COL_DIST = 2
MAX_COL_REPEL = 30
CON_HIT_LIMIT = 1


# window = pyglet.window.Window(width=WIN_WIDTH, height=WIN_HEIGHT)
circle = None
ball_pos = [WIN_WIDTH * 0.5, WIN_HEIGHT * 0.5]
ball_vel = np.array([0, 0])


def make_circle(ptcnt):
    verts = []
    for i in range(ptcnt):
        angle = math.radians(float(i)/ptcnt * 360.0)
        x = BALL_RAD * math.cos(angle)
        y = BALL_RAD * math.sin(angle)
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


def draw_wall():
    # left
    l, t, r, b = [0, WIN_HEIGHT, WALL_DEPTH, 0]
    v = ((l, b), (l, t), (r, t), (r, b))
    draw_polygon(v)

    # right
    l, t, r, b = [WIN_WIDTH-WALL_DEPTH, WIN_HEIGHT, WIN_WIDTH, 0]
    v = ((l, b), (l, t), (r, t), (r, b))
    draw_polygon(v)

    # bottom
    l, t, r, b = [0, WALL_DEPTH, WIN_WIDTH, 0]
    v = ((l, b), (l, t), (r, t), (r, b))
    draw_polygon(v)

    # top
    l, t, r, b = [0, WIN_HEIGHT, WIN_WIDTH, WIN_HEIGHT-WALL_DEPTH]
    v = ((l, b), (l, t), (r, t), (r, b))
    draw_polygon(v)


class Ball:
    circle = make_circle(BALL_POINTS)
    min_pos = np.array([WALL_DEPTH + BALL_RAD, WALL_DEPTH + BALL_RAD])
    max_pos = np.array([WIN_WIDTH - WALL_DEPTH - BALL_RAD, WIN_HEIGHT -
                        WALL_DEPTH - BALL_RAD])

    def __init__(self, name, no, pos, color):
        self.name = name
        self.no = no
        self.pos = np.array(pos)
        self.color = color
        self.reset_vel()
        self.hit = False

    def reset_vel(self):
        self.vel = np.array([0, 0])

    def __repr__(self):
        return self.name

    def draw(self):
        glPushMatrix()
        if self.hit:
            glColor3f(0.5, 0.5, 0.5)
        else:
            glColor3f(*self.color)
        glTranslatef(self.pos[0], self.pos[1], 0)
        Ball.circle.draw(GL_TRIANGLE_FAN)
        glPopMatrix()

    def dist(self, other):
        x, y = self.pos
        ox, oy = other.pos
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

        fric_rate = HIT_FRICTION_RATE2 if f < nf else HIT_FRICTION_RATE
        ofric_rate = HIT_FRICTION_RATE2 if of < nof else HIT_FRICTION_RATE
        nv = decelerate(nv, fric_rate)
        nov = decelerate(nov, ofric_rate)

        self.vel = nv
        other.vel = nov

    def update(self):
        self.pos = self.pos + self.vel*FIX_DELTA

        if self.pos[0] < Ball.min_pos[0]:
            self.vel[0] *= -1
        if self.pos[0] > Ball.max_pos[0]:
            self.vel[0] *= -1
        if self.pos[1] < Ball.min_pos[1]:
            self.vel[1] *= -1
        if self.pos[1] > Ball.max_pos[1]:
            self.vel[1] *= -1
        self.pos = np.clip(self.pos, Ball.min_pos, Ball.max_pos)
        self.vel = decelerate(self.vel)


def decelerate(vel, fric_rate=FRICTION_RATE):
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

    def __init__(self, ball_name, ball_color, ball_pos, display=None):
        display = get_display(display)
        self.window = pyglet.window.Window(width=WIN_WIDTH, height=WIN_HEIGHT,
                                           display=display)
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_drag = self.on_mouse_drag
        self.window.on_mouse_motion = self.on_mouse_motion
        self.window.on_draw = self.on_draw
        self.numball = len(ball_color)
        assert self.numball == len(ball_pos)
        self.balls = []
        self.hit_list = []
        for i in range(self.numball):
            name = ball_name[i]
            pos = ball_pos[i]
            color = ball_color[i]
            ball = Ball(name, i, pos, color)
            self.balls.append(ball)

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self.random_shot()

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            print("Left Button release, modifiers {}".format(modifiers))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        print("Mouse Drag x {}, y {}, dx {}, dy {}, buttons {}".format(x, y,
                                                                       dx, dy))

    def on_mouse_motion(self, x, y, dx, dy):
        print("Mouse Motion x {}, y {}, dx {}, dy {}".format(x, y, dx, dy))

    def on_draw(self):
        self.render()

    def shot(self, action):
        deg, force = action
        self.hit_list = []
        deg *= 360 / DIV_OF_CIRCLE
        rad = math.radians(deg)
        force *= 1 / DIV_OF_FORCE
        vel = force * MAX_VEL
        vx = math.sin(rad) * vel
        vy = math.cos(rad) * vel
        #print("shot: deg {}, force {:.2f}, vx {:.2f}, vy {:.2f}".format(deg,
        #                                                                force,
        #                                                                vx,
        #                                                                vy))
        self.balls[0].vel = np.array([vx, vy])

    def random_shot(self):
        deg = random.randint(0, DIV_OF_CIRCLE)
        force = random.randint(0, DIV_OF_FORCE)
        self.shot((deg, force))

    def draw_ball(self):
        for ball in self.balls:
            ball.draw()

    def frame_move(self, dt=None):
        for ball in self.balls:
            ball.update()

        # check collision
        for i in range(self.numball):
            cball = self.balls[i]
            for j in range(i, self.numball):
                if i == j:
                    continue
                ball = self.balls[j]
                dist = cball.dist(ball)
                # hit test
                if dist < BALL_RAD * 2:
                    if ball not in self.hit_list:
                        self.hit_list.append(ball)
                    cball.repel(ball, dist)
                    # cball.vel = [0, 0]
                    # ball.vel = [0, 0]
                    cball.apply_hit_vel(ball, dist)

    def move_end(self):
        zv = np.array([0, 0])
        for ball in self.balls:
            if not np.array_equal(ball.vel, zv):
                return False
        return True

    def render(self, return_rgb_array=False, window_flip=False):
        glClearColor(*SLATE_COLOR)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        draw_wall()
        self.draw_ball()
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
        arr = arr.reshape(WIN_HEIGHT, WIN_WIDTH, 4)
        arr = arr[::GET_IMAGE_SKIP, ::GET_IMAGE_SKIP, 0:3]
        return arr

    def _get_obs(self):
        return self._get_image()

    def save_image(self, fname):
        arr = self._get_image()
        scipy.misc.imsave(fname, arr)

    def shot_and_get_result(self, action):
        self.shot(action)
        while True:
            self.frame_move()
            if self.move_end():
                break
        self.render(True)
        return self.hit_list[:], self._get_obs()

    def store_balls(self):
        self.balls_store = {}
        for ball in self.balls:
            self.balls_store[ball] = ball.pos

    def restore_balls(self):
        for ball in self.balls:
            ball.pos = self.balls_store[ball]
            ball.reset_vel()

if __name__ == "__main__":
    viewer = Viewer(BALL_NAME, BALL_COLOR, BALL_POS)
    pyglet.clock.schedule(viewer.frame_move)
    viewer.shot((15, 5))
    pyglet.app.run()
