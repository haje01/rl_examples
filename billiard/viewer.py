import math
import random

import numpy as np
import pyglet
from pyglet.window import mouse
from pyglet.gl import *  # NOQA


WIN_WIDTH = 750
WIN_HEIGHT = 400
WALL_DEPTH = 30
BALL_POINTS = 12
SLATE_COLOR = (0, 0.4, 0, 0)
WALL_COLOR = (0.6, 0.3, 0.1)
BALL_RAD = 12
FORCE_EPS = 25
FRICTION_RATE = 0.99
VEL_SHIFT = 100
MAX_VEL = 700
FIX_DELTA = 0.01667
NUM_BALL = 5
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
HIT_DECELEL_RATE = 0.5

window = pyglet.window.Window(width=WIN_WIDTH, height=WIN_HEIGHT)
circle = None
ball_pos = [WIN_WIDTH * 0.5, WIN_HEIGHT * 0.5]
ball_vel = [0, 0]


def make_circle(ptcnt):
    verts = []
    for i in range(ptcnt):
        angle = math.radians(float(i)/ptcnt * 360.0)
        x = BALL_RAD * math.cos(angle)
        y = BALL_RAD * math.sin(angle)
        verts += [x, y]
    return pyglet.graphics.vertex_list(ptcnt, ('v2f', verts))


class Ball(object):
    circle = make_circle(BALL_POINTS)

    def __init__(self, no, pos, color):
        self.no = no
        self.pos = pos
        self.color = color
        self.vel = [0, 0]
        self.hit = False

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
        rp = [x - ox, y - oy] / np.linalg.norm([x - ox, y - oy]) * rd
        orp = [ox - x, oy - y] / np.linalg.norm([ox - x, oy - y]) * rd
        self.pos += rp
        other.pos += orp

    def apply_hit_vel(self, other, dist):
        dists = dist ** 2
        x, y = self.pos
        ox, oy = other.pos
        vx, vy = self.vel
        ovx, ovy = other.vel
        nvx = vx - np.dot([vx - ovx], [x - ox]) / dists * (x - ox)
        novx = vx - np.dot([ovx - vx], [ox - x]) / dists * (ox - x)
        nvy = vy - np.dot([vy - ovy], [y - oy]) / dists * (y - oy)
        novy = vy - np.dot([ovy - vy], [oy - y]) / dists * (oy - y)
        self.vel = decelerate((nvx, nvy), HIT_DECELEL_RATE)
        other.vel = decelerate((novx, novy), HIT_DECELEL_RATE)

    def update(self):
        x, y = self.pos
        vx, vy = self.vel
        x, y = x + vx * FIX_DELTA, y + vy * FIX_DELTA

        x_min = WALL_DEPTH + BALL_RAD
        x_max = WIN_WIDTH - WALL_DEPTH - BALL_RAD
        y_min = WALL_DEPTH + BALL_RAD
        y_max = WIN_HEIGHT - WALL_DEPTH - BALL_RAD

        if x < x_min:
            x = x_min
            vx *= -1
        if x > x_max:
            x = x_max
            vx *= -1
        if y < y_min:
            y = y_min
            vy *= -1
        if y > y_max:
            y = y_max
            vy *= -1
        self.pos = [x, y]
        self.vel = [vx, vy]
        self.vel = decelerate([vx, vy])


def decelerate(vel, fric_rate=FRICTION_RATE):
    f = np.linalg.norm(vel)
    if f > 0.0:
        nvel = vel / f
        vel = nvel * f * fric_rate
        if f < FORCE_EPS:
            vel = [0, 0]
    return vel


balls = []
for i in range(NUM_BALL):
    pos = BALL_POS[i]
    color = BALL_COLOR[i]
    ball = Ball(i, pos, color)
    balls.append(ball)


@window.event
def on_mouse_press(x, y, button, modifiers):
    vx, vy = balls[0].vel
    if button == mouse.LEFT:
        vx = random.randint(-MAX_VEL, MAX_VEL)
        vy = random.randint(-MAX_VEL, MAX_VEL)
        vx += VEL_SHIFT if vx > 0 else -VEL_SHIFT
        vy += VEL_SHIFT if vy > 0 else -VEL_SHIFT
        balls[0].vel = (vx, vy)


@window.event
def on_mouse_release(x, y, button, modifiers):
    if button == mouse.LEFT:
        print("Left Button release, modifiers {}".format(modifiers))


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    print("Mouse Drag x {}, y {}, dx {}, dy {}, buttons {}".format(x, y, dx,
                                                                   dy,
                                                                   buttons))


@window.event
def on_mouse_motion(x, y, dx, dy):
    print("Mouse Motion x {}, y {}, dx {}, dy {}".format(x, y, dx, dy))


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


def draw_ball():
    for ball in balls:
        ball.draw()


def update(dt):
    for ball in balls:
        ball.update()

    # check collision
    for i in range(NUM_BALL):
        cball = balls[i]
        for j in range(i, NUM_BALL):
            if i == j:
                continue
            ball = balls[j]
            dist = cball.dist(ball)
            if dist < BALL_RAD * 2:
                cball.repel(ball, dist)
                cball.apply_hit_vel(ball, dist)


pyglet.clock.schedule(update)


@window.event
def on_draw():
    glClearColor(*SLATE_COLOR)
    window.clear()
    draw_wall()
    draw_ball()

pyglet.app.run()
