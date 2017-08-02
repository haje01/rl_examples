import time
import math
from collections import namedtuple

import matplotlib.path as mplPath
import numpy as np
from numpy import array
import Box2D
from Box2D import (b2World, b2BodyDef, b2FixtureDef, b2Body, b2CircleShape,
    b2Vec2, b2PolygonShape)

import pyglet
from pyglet.window import mouse
from pyglet.gl import *  # NOQA

Point = namedtuple('Point', ['x', 'y'])
RGB = namedtuple('RGB', ['r', 'g', 'b'])
RGBA = namedtuple('RGBA', ['r', 'g', 'b', 'a'])
Body = namedtuple('Body', ['x', 'y', 'angle', 'center'])

SCALE = 30.0

ball_to_hit = None
ball_to_move = None
mouse_down_pt = None
MAX_IMPULSE = 25
rm = None

CANVAS_WIDTH = 700 # 750
CANVAS_HEIGHT = 385 # 440
BALL_RADIUS = 0.4
CUE_START = array((508, 192)) / SCALE
CUE_LINE_WIDTH = 5

# first ball starting point
SP = array((192, 191)) / SCALE

# ball exit
BE = array((70, 400)) / SCALE

GOAL_PTS = [array([34, 35]), array([350, 25]), array([664, 35]),
            array([664, 350]), array([350, 360]), array([35, 350])]

for i, gp in enumerate(GOAL_PTS):
    GOAL_PTS[i] = gp / SCALE


STICK_START_COLOR = array((254, 232, 214))
STICK_END_COLOR = array((255, 0, 0))
world = {}
worker = None
bodies_state = None
box = None

CLEAR_COLOR = (0.5, 0.5, 0.5, 1)

class Entity:
    def __init__(self, id, pos, static_body=False, hidden=False):
        self.id = id
        self.pos = array(pos, dtype='float64')
        self.angle = 0
        self.center = 0
        self.restitution = 0.3
        self.density = 1.0
        self.friction = 0.9
        self.linear_damping = 0
        self.angular_damping = 0
        self.static_body = static_body
        self.color = RGBA(128, 128, 128, 0.5)
        self.hidden = hidden

    def update(self, state):
        self.pos[0] = state.x
        self.pos[1] = state.y
        self.angle = state.angle


class CircleEntity(Entity):
    def __init__(self, id, pos, radius=1):
        super(CircleEntity, self).__init__(id, pos)
        self.radisu = radius

    def draw(self, ctx, scale):
        ctx.fill_style = self.color


class PolygonEntity(Entity):
    def __init__(self, id, pos, points, static_body, hidden):
        super(PolygonEntity, self).__init__(id, pos, static_body=static_body,
                                            hidden=hidden)
        self.points = array(points, dtype='float64')


class RectangleEntity(Entity):
    def __init__(self, id, pos, half_width=1, half_height=1,
                 static_body=False, hidden=False):
        super(RectangleEntity, self).__init__(id, pos,
                                              static_body=static_body,
                                              hidden=hidden)
        self.half_width = half_width
        self.half_height = half_height


class Box:
    def __init__(self, interval_rate=60, adaptive=False, width=640, height=480,
                 scale=30, gravity_x=0, gravity_y=10):
        self.interval_rate = interval_rate
        self.adaptive = adaptive
        self.width = width
        self.height = height
        self.scale = scale
        self.bodies_map = {}
        self.fixtures_map = []
        self.world = None
        self.gravity_x = gravity_x
        self.gravity_y = gravity_y
        self.allow_sleep = True
        self.world = b2World(b2Vec2(self.gravity_x, self.gravity_y), self.allow_sleep)

    def update(self):
        start = time.time()
        step_rate = 1 / self.interval_rate
        self.world.Step(step_rate,  # frame rate
                        10,  # velocity iterations
                        10)  # position iterations
        self.world.ClearForces()
        return time.time() - start

    def get_state(self):
        state = {}
        for i, b in enumerate(self.world.bodies):
            if b.active and b.userData is not None:
                body = Body(x=b.position.x, y=b.position.y, angle=b.angle,
                            center=Point(x=b.worldCenter.x, y=b.worldCenter.y))
                state[b.userData] = body

        return state

    def set_bodies(self, body_entities):
        for id, entity in body_entities.items():
            self.add_body(entity)
        self.ready = True

    def add_body(self, entity):
        body_def = b2BodyDef()
        fix_def = b2FixtureDef()
        fix_def.restitution = entity.restitution
        fix_def.density = entity.density
        fix_def.friction = entity.friction

        if entity.static_body:
            body_def.type = Box2D.b2_staticBody
        else:
            body_def.type = Box2D.b2_dynamicBody

        if hasattr(entity, 'radius'):
            fix_def.shape = b2CircleShape(radius=entity.radius)
        elif hasattr(entity, 'points'):
            fix_def.shape = b2PolygonShape()
            vertices = []
            for i, pt in enumerate(entity.points):
                vec = b2Vec2()
                vec.Set(pt[0], pt[1])
                vertices.append(vec)
            fix_def.shape.vertices = vertices
        else:
            fix_def.shape = b2PolygonShape()
            fix_def.shape.SetAsBox(entity.half_width, entity.half_height)

        body_def.position.x = float(entity.pos[0])
        body_def.position.y = float(entity.pos[1])
        body_def.userData = entity.id
        body_def.linearDamping = entity.linear_damping
        body_def.angularDamping = entity.angular_damping
        self.bodies_map[entity.id] = self.world.CreateBody(body_def)
        self.bodies_map[entity.id].CreateFixture(fix_def)


    def apply_impulse(self, body_id, degrees, power):
        body = self.bodies_map[body_id]
        body.ApplyLinearImpulse(b2Vec2(math.cos(degrees * (math.pi / 180)) * power,
                                      math.sin(degrees * (math.pi / 180)) * power),
                                      body.worldCenter, True)

    def remove_body(self, id):
        if id in self.bodies_map:
            self.world.DestroyBody(self.bodies_map[id])


def arc_poss(verts, r, start_deg, arc_deg, num_seg):
    theta = math.radians(arc_deg) / float(num_seg - 1)
    tan_fact = math.tan(theta)
    rad_fact = math.cos(theta)
    start = math.radians(start_deg)
    x = r * math.cos(start)
    y = r * math.sin(start)

    for i in range(num_seg):
        verts += [x, y]
        tx = -y
        ty = x
        x += tx * tan_fact
        y += ty * tan_fact
        x *= rad_fact
        y *= rad_fact


def hex_to_rgb(hex):
    return tuple(int(hex[i+1:i+3], 16) for i in (0, 2 ,4))


class Ball(CircleEntity):
    @staticmethod
    def make_solid_verts(r, color):
        poss = []
        arc_poss(poss, r, 0, 360, 54)
        n_verts = int(len(poss) / 2)
        colors = color * n_verts
        return pyglet.graphics.vertex_list(n_verts, ('v2f', poss),
                                           ('c3B', colors))

    @staticmethod
    def make_stripe_verts(r, color):
        poss = []
        stripe_deg = 40
        arc_poss(poss, r, -stripe_deg, stripe_deg * 2, 6)
        arc_poss(poss, r, 180 - stripe_deg, stripe_deg * 2, 6)
        n_verts = int(len(poss) / 2)
        colors = color * n_verts
        return pyglet.graphics.vertex_list(n_verts, ('v2f', poss),
                                           ('c3B', colors))

    def __init__(self, id, color, striped, pos=(0, 0)):
        super(Ball, self).__init__(id, pos)
        color = hex_to_rgb(color)
        self.color = color
        self.radius = BALL_RADIUS
        self.striped = striped
        if not striped:
            self.solid_verts = self.make_solid_verts(self.radius * SCALE, color)
        else:
            self.solid_verts = self.make_solid_verts(self.radius * SCALE,
                                                     [255, 255, 255])
            self.stripe_verts = self.make_stripe_verts(self.radius * SCALE, color)
        self.text = pyglet.text.Label(str(id), font_size=10,
                                      color=(255, 255, 255, 255), x=-1, y=1,
                                      anchor_x='center', anchor_y='center')

    def draw(self):
        glPushMatrix()
        glLoadIdentity()
        pos = self.pos * SCALE
        glTranslatef(pos[0], pos[1], 0)
        glRotatef(self.angle * 180 / math.pi, 0, 0, 1)
        self.solid_verts.draw(GL_TRIANGLE_FAN)
        if self.striped:
            self.stripe_verts.draw(GL_TRIANGLE_FAN)
        self.text.draw()
        glPopMatrix()


eight_ball_locs = [
    [*CUE_START],
    [*SP],
    [SP[0] - (2 * BALL_RADIUS), SP[1] - BALL_RADIUS],
    [SP[0] - (4 * BALL_RADIUS), SP[1] + (2 * BALL_RADIUS)],
    [SP[0] - (6 * BALL_RADIUS), SP[1] - (3 * BALL_RADIUS)],
    [SP[0] - (8 * BALL_RADIUS), SP[1] + (4 * BALL_RADIUS)],
    [SP[0] - (8 * BALL_RADIUS), SP[1] - (2 * BALL_RADIUS)],
    [SP[0] - (6 * BALL_RADIUS), SP[1] + BALL_RADIUS],
    [SP[0] - (4 * BALL_RADIUS), SP[1]],
    [SP[0] - (2 * BALL_RADIUS), SP[1] + BALL_RADIUS],
    [SP[0] - (4 * BALL_RADIUS), SP[1] - (2 * BALL_RADIUS)],
    [SP[0] - (6 * BALL_RADIUS), SP[1] + (3 * BALL_RADIUS)],
    [SP[0] - (8 * BALL_RADIUS), SP[1] - (4 * BALL_RADIUS)],
    [SP[0] - (8 * BALL_RADIUS), SP[1] + (2 * BALL_RADIUS)],
    [SP[0] - (6 * BALL_RADIUS), SP[1] - BALL_RADIUS],
    [SP[0] - (8 * BALL_RADIUS), SP[1]]
]


initial_state = [
    Ball(id=0, color="#FFFFFF", striped=False),
    Ball(id=1, color="#DDDD00", striped=False),
    Ball(id=2, color="#0000CC", striped=False),
    Ball(id=3, color="#FF0000", striped=False),
    Ball(id=4, color="#880088", striped=False),
    Ball(id=5, color="#FF6600", striped=False),

    Ball(id=6, color="#007700", striped=False),
    Ball(id=7, color="#770000", striped=False),
    Ball(id=8, color="#000000", striped=False),
    Ball(id=9, color="#DDDD00", striped=True),
    Ball(id=10, color="#0000CC", striped=True),

    Ball(id=11, color="#FF0000", striped=True),
    Ball(id=12, color="#880088", striped=True),
    Ball(id=13, color="#FF6600", striped=True),
    Ball(id=14, color="#007700", striped=True),
    Ball(id=15, color="#770000", striped=True),

    PolygonEntity(id=16, pos=(0, 0), points=[(60, 35), (25, 0), (338, 0), (330, 35)], static_body=True, hidden=True),
    PolygonEntity(id=17, pos=(0, 0), points=[(369, 35), (362, 0), (675, 0), (639, 35)], static_body=True, hidden=True),

    PolygonEntity(id=18, pos=(0, 0), points=[(664, 60), (700, 24), (700, 362), (664, 324)], static_body=True, hidden=True),

    PolygonEntity(id=19, pos=(0, 0), points=[(370, 349), (638, 349), (675, 385), (362, 385)], static_body=True, hidden=True),
    PolygonEntity(id=20, pos=(0, 0), points=[(60, 349), (330, 349), (338, 385), (25, 385)], static_body=True, hidden=True),

    PolygonEntity(id=21, pos=(0, 0), points=[(0, 35), (35, 60), (35, 324), (0, 361)], static_body=True, hidden=True),

    PolygonEntity(id=22, pos=(0, 0), points=[(9, 9), (31, 15), (11, 32)], static_body=True, hidden=True),
    PolygonEntity(id=23, pos=(0, 0), points=[(350, 1), (360, 13), (339, 13)], static_body=True, hidden=True),
    PolygonEntity(id=24, pos=(0, 0), points=[(695, 5), (684, 30), (667, 13)], static_body=True, hidden=True),

    PolygonEntity(id=25, pos=(0, 0), points=[(696, 381), (665, 369), (685, 355)], static_body=True, hidden=True),
    PolygonEntity(id=26, pos=(0, 0), points=[(349, 382), (338, 371), (361, 371)], static_body=True, hidden=True),
    PolygonEntity(id=27, pos=(0, 0), points=[(3, 381), (15, 355), (30, 371)], static_body=True, hidden=True),

    RectangleEntity(id=28, pos=(30/SCALE, 415/SCALE), half_height=30/SCALE, half_width=30/SCALE, static_body=True, hidden=True),
    RectangleEntity(id=29, pos=(670/SCALE, 415/SCALE), half_height=30/SCALE, half_width=30/SCALE, static_body=True, hidden=True),
    RectangleEntity(id=30, pos=(350/SCALE, 430/SCALE), half_height=15/SCALE, half_width=290/SCALE, static_body=True, hidden=True)
  ];


def is_point_in_poly(poly, pt):
    path_pts = np.array([[poly[i], poly[i+1]] for i in range(len(poly) - 1)])
    plpath = mplPath.Path(path_pts)
    return plpath.contains_point(pt)


def get_color_fade(start, end, percent):
    return ((end - start) * percent).astype(int) + start


def intersect(s1, s2, radii_squared):
    distance_squared = np.linalg.norm(s1 - s2)
    return distance_squared < radii_squared


def get_gfx_mouse(x, y):
    return array((x, y)) / SCALE


def get_collided_ball(mouse):
    for k, entity in world.items():
        if intersect(mouse, entity.pos, 0.5):
            return entity
    return None


def get_degrees(center, pt):
    if np.array_equal(center, pt):
        return 0
    elif center[0] == pt[0]:
        if center[1] < pt[1]:
            return 180
        else:
            return 0
    elif center[1] == pt[1]:
        if center[0] > pt[0]:
            return 270
        else:
            return 90
    elif center[0] < pt[0] and center[1] > pt[1]:
        # quadrant 1
        return math.atan((pt[0] - center[0])/(center[1] - pt[1])) * (180 / math.pi)
    elif center[0] < pt[0] and center[1] < pt[1]:
        # quadrant 2
        return 90 + math.atan((pt[1] - center[1])/(pt[0] - center[0])) * (180 / math.pi)
    elif center[0] > pt[0] and center[1] < pt[1]:
        # quadrant 3
        return 180 + math.atan((center[0] - pt[0])/(pt[1] - center[1])) * (180 / math.pi)
    else:
        # quadrant 4
        return 270 + math.atan((center[1] - pt[1])/(center[0] - pt[0])) * (180 / math.pi)


def get_distance(a, b):
    return np.linalg.norm(a - b)


def pt_on_table(pt):
    return (pt.x > (38/SCALE) and pt.x < (661/SCALE)) and (pt.y > (38/SCALE) and pt.y < (347/SCALE))


#def clear_table():
    #for i in range(16):
        #entity = world[i]
        #box.remove_body(i)
        #entity.y = BE.y
        #entity.x = BE.x + (2 * i * BALL_RADIUS) + (2 * BALL_RADIUS)
        #box.add_body(entity)


def rack_8ball():
    for i in range(16):
        entity = world[i]
        box.remove_body(i)
        entity.pos = array(eight_ball_locs[i])
        entity.on_table = True
        box.add_body(entity)


def draw_line(start, end, color, width, alpha=0.8):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glLineWidth(width)
    pyglet.gl.glColor4f(*(color / 255.0), alpha)
    line = [int(i) for i in (*start, *end)]
    pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', line))

    glDisable(GL_BLEND)
    pyglet.gl.glColor4f(1, 1, 1, 1)


class GameCore:
    def __init__(self):
        self.width = CANVAS_WIDTH
        self.height = CANVAS_HEIGHT
        self.is_running = False

    #def update(self, elapsed_time):
        #box.update()
        #bodies_state = box.get_state()
        #for bid in bodies_state:
            #entity = world[bid]
            #if entity is not None:
                #entity.update(bodies_state)
                #for pt in GOAL_PTS:
                    #if intersect(entity, pt, 0.2):
                        #entity.dead = True
                        #try:
                            #box.remove_body(bid)
                            #entity.pos = BE.pos
                            #entity.on_table = False
                            #box.add_body(entity)
                            #box.apply_impulse(bid, 0, 2)
                        #except Exception as e:
                            #print(e)

    def render(self):
        self.window.clear()
        glClearColor(*CLEAR_COLOR)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.back_img.blit(0, 0)

        if ball_to_hit is not None and mouse_down_pt is not None:
            imp_perc = min(get_distance(ball_to_hit.pos, mouse_down_pt) * 3,
                           MAX_IMPULSE) * 1.0 / MAX_IMPULSE
            color_fade = get_color_fade(STICK_START_COLOR, STICK_END_COLOR, imp_perc)
            draw_line(ball_to_hit.pos * SCALE, mouse_down_pt * SCALE,
                      color_fade, CUE_LINE_WIDTH)

        for k, entity in world.items():
            if not entity.hidden:
                entity.draw()

        self.window.switch_to()
        self.window.dispatch_events()
        self.window.flip()

    def init(self):
        self.canvas = None
        config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        self.window = pyglet.window.Window(width=self.width,
                                           height=self.height, display=None,
                                           config=config, resizable=True)
        self.window.on_draw = self.on_draw
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_drag = self.on_mouse_drag

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        global mouse_down_pt
        if mouse_down_pt is not None:
            mouse_down_pt = get_gfx_mouse(x, y)

    def on_mouse_press(self, x, y, button, modifiers):
        global selected_ball, mouse_down_pt, ball_to_hit

        for k, entity in world.items():
            entity.selected = False

        pt = get_gfx_mouse(x, y)
        mouse_down_pt = pt
        selected_ball = get_collided_ball(pt)
        if selected_ball is not None:
            if selected_ball.on_table:
                ball_to_hit = selected_ball
                ball_to_hit.selected = True

    def on_mouse_release(self, x, y, button, modifiers):
        global ball_to_move, ball_to_hit

        pt = get_gfx_mouse(x, y)

        if ball_to_hit is not None:
            degrees = get_degrees(ball_to_hit.pos, pt)
            box.apply_impulse(ball_to_hit.id, degrees + 90,
                              min(get_distance(ball_to_hit.pos, pt) * 3,
                                  MAX_IMPULSE))
            ball_to_hit = None

        if ball_to_move is not None:
            if pt_on_table(pt):
                box.remove_body(ball_to_move.id)
                ball_to_move.pos = pos
                ball_to_move.on_table = True
                box.add_body(ball_to_move)
            ball_to_move = None



    def load_resources(self, canvas):
        self.back_img = pyglet.image.load('pool_table_700x385.png')

    def run(self):
        self.init()
        self.load_resources(self.canvas)
        self.launch_loop()
        while True:
            self.render()
            box.update()
            bodies_state = box.get_state()
            for bid in bodies_state:
                entity = world[bid]
                if entity is not None:
                    entity.update(bodies_state[bid])
                    for pt in GOAL_PTS:
                        if intersect(entity.pos, pt, 0.2):
                            entity.dead = True
                            try:
                                box.remove_body(bid)
                                entity.pos = BE
                                entity.on_table = False
                                box.add_body(entity)
                                box.apply_impulse(bid, 0, 2)
                            except Exception as e:
                                print(e)


    def stop(self):
        self.is_running = False

    def launch_loop(self):
        self.elapsed_time = 0
        start_time = time.time()
        self.curr_time = start_time
        self.prev_time = start_time

    def on_draw(self):
        self.render()


def main():
    global game, box
    for i, iS in enumerate(initial_state):
        if i < 16:
            iS.radius = BALL_RADIUS
            iS.linear_damping = 0.7
            iS.angular_damping = 0.5
            iS.restitution = 0.9

        if hasattr(iS, 'points'):
            for j in range(len(iS.points)):
                iS.points[j] = iS.points[j] / SCALE

        if i > 15 and i < 28:
            iS.pos = array([0, 0])

        world[i] = iS

    game = GameCore()

    box = Box(
        interval_rate=80,
        adaptive=False,
        width=game.width,
        height=game.height,
        scale=SCALE,
        gravity_y=0
    )
    box.set_bodies(world)

    rack_8ball()
    game.run()


if __name__ == '__main__':
    main()
