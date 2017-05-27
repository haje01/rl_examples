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
max_impulse = 25
rm = None

shot = 0

canvas_width = 700 # 750
canvas_height = 385 # 440
table_height = 385
ball_radius = 0.4
cue_start = array((508, 192)) / SCALE

# first ball starting point
sp = array((192, 191)) / SCALE

# ball exit
be = array((70, 400)) / SCALE

goal_pts = [array([34, 35]), array([350, 25]), array([664, 35]),
            array([664, 350]), array([350, 360]), array([35, 350])]

for gp in goal_pts:
    gp = gp / SCALE


stick_start_color = RGB(r=254, g=232, b=214)
stick_end_color = RGB(r=255, g=0, b=0)
stick_distance = 0
world = {}
worker = None
bodies_state = None
box = None

clear_color = (0.5, 0.5, 0.5, 1)

class Entity:
    def __init__(self, id, pos, static_body=False, hidden=False):
        self.id = id
        self.pos = array(pos)
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
        self.points = array(points)


class RectangleEntity(Entity):
    def __init__(self, id, pos, half_width=1, half_height=1,
                 static_body=False, hidden=False):
        super(RectangleEntity, self).__init__(id, pos,
                                              static_body=static_body,
                                              hidden=hidden)
        self.half_width = half_width
        self.half_height = half_height


class Box:
    def __init__(self, interval_rate, adaptive, width, height, scale, gravity_y):
        self.interval_rate = 60
        self.adaptive = False
        self.width = 640
        self.height = 480
        self.scale = 30
        self.bodies_map = {}
        self.fixtures_map = []
        self.world = None
        self.gravity_x = 0
        self.gravity_y = 10
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
        for b in self.world.bodies_gen():
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
            vec = b2Vec2()
            for i, pt in enumerate(entity.points):
                vec.Set(float(pt[0]), float(pt[1]))
                fix_def.shape.set_vertex(i, vec)
        else:
            fix_def.shape = b2PolygonShape()
            fix_def.shape.SetAsBox(entity.half_width, entity.half_height)

        body_def.position.x = float(entity.pos[0])
        body_def.position.y = float(entity.pos[1])
        body_def.user_data = entity.id
        body_def.linear_damping = entity.linear_damping
        body_def.angular_damping = entity.angular_damping
        self.bodies_map[entity.id] = self.world.CreateBody(body_def)
        self.bodies_map[entity.id].CreateFixture(fix_def)


    def apply_impulse(body_id, degrees, power):
        body = self.bodies_map[body_id]
        body.ApplyImpulse(b2Vec2(math.cos(degrees * (math.pi / 180)) * power,
                                 math.sin(degrees * (math.pi / 180)) * power,
                                 body.GetWorldCenter()))

    def remove_body(self, id):
        if id in self.bodies_map:
            self.world.DestroyBody(self.bodies_map[id])
            del self.bodies_map[id]


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
        self.radius = ball_radius
        self.striped = striped
        if not striped:
            self.solid_verts = self.make_solid_verts(self.radius * SCALE, color)
        else:
            self.solid_verts = self.make_solid_verts(self.radius * SCALE,
                                                     [255, 255, 255])
            self.stripe_verts = self.make_stripe_verts(self.radius * SCALE, color)

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0] * SCALE, self.pos[1] * SCALE, 0)
        self.solid_verts.draw(GL_TRIANGLE_FAN)
        if self.striped:
            self.stripe_verts.draw(GL_TRIANGLE_FAN)
        glPopMatrix()


eight_ball_locs = [
    [*cue_start],
    [*sp],
    [sp[0] - (2 * ball_radius), sp[1] - ball_radius],
    [sp[0] - (4 * ball_radius), sp[1] + (2 * ball_radius)],
    [sp[0] - (6 * ball_radius), sp[1] - (3 * ball_radius)],
    [sp[0] - (8 * ball_radius), sp[1] + (4 * ball_radius)],
    [sp[0] - (8 * ball_radius), sp[1] - (2 * ball_radius)],
    [sp[0] - (6 * ball_radius), sp[1] + ball_radius],
    [sp[0] - (4 * ball_radius), sp[1]],
    [sp[0] - (2 * ball_radius), sp[1] + ball_radius],
    [sp[0] - (4 * ball_radius), sp[1] - (2 * ball_radius)],
    [sp[0] - (6 * ball_radius), sp[1] + (3 * ball_radius)],
    [sp[0] - (8 * ball_radius), sp[1] - (4 * ball_radius)],
    [sp[0] - (8 * ball_radius), sp[1] + (2 * ball_radius)],
    [sp[0] - (6 * ball_radius), sp[1] - ball_radius],
    [sp[0] - (8 * ball_radius), sp[1]]
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
    r = math.floor((end.r - start.r) * percent) + start.r
    g = math.floor((end.g - start.g) * percent) + start.g
    b = math.floor((end.b - start.b) * percent) + start.b
    return dict(r=r, g=g, b=b)


def intersect(s1, s2, radii_squared):
    distance_squared = np.linalg.norm(s1, s2)
    # math.pow(s1.x, s2.x, 2) - math.pow(s1.y - s2.y, 2)
    return distance_squared < radii_squared


def get_collided_ball(mouse):
    for i in range(len(initial_state)):
        if intersect(mouse, world[i], 0.5):
            return world[i]
    return None


def get_degree(center, pt):
    if center.x == pt.x and center.y == pt.y:
        return 0
    elif center.x == pt.x:
        if center.y < pt.y:
            return 180
        else:
            return 0
    elif center.y == pt.y:
        if center.x > pt.x:
            return 270
        else:
            return 90
    elif center.x < pt.x and center.y > pt.y:
        # quadrant 1
        return math.atan((pt.x - center.x)/(center.y - pt.y)) * (180 / math.pi)
    elif center.x < pt.x and center.y < pt.y:
        # quadrant 2
        return 90 + math.atan((pt.y - center.y)/(pt.x - center.x)) * (180 / math.pi)
    elif center.x > pt.x and center.y < pt.y:
        # quadrant 3
        return 180 + math.atan((center.x - pt.x)/(pt.y - center.y)) * (180 / math.pi)
    else:
        # quadrant 4
        return 270 + math.atan((center.y - pt.y)/(center.x - pt.x)) * (180 / math.pi)


def get_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)


def pt_on_table(pt):
    return (pt.x > (38/SCALE) and pt.x < (661/SCALE)) and (pt.y > (38/SCALE) and pt.y < (347/SCALE))


def clear_table():
    for i in range(16):
        entity = world[i]
        box.remove_body(i)
        entity.y = be.y
        entity.x = be.x + (2 * i * ball_radius) + (2 * ball_radius)
        box.add_body(entity)


def rack_8ball():
    for i in range(16):
        entity = world[i]
        box.remove_body(i)
        entity.pos = eight_ball_locs[i]
        entity.on_table = True
        box.add_body(entity)


class GameCore:
    def __init__(self):
        self.width = canvas_width
        self.height = canvas_height
        self.is_running = False

    def update(self, elapsed_time):
        box.update()
        bodies_state = box.get_state()
        for bid in bodies_state:
            entity = world[id]
            if entity is not None:
                entity.update(bodies_state)
                for pt in goal_pts:
                    if intersect(entity, pt, 0.2):
                        entity.dead = True
                        try:
                            box.remove_body(bid)
                            entity.pos = be.pos
                            entity.on_table = False
                            box.add_body(entity)
                            box.apply_impulse(bid, 0, 2)
                        except Exception as e:
                            print(e)

    def render(self):
        self.window.clear()
        glClearColor(*clear_color)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.back_img.blit(0, 0)

        for k, entity in world.items():
            if not entity.hidden:
                entity.draw()

        self.window.switch_to()
        self.window.dispatch_events()
        self.window.flip()

    def draw(self, ctx):
        ctx.line_width = 1
        ctx.clear_rect(0, 0, this.width, this.height)
        ctx.draw_image(back_img, 0, 0, self.width, table_height)

        if ball_to_hit and mouse_down_pt:
            imp_perc = math.min(get_distance(ball_to_hit, mouse_down_pt) * 3,
                                max_impulse) * 1.0 / max_impulse
            color_fade = get_color_fade(stick_start_color, stick_end_color, imp_perc)
            line_width = ctx.line_width
            ctx.line_width = 3
            ctx.begin_path()
            ctx.move_to(ball_to_hit * SCALE)
            ctx.line_to(mouse_down_pt * SCALE)
            ctx.stroke_style = RGB(r=color_fade.r, g=color_fade.g, b=color_fade.b)
            ctx.stroke()
            ctx.close_path()
            ctx.line_width = line_width

        for entity in world:
            if not entity.hidden:
                entity.draw(ctx, SCALE)

        if ball_to_move and mouse_down_pt:
            line_width = ctx.line_width
            ctx.line_width = 5
            ctx.stroke_style = RGB(0, 0, 0, 0.5)
            ctx.arc(mouse_down_pt * SCALE, ball_radius * SCALE, 0, math.pi * 2, True)
            ctx.close_path()
            ctx.stroke()
            ctx.line_width = line_width

    def init(self):
        self.canvas = None
        config = pyglet.gl.Config(sample_buffers=1, samples=4, double_buffer=True)
        self.window = pyglet.window.Window(width=self.width,
                                           height=self.height, display=None,
                                           config=config, resizable=True)
        self.window.on_draw = self.on_draw

    def load_resources(self, canvas):
        self.back_img = pyglet.image.load('pool_table_700x385.png')

    def run(self):
        self.init()
        self.load_resources(self.canvas)
        self.launch_loop()
        while True:
            self.render()

    def stop(self):
        self.is_running = False

    def launch_loop(self):
        self.elapsed_time = 0
        start_time = time.time()
        self.curr_time = start_time
        self.prev_time = start_time

    def on_draw(self, scale):
        self.render()


def main():
    global game, box
    for i, iS in enumerate(initial_state):
        if i < 16:
            iS.radius = ball_radius
            iS.linear_damping = 0.6
            iS.angular_damping = 0.5
            iS.reset_shots = 0.9

        if hasattr(iS, 'points'):
            for j in range(len(iS.points)):
                iS.points[j] = iS.points[j] / SCALE

        if i > 15 and i < 28:
            iS.x = 0
            iS.y = 0

        world[i] = iS

    game = GameCore()

    box = Box(
        interval_rate=60,
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
