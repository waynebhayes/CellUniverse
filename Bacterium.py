from __future__ import division

from math import atan2, cos, pi, sin, sqrt, tan

import numpy as np

from constants import Config, Globals


def normalize(v):
    norm = sqrt(np.dot(v, v))
    if norm == 0:
       return v
    return v / norm

def findStartAngle(x0, y0, x1, y1):
    return atan2(y1 - y0, x1 - x0) * 180 / pi

class Line:
    def __init__(self, p1 = None, p2 = None, m = None, b = None, normal_vector = None):
        self.p1 = p1
        self.p2 = p2
        self.m = m
        self.b = b
        self.normal_vector = normal_vector
        
    def astuple(self):
        return self.p1 + self.p2
    
class Bacterium:
    def __init__(self):
        # Dimensions
        self.length = Config.init_length
        self.width = Config.init_width
        self.radius = int(self.width/2)
        self.theta = 0 #for rotation
        
        # Structures
        self.line_1 = Line()
        self.line_2 = Line()
        self.bend_ratio = 0.5
        self.bend_angle = 0

        # Coordinates
        self.pos = np.array([Globals.image_width/2, Globals.image_height/2, 0])
        
        self.head_pos = np.zeros(3)
        self.tail_pos = np.zeros(3)
                
        # Characteristics
        self.name = ''
        self.collided = False
        self.w = np.zeros(3)
        self.v = np.zeros(3)
        self.grow_rate = 0
        self.f = np.zeros(3)
        self.fpos = np.zeros(3)
        self.k = 10
        self.m = 1
       
    def update(self):
        length = self.length
        pos = self.pos
        theta = self.theta
        radius = self.radius
        width = self.width
        m = tan(theta)

        # update head
        x = pos[0] - (length/2-radius)*cos(theta)
        y = pos[1] - (length/2-radius)*sin(theta)
        self.head_pos = np.array([x, y, 0])

        self.end_point_1 = np.array([x + radius*cos(theta - pi/2), y + radius*sin(theta - pi/2), 0])
        self.end_point_2 = np.array([x - radius*cos(theta - pi/2), y - radius*sin(theta - pi/2), 0])

        self.head_box = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
        self.head_start_angle = int(findStartAngle(x, y, self.end_point_2[0], self.end_point_2[1]))

        # update tail
        x = pos[0] + (length/2-radius)*cos(theta)
        y = pos[1] + (length/2-radius)*sin(theta)
        self.tail_pos = np.array([x, y, 0])

        self.end_point_3 = np.array([x - radius*cos(theta - pi/2), y - radius*sin(theta - pi/2), 0])
        self.end_point_4 = np.array([x + radius*cos(theta - pi/2), y + radius*sin(theta - pi/2), 0])

        self.tail_box = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
        self.tail_start_angle = int(findStartAngle(x, y, self.end_point_4[0], self.end_point_4[1]))

        # update body
        normal = normalize(self.end_point_1 - self.end_point_2)

        # body line 1
        b = self.end_point_1[1] - m*self.end_point_1[0]
        self.body_line_1 = np.array([m, b, 0])
        self.line_1 = Line(self.end_point_1, self.end_point_4, m, b, normal)

        # body line 2
        b = self.end_point_2[1] - m*self.end_point_2[0]
        self.body_line_2 = np.array([m, b])
        self.line_2 = Line(self.end_point_2, self.end_point_3, m, b, -normal)

        self.mid_point_1 = self.end_point_1
        self.mid_point_2 = self.end_point_2
