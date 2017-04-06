from __future__ import division
import numpy as np
from constants import *

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def two_tuple(v):
    return tuple(list(v)[:2])

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
        self.length = INIT_LENGTH
        self.width = INIT_WIDTH
        self.radius = int(self.width/2)
        self.theta = 0 #for rotation
        
        # Structures
        self.line_1 = Line()
        self.line_2 = Line()
        self.bending = False
        self.bend_ratio = 0.5
        self.bend_angle = 0

        # Coordinates
        self.pos = np.array([IMAGE_SIZE[0]/2, IMAGE_SIZE[1]/2, 0])
        
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
        self.head_pos = np.array([x,y,0])

        self.end_point_1 = np.array([x + radius*cos(theta - pi/2), y + radius*sin(theta - pi/2), 0])
        self.end_point_2 = np.array([x - radius*cos(theta - pi/2), y - radius*sin(theta - pi/2), 0])

        self.head_box = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
        self.head_start_angle = int(findStartAngle(x, y, self.end_point_2[0], self.end_point_2[1]))

        # update tail
        x = pos[0] + (length/2-radius)*cos(theta)
        y = pos[1] + (length/2-radius)*sin(theta)
        self.tail_pos = np.array([x,y, 0])

        self.end_point_3 = np.array([x - radius*cos(theta - pi/2), y - radius*sin(theta - pi/2), 0])
        self.end_point_4 = np.array([x + radius*cos(theta - pi/2), y + radius*sin(theta - pi/2), 0])

        self.tail_box = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
        self.tail_start_angle = int(findStartAngle(x, y, self.end_point_4[0], self.end_point_4[1]))

        # update body
        # body line 1
        b = self.end_point_1[1] - m*self.end_point_1[0]
        self.body_line_1 = np.array([m, b, 0])
        self.line_1 = Line(self.end_point_1, self.end_point_4, m, b, normalize(self.end_point_1 - self.end_point_2))

        # body line 2
        b = self.end_point_2[1] - m*self.end_point_2[0]
        self.body_line_2 = np.array([m, b])
        self.line_2 = Line(self.end_point_2, self.end_point_3, m, b, normalize(self.end_point_2 - self.end_point_1))

        # body box
        self.body_box = two_tuple(self.end_point_1) + two_tuple(self.end_point_2) + two_tuple(self.end_point_3) + two_tuple(self.end_point_4)
        
        self.mid_point_1 = self.end_point_1
        self.mid_point_2 = self.end_point_2

        
        # If bending
        # This bending part is not used in the simulation yet
        # because it will make the simulation way slower
        if self.bending:
            L = self.length - 2*self.radius
            L_1 = self.bend_ratio*L

            # Finding mid points 1 and 2
            v = normalize(self.end_point_4 - self.end_point_1)
            self.mid_point_1 = self.end_point_1 + L_1*v
            self.mid_point_2 = self.end_point_2 + L_1*v
            

            # Finding new tail position
            L_2 = L - L_1
            u = vector(v[0]*cos(self.bend_angle) - v[1]*sin(self.bend_angle), v[0]*sin(self.bend_angle) + v[1]*cos(self.bend_angle))
            self.tail_pos = self.head_pos + L_1*v + L_2*u

            # Finding new end points 3 and 4
            x = self.tail_pos[0]
            y = self.tail_pos[1]
            theta = self.theta + self.bend_angle
            
            self.end_point_3 = vector(x - radius*cos(theta - pi/2), y - radius*sin(theta - pi/2))
            self.end_point_4 = vector(x + radius*cos(theta - pi/2), y + radius*sin(theta - pi/2))

            # Finding new tail box and start angle
            self.tail_box = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
            self.tail_start_angle = int(findStartAngle(x, y, self.end_point_4[0], self.end_point_4[1]))



