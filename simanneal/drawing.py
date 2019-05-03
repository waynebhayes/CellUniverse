# -*- coding: utf-8 -*-

"""
cellanneal.drawing
~~~~~~~~~~~~~~~~~~

Contains drawing functions.
"""

import numpy as np

from math import cos, pi, sin, sqrt, floor, ceil


def _draw_line_low(array, x0, y0, x1, y1, color):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = y0

    for x in range(x0, x1 + 1):
        array[y,x] = color
        if D > 0:
            y += yi
            D -= 2*dx
        D += 2*dy


def _draw_line_high(array, x0, y0, x1, y1, color):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = x0

    for y in range(y0, y1 + 1):
        array[y,x] = color
        if D > 0:
            x += xi
            D -= 2*dy
        D += 2*dx


def draw_line(array, x0, y0, x1, y1, color):
    """Draws a line on the numpy array with the specified color."""
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            _draw_line_low(array, x1, y1, x0, y0, color)
        else:
            _draw_line_low(array, x0, y0, x1, y1, color)
    else:
        if y0 > y1:
            _draw_line_high(array, x1, y1, x0, y0, color)
        else:
            _draw_line_high(array, x0, y0, x1, y1, color)


def draw_arc(array, x, y, radius, theta0, theta1, color):
    """Draws an arc on the numpy array with the specified color."""
    num_steps = int(round(2*radius))
    if theta0 > theta1:
        theta0 -= 2*pi
    dt = (theta1 - theta0)/num_steps
    for ti in range(0, num_steps):
        t0 = theta0 + ti*dt
        t1 = t0 + dt
        x0 = int(radius*cos(t0) + x)
        y0 = int(radius*sin(t0) + y)
        x1 = int(radius*cos(t1) + x)
        y1 = int(radius*sin(t1) + y)
        draw_line(array, x0, y0, x1, y1, color)

meshgrids = {}
def circle(x, y, radius, shape):
    xl = floor(x - radius)
    xh = ceil(x + radius)
    yl = floor(y - radius)
    yh = ceil(y + radius)
    w = max(xh - xl, yh - yl)
    xh = xl + w
    yh = yl + w
    if w not in meshgrids:
        X = np.linspace(0.5, w - 0.5, w)
        X, Y = np.meshgrid(X, X)
        meshgrids[w] = (X, Y)
    else:
        X, Y = meshgrids[w]
    xx = x - xl
    yy = y - yl
    mask = ((X - xx)**2 + (Y - yy)**2) < radius**2
    full = np.zeros(shape, dtype=np.bool)
    cxl = max(xl, 0)
    cxh = min(xh, shape[1])
    cx0 = cxl - xl
    cxw = cxh - cxl
    cyl = max(yl, 0)
    cyh = min(yh, shape[0])
    cy0 = cyl - yl
    cyw = cyh - cyl
    full[cyl:cyh,cxl:cxh] = mask[cy0:cy0+cyw,cx0:cx0+cxw]
    return full


def main():
    N = 1000

    start = time()
    for _ in range(N):
        im = np.zeros((240, 320))
        mask = draw.circle(100.2, 100.7, 9.2, im.shape)
        im[mask] += 1.0
    end = (time() - start)*1000000/N
    print(f'{end} us')
    plt.imshow(im)
    plt.show()

    start = time()
    for _ in range(N):
        im = np.zeros((240, 320))
        mask = circle(100.2, 100.7, 9.2, im.shape)
        im[mask] += 1.0
    end = (time() - start)*1000000/N
    print(f'{end} us')
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    from time import time
    import numpy as np
    import skimage.draw as draw
    import matplotlib.pyplot as plt
    main()