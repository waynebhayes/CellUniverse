from Bacterium import *

# circle-circle collisions
def circles_collision(B, i, j, circle_i, circle_j):
    r = circle_j - circle_i
    d = sqrt(np.dot(r, r))

    # if the circles are overlapping
    if d < 2*B[i].radius:

        B[i].collided = True
        B[j].collided = True

        # amount of overlap
        x = 2*B[i].radius - d

        # compute spring force acting on bacterium i
        F = B[i].k*x*normalize(circle_i - circle_j)

        # change in momentum
        dp = F*dt
        B[i].v += dp/B[i].m

        # moment of intertia
        inertia = B[i].m*B[i].length**2/12

        # change in angular velocity
        dw = np.cross(circle_i - B[i].pos, dp)/inertia
        B[i].w += dw

        # equal and opposite force acting on bacterium j
        F = -F

        # change in momentum
        dp = F*dt
        B[j].v += dp/B[j].m
        
        # moment of intertia
        inertia = B[j].m*B[j].length**2/12

        # change in angular velocity
        dw = np.cross(circle_j - B[j].pos, dp)/inertia
        B[j].w += dw

        return True

    return False
        
# circle-line collisions
def circle_line(B, i, j, circle_i, line_j):
    x = circle_i[0]
    y = circle_i[1]
    m = line_j.m
    b = line_j.b

    # point on the line nearest the circle's center
    z = 1.0/(m**2 + 1)
    ip = np.array([(x + m*y - m*b)*z, (m*x + m**2*y + b)*z, 0])

    # distance between the circle's center and nearest point on the line
    d = abs(m*x - y + b)*sqrt(z)

    # if the circle is overlapping the line,
    #   and the nearest point is within the line segment
    if d < B[i].radius and np.dot(line_j.p1 - ip, line_j.p2 - ip) < 0:

        B[i].collided = True
        B[j].collided = True

        # the amount of overlap
        x = B[i].radius - d

        # compute spring force acting on bacterium i
        F = B[i].k*x*(line_j.normal_vector)
        
        # linear movement
        dp = F*dt
        B[i].v += dp/B[i].m

        # moment of inertia
        inertia = B[i].m*B[i].length**2/12
        
        # change in angular velocity
        dw = np.cross(circle_i - B[i].pos, dp)/inertia
        B[i].w += dw

        # equal and opposite force acting on bacterium j
        F = -F

        # change in momentum
        dp = F*dt
        B[j].v += dp/B[j].m

        return True

    return False


# bacterium moves
def move(bacterium):
    bacterium.v[0] = min(MAX_SPEED, bacterium.v[0])
    bacterium.v[1] = min(MAX_SPEED, bacterium.v[1])
    bacterium.v[0] = max(-MAX_SPEED, bacterium.v[0])
    bacterium.v[1] = max(-MAX_SPEED, bacterium.v[1])
    
    bacterium.pos += bacterium.v*dt

# bacterium spins
def spin(bacterium):
    bacterium.w[2] = min(MAX_SPIN, bacterium.w[2])
    bacterium.w[2] = max(-MAX_SPIN, bacterium.w[2])
    
    d_theta = bacterium.w[2]*dt
    bacterium.theta += d_theta


# Check for collision and move all collided bacteria 
def run(B, M):
    for p in range(3):

        collided = False

        for i in range(len(M)):
            for j in range(i+1, len(M)):

                if not M[i][j]: 
                    continue
            
                # circle-circle collisions
                collided |= circles_collision(B, i, j, B[i].head_pos, B[j].head_pos)
                collided |= circles_collision(B, i, j, B[i].head_pos, B[j].tail_pos)
                collided |= circles_collision(B, i, j, B[i].tail_pos, B[j].head_pos)
                collided |= circles_collision(B, i, j, B[i].tail_pos, B[j].tail_pos)                 

                # circle-line collisions
                collided |= circle_line(B, i, j, B[i].head_pos, B[j].line_1)
                collided |= circle_line(B, i, j, B[i].head_pos, B[j].line_2)
                collided |= circle_line(B, i, j, B[i].tail_pos, B[j].line_1)
                collided |= circle_line(B, i, j, B[i].tail_pos, B[j].line_2)
                collided |= circle_line(B, j, i, B[j].head_pos, B[i].line_1)
                collided |= circle_line(B, j, i, B[j].head_pos, B[i].line_2)
                collided |= circle_line(B, j, i, B[j].tail_pos, B[i].line_1)
                collided |= circle_line(B, j, i, B[j].tail_pos, B[i].line_2)

        if not collided: 
            break

        for bacterium in B:
            if bacterium.collided:
                move(bacterium)
                spin(bacterium)

                bacterium.update()
            
            bacterium.collided = False



# Move only bacterium i
def run2(B, i, M):
    for p in range(3):

        collided = False

        for j in range(len(B)):

            if not M[i][j]: 
                continue
            
            # circle-circle collisions
            collided |= circles_collision(B, i, j, B[i].head_pos, B[j].head_pos)
            collided |= circles_collision(B, i, j, B[i].head_pos, B[j].tail_pos)
            collided |= circles_collision(B, i, j, B[i].tail_pos, B[j].head_pos)
            collided |= circles_collision(B, i, j, B[i].tail_pos, B[j].tail_pos)                 

            # circle-line collisions
            collided |= circle_line(B, i, j, B[i].head_pos, B[j].line_1)
            collided |= circle_line(B, i, j, B[i].head_pos, B[j].line_2)
            collided |= circle_line(B, i, j, B[i].tail_pos, B[j].line_1)
            collided |= circle_line(B, i, j, B[i].tail_pos, B[j].line_2)
            collided |= circle_line(B, j, i, B[j].head_pos, B[i].line_1)
            collided |= circle_line(B, j, i, B[j].head_pos, B[i].line_2)
            collided |= circle_line(B, j, i, B[j].tail_pos, B[i].line_1)
            collided |= circle_line(B, j, i, B[j].tail_pos, B[i].line_2)

            B[j].collided = False

        if not collided: 
            break

        move(B[i])
        spin(B[i])

        B[i].update()
        B[i].collided = False
        
