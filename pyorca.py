# Copyright (c) 2013 Mak Nazecic-Andrlon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Implementation of the 2D ORCA algorithm as described by J. van der Berg,
S. J. Guy, M. Lin and D. Manocha in 'Reciprocal n-body Collision Avoidance'."""

from __future__ import division

import numpy
from numpy import array, sqrt, copysign, dot
from numpy.linalg import det

from halfplaneintersect import halfplane_optimize, Line, perp

# Method:
# For each robot A and potentially colliding robot B, compute smallest change
# in relative velocity 'u' that avoids collision. Find normal 'n' to VO at that
# point.
# For each such velocity 'u' and normal 'n', find half-plane as defined in (6).
# Intersect half-planes and pick velocity closest to A's preferred velocity.

class Agent(object):
    """A disk-shaped agent."""
    def __init__(self, position, velocity, radius, max_speed, pref_velocity):
        super(Agent, self).__init__()
        self.position = array(position)
        self.velocity = array(velocity)
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = array(pref_velocity)


def orca(agent, colliding_agents, t, dt):
    """Compute ORCA solution for agent. NOTE: velocity must be _instantly_
    changed on tick *edge*, like first-order integration, otherwise the method
    undercompensates and you will still risk colliding."""
    lines = []
    for collider in colliding_agents:
        dv, n = get_avoidance_velocity(agent, collider, t, dt)
        line = Line(agent.velocity + dv / 2, n)
        lines.append(line)
    return halfplane_optimize(lines, agent.pref_velocity), lines

def get_avoidance_velocity(agent, collider, t, dt):
    """Get the smallest relative change in velocity between agent and collider
    that will get them onto the boundary of each other's velocity obstacle
    (VO), and thus avert collision."""

    # This is a summary of the explanation from the AVO paper.
    #
    # The set of all relative velocities that will cause a collision within
    # time tau is called the velocity obstacle (VO). If the relative velocity
    # is outside of the VO, no collision will happen for at least tau time.
    #
    # The VO for two moving disks is a circularly truncated triangle
    # (spherically truncated cone in 3D), with an imaginary apex at the
    # origin. It can be described by a union of disks:
    #
    # Define an open disk centered at p with radius r:
    # D(p, r) := {q | ||q - p|| < r}        (1)
    #
    # Two disks will collide at time t iff ||x + vt|| < r, where x is the
    # displacement, v is the relative velocity, and r is the sum of their
    # radii.
    #
    # Divide by t:  ||x/t + v|| < r/t,
    # Rearrange: ||v - (-x/t)|| < r/t.
    #
    # By (1), this is a disk D(-x/t, r/t), and it is the set of all velocities
    # that will cause a collision at time t.
    #
    # We can now define the VO for time tau as the union of all such disks
    # D(-x/t, r/t) for 0 < t <= tau.
    #
    # Note that the displacement and radius scale _inversely_ proportionally
    # to t, generating a line of disks of increasing radius starting at -x/t.
    # This is what gives the VO its cone shape. The _closest_ velocity disk is
    # at D(-x/tau, r/tau), and this truncates the VO.

    x = -(agent.position - collider.position)
    v = agent.velocity - collider.velocity
    r = agent.radius + collider.radius

    x_len_sq = norm_sq(x)

    if x_len_sq >= r * r:
        # We need to decide whether to project onto the disk truncating the VO
        # or onto the sides.
        #
        # The center of the truncating disk doesn't mark the line between
        # projecting onto the sides or the disk, since the sides are not
        # parallel to the displacement. We need to bring it a bit closer. How
        # much closer can be worked out by similar triangles. It works out
        # that the new point is at x/t cos(theta)^2, where theta is the angle
        # of the aperture (so sin^2(theta) = (r/||x||)^2).
        adjusted_center = x/t * (1 - (r*r)/x_len_sq)

        if dot(v - adjusted_center, adjusted_center) < 0:
            # v lies in the front part of the cone
            # print("front")
            # print("front", adjusted_center, x_len_sq, r, x, t)
            w = v - x/t
            u = normalized(w) * r/t - w
            n = normalized(w)
        else: # v lies in the rest of the cone
            # print("sides")
            # Rotate x in the direction of v, to make it a side of the cone.
            # Then project v onto that, and calculate the difference.
            leg_len = sqrt(x_len_sq - r*r)
            # The sign of the sine determines which side to project on.
            sine = copysign(r, det((v, x)))
            rot = array(
                ((leg_len, sine),
                (-sine, leg_len)))
            rotated_x = rot.dot(x) / x_len_sq
                n = perp(rotated_x)
            if sine < 0:
                # Need to flip the direction of the line to make the
                # half-plane point out of the cone.
                n = -n
            # print("rotated_x=%s" % rotated_x)
            u = rotated_x * dot(v, rotated_x) - v
            # print("u=%s" % u)
    else:
        # We're already intersecting. Pick the closest velocity to our
        # velocity that will get us out of the collision within the next
        # timestep.
        # print("intersecting")
        w = v - x/dt
        u = normalized(w) * r/dt - w
        n = normalized(w)
    return u, n

def norm_sq(x):
    return dot(x, x)

def normalized(x):
    l = norm_sq(x)
    assert l > 0, (x, l)
    return x / sqrt(l)

def dist_sq(a, b):
    return norm_sq(b - a)
