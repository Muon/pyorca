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


from __future__ import division

from pyorca import Agent, get_avoidance_velocity, orca, normalized, perp
from numpy import array, rint, linspace, pi, cos, sin
import pygame

import itertools
import random

N_AGENTS = 8
RADIUS = 8.
SPEED = 10

agents = []
for i in range(N_AGENTS):
    theta = 2 * pi * i / N_AGENTS
    x = RADIUS * array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
    vel = normalized(-x) * SPEED
    pos = (random.uniform(-20, 20), random.uniform(-20, 20))
    agents.append(Agent(pos, (0., 0.), 1., SPEED, vel))


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]

pygame.init()

dim = (640, 480)
screen = pygame.display.set_mode(dim)

O = array(dim) / 2  # Screen position of origin.
scale = 6  # Drawing scale.

clock = pygame.time.Clock()
FPS = 20
dt = 1/FPS
tau = 5

def draw_agent(agent, color):
    pygame.draw.circle(screen, color, rint(agent.position * scale + O).astype(int), int(round(agent.radius * scale)), 0)

def draw_orca_circles(a, b):
    for x in linspace(0, tau, 21):
        if x == 0:
            continue
        pygame.draw.circle(screen, pygame.Color(0, 0, 255), rint((-(a.position - b.position) / x + a.position) * scale + O).astype(int), int(round((a.radius + b.radius) * scale / x)), 1)

def draw_velocity(a):
    pygame.draw.line(screen, pygame.Color(0, 255, 255), rint(a.position * scale + O).astype(int), rint((a.position + a.velocity) * scale + O).astype(int), 1)
    # pygame.draw.line(screen, pygame.Color(255, 0, 255), rint(a.position * scale + O).astype(int), rint((a.position + a.pref_velocity) * scale + O).astype(int), 1)

running = True
accum = 0
all_lines = [[]] * len(agents)
while running:
    accum += clock.tick(FPS)

    while accum >= dt * 1000:
        accum -= dt * 1000

        new_vels = [None] * len(agents)
        for i, agent in enumerate(agents):
            candidates = agents[:i] + agents[i + 1:]
            # print(candidates)
            new_vels[i], all_lines[i] = orca(agent, candidates, tau, dt)
            # print(i, agent.velocity)

        for i, agent in enumerate(agents):
            agent.velocity = new_vels[i]
            agent.position += agent.velocity * dt

    screen.fill(pygame.Color(0, 0, 0))

    for agent in agents[1:]:
        draw_orca_circles(agents[0], agent)

    for agent, color in zip(agents, itertools.cycle(colors)):
        draw_agent(agent, color)
        draw_velocity(agent)
        # print(sqrt(norm_sq(agent.velocity)))

    for line in all_lines[0]:
        # Draw ORCA line
        alpha = agents[0].position + line.point + perp(line.direction) * 100
        beta = agents[0].position + line.point + perp(line.direction) * -100
        pygame.draw.line(screen, (255, 255, 255), rint(alpha * scale + O).astype(int), rint(beta * scale + O).astype(int), 1)

        # Draw normal to ORCA line
        gamma = agents[0].position + line.point
        delta = agents[0].position + line.point + line.direction
        pygame.draw.line(screen, (255, 255, 255), rint(gamma * scale + O).astype(int), rint(delta * scale + O).astype(int), 1)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()
